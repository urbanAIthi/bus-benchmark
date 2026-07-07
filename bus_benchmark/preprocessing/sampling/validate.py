"""
Reads in LAU data and marks invalid sections.
"""

from tqdm import tqdm
import argparse
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Tuple

TIME_FORMAT_1 = "%Y-%m-%d %H:%M:%S%z"
TIME_FORMAT_2 = "%Y-%m-%d %H:%M:%S.%f%z"

parser = argparse.ArgumentParser()
parser.add_argument("--tt-input", type=str, required=True)
parser.add_argument("--dt-input", type=str, required=True)
parser.add_argument("--traj-input", type=str, required=True)
parser.add_argument("--routes-input", type=str, required=True)
parser.add_argument("--route-dict-input", type=str, required=True)
parser.add_argument("--tt-output", type=str, required=True)
parser.add_argument("--dt-output", type=str, required=True)
parser.add_argument("--traj-output", type=str, required=True)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
tqdm.pandas()


def to_datetime_robust(column: pd.Series) -> pd.Series:
    """
    Parses timestamps in ISO 8601 with timezones and with or without milliseconds.
    """
    parsed = pd.to_datetime(column, format=TIME_FORMAT_1, utc=True, errors="coerce")
    mask = parsed.isna()
    parsed.loc[mask] = pd.to_datetime(
        column.loc[mask], format=TIME_FORMAT_2, utc=True, errors="coerce"
    )
    if parsed.isnull().any():
        raise ValueError("Could not parse all dates")
    return parsed


def load_travel_times(path: str) -> pd.DataFrame:
    logging.info(f"Loading travel times from {path}")

    df = pd.read_csv(
        path,
        dtype={
            "lau": "category",
            "date": str,
            "line": "category",
            "trip": str,
            "from_stop": "category",
            "to_stop": "category",
            "from_geometry": "category",
            "to_geometry": "category",
            "from_time": str,
            "to_time": str,
            "valid": "int8",
        },
        parse_dates=["date"],
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", utc=True)
    df["from_time"] = to_datetime_robust(df.from_time)
    df["to_time"] = to_datetime_robust(df.to_time)
    return df


def load_dwell_times(path: str) -> pd.DataFrame:
    logging.info(f"Loading dwell times from {path}")

    df = pd.read_csv(
        path,
        dtype={
            "lau": "category",
            "date": str,
            "line": "category",
            "trip": str,
            "stop": "category",
            "geometry": "category",
            "from_time": str,
            "to_time": str,
            "valid": "int8",
        },
        parse_dates=["date"],
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", utc=True)
    df["from_time"] = to_datetime_robust(df.from_time)
    df["to_time"] = to_datetime_robust(df.to_time)
    return df


def add_travel_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Adding travel time columns")

    df["travel_time"] = (df.to_time - df.from_time).dt.total_seconds()
    return df


def add_dwell_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Adding dwell time columns")

    df["dwell_time"] = (df.to_time - df.from_time).dt.total_seconds()
    return df


def mark_broken_dwell_times(
    tt: pd.DataFrame, dt: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Marking trips with broken dwell times")

    dt["week"] = dt["date"].dt.strftime("%Y-%W")
    tt["week"] = tt["date"].dt.strftime("%Y-%W")

    dt["is_zero_dt"] = dt["dwell_time"] == 0
    dt["is_small_dt"] = (dt["dwell_time"] > 0) & (dt["dwell_time"] < 5)
    group_sums = dt.groupby(["line", "week"])[["is_zero_dt", "is_small_dt"]].sum()
    group_sums["valid_dwell_times"] = (
        group_sums["is_zero_dt"] > group_sums["is_small_dt"]
    )

    tt = pd.merge(
        tt,
        group_sums["valid_dwell_times"],
        how="left",
        left_on=["line", "week"],
        right_index=True,
    )
    dt = pd.merge(
        dt,
        group_sums["valid_dwell_times"],
        how="left",
        left_on=["line", "week"],
        right_index=True,
    )

    tt["valid_dwell_times"] = tt["valid_dwell_times"].fillna(False)
    dt["valid_dwell_times"] = dt["valid_dwell_times"].fillna(False)

    tt = tt.drop(columns=["week"])
    dt = dt.drop(columns=["week"])

    return tt, dt


def stream_and_write_trajectories(
    path: str, out_path: str, route_map: pd.DataFrame
) -> None:
    logging.info(f"Processing trajectories from {path}")
    writer = None
    str_cols = ["lau", "line", "trip", "geometry", "time"]
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        chunksize=500_000,
        dtype={
            "lau": "string",
            "date": str,
            "line": "string",
            "trip": "string",
            "geometry": "string",
            "time": "string",
        },
    ):
        chunk["date"] = pd.to_datetime(chunk["date"], format="%Y-%m-%d", utc=True)
        chunk = pd.merge(chunk, route_map, on=["date", "line", "trip"], how="left")
        if chunk.empty:
            continue
        for col in str_cols:
            chunk[col] = chunk[col].astype("string")
        table = pa.Table.from_pandas(chunk)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
    if writer:
        writer.close()


def load_routes(path: str, dict_path: str) -> pd.DataFrame:
    logging.info(f"Loading routes from {path} (dict {dict_path})")
    routes = pd.read_csv(
        path,
        compression="gzip",
        dtype={"line": str, "trip": str, "route_id": "Int32"},
    )
    routes["date"] = pd.to_datetime(routes["date"], format="%Y-%m-%d", utc=True)
    route_dict = pd.read_csv(
        dict_path,
        compression="gzip",
        dtype={"route_id": "Int32", "route": "category"},
    )
    routes = pd.merge(routes, route_dict, on="route_id", how="left")
    return routes


def add_route_ids(
    tt: pd.DataFrame, dt: pd.DataFrame, routes: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Merging route ids into travel times and dwell times")
    tt = pd.merge(tt, routes, on=["date", "line", "trip"], how="left")
    dt = pd.merge(dt, routes, on=["date", "line", "trip"], how="left")
    return tt, dt


tt = load_travel_times(args.tt_input)
dt = load_dwell_times(args.dt_input)
routes = load_routes(args.routes_input, args.route_dict_input)

tt = add_travel_time_columns(tt)
dt = add_dwell_time_columns(dt)

tt, dt = mark_broken_dwell_times(tt, dt)
tt, dt = add_route_ids(tt, dt, routes)

logging.info("Merging route ids into trajectories")
stream_and_write_trajectories(args.traj_input, args.traj_output, routes)

logging.info(f"Writing travel times to {args.tt_output}")
tt.to_parquet(args.tt_output)
logging.info(f"Writing dwell times to {args.dt_output}")
dt.to_parquet(args.dt_output)
