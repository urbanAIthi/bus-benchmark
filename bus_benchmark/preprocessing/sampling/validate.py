"""
Reads in LAU data and marks invalid sections.
"""

from tqdm import tqdm
import argparse
import logging
import pandas as pd
import geopandas as gpd
from typing import Tuple

TIME_FORMAT_1 = "%Y-%m-%d %H:%M:%S%z"
TIME_FORMAT_2 = "%Y-%m-%d %H:%M:%S.%f%z"

parser = argparse.ArgumentParser()
parser.add_argument("--tt-input", type=str, required=True)
parser.add_argument("--dt-input", type=str, required=True)
parser.add_argument("--tt-output", type=str, required=True)
parser.add_argument("--dt-output", type=str, required=True)
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
            "lau": str,
            "date": str,
            "line": str,
            "trip": str,
            "from_stop": str,
            "to_stop": str,
            "from_geometry": str,
            "to_geometry": str,
            "from_time": str,
            "to_time": str,
            "valid": int,
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
            "lau": str,
            "date": str,
            "line": str,
            "trip": str,
            "stop": str,
            "geometry": str,
            "from_time": str,
            "to_time": str,
            "valid": int,
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

    tt["valid_dwell_times"] = tt["valid_dwell_times"].notnull()
    dt["valid_dwell_times"] = dt["valid_dwell_times"].notnull()

    tt = tt.drop(columns=["week"])
    dt = dt.drop(columns=["week"])

    return tt, dt


def add_route_ids(
    tt: pd.DataFrame, dt: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def get_full_tt_route(pair):
        from_stop, to_stop = zip(*pair)
        group_id = ">".join(from_stop) + ">" + to_stop[-1]
        return pd.Series(group_id, index=pair.index)

    logging.info("Gathering link tuples")
    tt["link_tuple"] = list(zip(tt["from_stop"], tt["to_stop"]))

    logging.info("Aggregating route sequences")
    tt["route"] = tt.groupby(["date", "line", "trip"])["link_tuple"].transform(
        get_full_tt_route
    )

    logging.info("Assigning route ids")
    codes, _ = pd.factorize(tt["route"], sort=True)
    tt["route_id"] = codes

    logging.info("Merging route ids into dwell times")
    dt = pd.merge(
        dt,
        tt[["date", "line", "trip", "route", "route_id"]].drop_duplicates(),
        on=["date", "line", "trip"],
        how="left",
    )

    return tt.drop(columns=["link_tuple"]), dt


tt = load_travel_times(args.tt_input)
dt = load_dwell_times(args.dt_input)

tt = add_travel_time_columns(tt)
dt = add_dwell_time_columns(dt)

tt, dt = mark_broken_dwell_times(tt, dt)
tt, dt = add_route_ids(tt, dt)

logging.info(f"Writing travel times to {args.tt_output}")
tt.to_parquet(args.tt_output)
logging.info(f"Writing dwell times to {args.dt_output}")
dt.to_parquet(args.dt_output)
