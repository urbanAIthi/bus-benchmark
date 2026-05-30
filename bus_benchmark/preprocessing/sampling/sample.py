"""
Sample LAUs for bus benchmark based on route-level coverage and trip counts
and export filtered travel-time and dwell-time CSVs for the sampled LAUs.
"""

import argparse
import os
from datetime import date
from glob import glob

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from dotenv import load_dotenv

# Configuration

load_dotenv()
TRAVEL_TIME_DIR = os.environ.get("KV6_VALIDATED", "") + "/travel_time"
DWELL_TIME_DIR = os.environ.get("KV6_VALIDATED", "") + "/dwell_time"
TRAJECTORY_DIR = os.environ.get("KV6_VALIDATED", "") + "/trajectory"

EXPORT_TRAVEL_TIMES_DIR=os.environ.get("KV6_EXPORT", "") + "/travel_time"
EXPORT_DWELL_TIMES_DIR=os.environ.get("KV6_EXPORT", "") + "/dwell_time"
EXPORT_TRAJECTORIES_DIR=os.environ.get("KV6_EXPORT", "") + "/trajectory"
NUTS_CSV_PATH = os.environ.get("NUTS_CSV_PATH", "")

DUMMY_BLACKLIST_PATH = os.environ.get("DUMMY_BLACKLIST_PATH", "")
SUMMARY_PATH = os.environ.get("SUMMARY_PATH", "")
SAMPLED_LAUS_PATH = os.environ.get("SAMPLED_LAUS_PATH", "")

LAU_WHITELIST = os.getenv("LAU_WHITELIST", "").split(" ")

MIN_COVERAGE = 0.5
MIN_TRIP_COUNT = 100
MIN_LINK_COUNT = 5
MIN_RETENTION_RATE = 0.25
LAU_CATEGORY_SAMPLE_LIMIT = 5

def get_file_name(path: str) -> str:
    name, _ = os.path.splitext(os.path.basename(path))
    return name


def iso_weeks_in_year(yr: int) -> int:
    """Number of ISO weeks in calendar year *yr*."""
    w = date(yr, 12, 31).isocalendar()[1]
    return 52 if w == 1 else w


def calc_coverage(gp: pd.DataFrame) -> float:
    """Minimum weekly coverage across 2022-2024."""
    covs = []
    for yr in (2022, 2023, 2024):
        weeks_seen = gp.loc[gp["year"] == yr, "week"].nunique()
        covs.append(weeks_seen / iso_weeks_in_year(yr))
    return min(covs)


def load_dummy_stops() -> pd.Series:
    dummy = pd.read_csv(DUMMY_BLACKLIST_PATH, dtype=str)
    return dummy["dataownercode"] + ":" + dummy["userstopcode"]


def route_has_dummy_stop(row: pd.Series, dummy_stops: pd.Series) -> bool:
    stops = row["route"].split(">")
    return dummy_stops.isin(stops).any()


def clean_for_output(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["from_time"] = df["from_time"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df["to_time"] = df["to_time"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df["valid"] = (df["valid"] & df["valid_dwell_times"]).astype("int8")
    df["route"] = df["route_id"]
    if "from_geometry" in df.columns:
        return df[
            ["lau", "date", "line", "trip", "route", "from_stop", "to_stop",
             "from_geometry", "to_geometry", "from_time", "to_time"]
        ]
    else:
        return df[
            ["lau", "date", "line", "trip", "route", "stop", "geometry",
             "from_time", "to_time"]
        ]


def get_stratum(row: pd.Series) -> str:
    if row["DEGURBA"] == 1 and row["POPULATION"] > 250000:
        return "1.0"
    elif row["DEGURBA"] == 1:
        return "1.1"
    else:
        return str(row["DEGURBA"])


# Stage 1: Summarize

def summarize_trips(path: str) -> pd.DataFrame:
    """Compute route-level summary statistics for a single LAU parquet file."""
    df_uncleaned = pd.read_parquet(path)
    df_uncleaned["has_geometry"] = (
        df_uncleaned["from_geometry"].notna() & df_uncleaned["to_geometry"].notna()
    )

    df = df_uncleaned[
        df_uncleaned["valid_dwell_times"]
        & df_uncleaned["from_geometry"].notna()
        & df_uncleaned["to_geometry"].notna()
    ]
    has_geometry_mask = df.groupby(
        ["line", "route", "route_id", "trip", "date"]
    )["has_geometry"].transform("all")
    df = df.loc[has_geometry_mask].copy()

    if df.empty:
        return pd.DataFrame(
            {"lau": [get_file_name(path)], "pct_valid": [0], "line_pct_valid": [0]}
        )

    df["year"] = df["date"].dt.year
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    lau_coverage = calc_coverage(df[["year", "week"]].drop_duplicates())

    line_coverage = (
        df[["line", "year", "week"]]
        .drop_duplicates()
        .groupby(["line"])
        .apply(calc_coverage)
        .reset_index(name="line_coverage")
    )

    link_counts = df[["line", "route", "route_id"]].drop_duplicates()
    link_counts["link_count"] = link_counts["route"].str.count(">") + 1

    trip_counts = df[["date", "line", "route", "route_id", "trip"]].drop_duplicates()
    trip_counts = (
        trip_counts.groupby(["line", "route", "route_id"])
        .apply(lambda g: g.drop_duplicates(subset=["date", "trip"]).shape[0])
        .reset_index(name="trip_count")
    )

    route_info = (
        line_coverage
        .merge(link_counts, on=["line"])
        .merge(trip_counts, on=["route", "route_id"])
    )

    total_trip_count = df[["date", "line", "route", "route_id", "trip"]].drop_duplicates()
    total_trip_count = total_trip_count.drop_duplicates(
        subset=["line", "date", "trip"]
    ).shape[0]

    uncleaned_total_trip_count = df_uncleaned[
        ["date", "line", "route", "route_id", "trip"]
    ].drop_duplicates()
    uncleaned_total_trip_count = uncleaned_total_trip_count.drop_duplicates(
        subset=["line", "date", "trip"]
    ).shape[0]

    route_info["lau"] = get_file_name(path)
    route_info["lau_coverage"] = lau_coverage
    route_info["total_trip_count"] = total_trip_count
    route_info["uncleaned_total_trip_count"] = uncleaned_total_trip_count

    return route_info


def run_summarize() -> pd.DataFrame:
    """Stage 1: build route-level summary from raw travel-time parquet files."""
    print("=== Stage 1: Summarize trips ===")
    paths = glob(os.path.join(TRAVEL_TIME_DIR, "*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {TRAVEL_TIME_DIR}")

    counts = [summarize_trips(p) for p in tqdm(paths, desc="Summarizing")]
    summary_df = pd.concat(counts, ignore_index=True)
    summary_df.to_parquet(SUMMARY_PATH)
    print(f"Saved summary to {SUMMARY_PATH}  ({len(summary_df)} rows)")
    return summary_df


# Stage 2: Sample

def run_sample(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Stage 2: stratified sampling of LAUs."""
    print("\n=== Stage 2: Sample LAUs ===")

    nuts = pd.read_csv(NUTS_CSV_PATH, sep=";")
    nuts["DEGURBA_EXT"] = nuts.apply(get_stratum, axis=1)

    dummy_stops = load_dummy_stops()

    tmp = summary_df[
        (summary_df["lau_coverage"] >= MIN_COVERAGE)
        & (summary_df["line_coverage"] >= MIN_COVERAGE)
        & (summary_df["trip_count"] >= MIN_TRIP_COUNT)
        & (summary_df["link_count"] >= MIN_LINK_COUNT)
    ]
    tmp = tmp[~tmp.apply(route_has_dummy_stop, axis=1, dummy_stops=dummy_stops)]
    tmp["retention_rate"] = (
        tmp.groupby("lau")["trip_count"].transform("sum")
        / tmp["uncleaned_total_trip_count"]
    )
    tmp = tmp[tmp["retention_rate"] > MIN_RETENTION_RATE].groupby("lau").agg(
        {"trip_count": "sum"}
    )

    df_merged = pd.merge(
        nuts, tmp, left_on="LAU CODE", right_on="lau", how="inner"
    )

    if LAU_WHITELIST:
        df_merged = df_merged[df_merged["LAU CODE"].isin(LAU_WHITELIST)]

    sampled_laus = []
    for degurba, group in df_merged.groupby("DEGURBA_EXT"):
        if degurba == "1.0" or LAU_CATEGORY_SAMPLE_LIMIT is None:
            sampled = group
        else:
            sampled = group.sample(LAU_CATEGORY_SAMPLE_LIMIT)
        sampled = sampled.sort_values(
            ["DEGURBA_EXT", "POPULATION"], ascending=[True, False]
        )
        print(f"  Stratum {degurba}: sampled {len(sampled)} / {len(group)}, "
              f"trip_count={sampled['trip_count'].sum()}")
        sampled_laus.append(sampled)

    sampled_laus_df = pd.concat(sampled_laus)
    sampled_laus_df.to_csv(SAMPLED_LAUS_PATH, index=False)
    print(f"Saved sampled LAUs to {SAMPLED_LAUS_PATH}  ({len(sampled_laus_df)} rows)")
    return sampled_laus_df


# Stage 3: Export

def run_export(
    summary_df: pd.DataFrame, sampled_laus_df: pd.DataFrame
) -> None:
    """Stage 3: filter & export travel-time and dwell-time CSVs."""
    print("\n=== Stage 3: Export filtered data ===")

    os.makedirs(EXPORT_TRAVEL_TIMES_DIR, exist_ok=True)
    os.makedirs(EXPORT_DWELL_TIMES_DIR, exist_ok=True)
    os.makedirs(EXPORT_TRAJECTORIES_DIR, exist_ok=True)

    dummy_stops = load_dummy_stops()

    lau_codes = sampled_laus_df["LAU CODE"].drop_duplicates().iloc[::-1]

    for lau in tqdm(lau_codes, desc="Exporting LAUs"):
        # Determine valid routes for this LAU
        routes = summary_df[
            (summary_df["lau"] == lau)
            & (summary_df["lau_coverage"] >= MIN_COVERAGE)
            & (summary_df["line_coverage"] >= MIN_COVERAGE)
            & (summary_df["trip_count"] >= MIN_TRIP_COUNT)
            & (summary_df["link_count"] >= MIN_LINK_COUNT)
        ]
        routes = routes[
            ~routes.apply(route_has_dummy_stop, axis=1, dummy_stops=dummy_stops)
        ]
        routes = routes[["route", "route_id"]].drop_duplicates()

        # Travel times
        tt_path = os.path.join(TRAVEL_TIME_DIR, f"{lau}.parquet")
        tt = pd.read_parquet(tt_path)
        tt["has_geometry"] = tt["from_geometry"].notna() & tt["to_geometry"].notna()
        tt = tt[tt["valid_dwell_times"]]
        has_geo = tt.groupby(
            ["line", "route", "route_id", "trip", "date"]
        )["has_geometry"].transform("all")
        tt = tt.loc[has_geo]
        tt = pd.merge(tt, routes, on=["route", "route_id"], how="inner")
        valid_trips = tt[["date", "line", "trip"]].drop_duplicates()
        tt = clean_for_output(tt)
        tt_out = os.path.join(EXPORT_TRAVEL_TIMES_DIR, f"{lau}.csv")
        tt.to_csv(tt_out, index=False)

        # Dwell times
        dt_path = os.path.join(DWELL_TIME_DIR, f"{lau}.parquet")
        dt = pd.read_parquet(dt_path)
        dt["has_geometry"] = dt["geometry"].notna()
        dt = dt[dt["valid_dwell_times"] & dt["route_id"].notna()]
        has_geo = dt.groupby(
            ["line", "route", "route_id", "trip", "date"]
        )["has_geometry"].transform("all")
        dt = dt.loc[has_geo]
        dt = pd.merge(dt, routes, on=["route", "route_id"], how="inner")
        dt = clean_for_output(dt)
        dt_out = os.path.join(EXPORT_DWELL_TIMES_DIR, f"{lau}.csv")
        dt.to_csv(dt_out, index=False)

        # Trajectories
        traj_path = os.path.join(TRAJECTORY_DIR, f"{lau}.parquet")
        traj_out = os.path.join(EXPORT_TRAJECTORIES_DIR, f"{lau}.csv")
        pf = pq.ParquetFile(traj_path)
        first_chunk = True
        for batch in pf.iter_batches(batch_size=500_000):
            chunk = batch.to_pandas()
            chunk = pd.merge(chunk, valid_trips, on=["date", "line", "trip"], how="inner")
            chunk = pd.merge(chunk, routes, on=["route", "route_id"], how="inner")
            if chunk.empty:
                continue
            chunk["date"] = chunk["date"].dt.strftime("%Y-%m-%d")
            chunk["route"] = chunk["route_id"]
            chunk = chunk[["lau", "date", "line", "trip", "route", "geometry", "time"]]
            chunk.to_csv(traj_out, index=False, mode="w" if first_chunk else "a", header=first_chunk)
            first_chunk = False

    print("Export complete.")

# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Sampling pipeline for bus benchmark data."
    )
    parser.add_argument(
        "--summarize", action="store_true",
        help="Stage 1: build route-level summary from parquet files.",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Stage 2: stratified sampling of LAUs.",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Stage 3: export filtered travel-time & dwell-time CSVs.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all three stages sequentially.",
    )
    args = parser.parse_args()

    run_all = args.all or not (args.summarize or args.sample or args.export)

    summary_df = None
    sampled_laus_df = None

    if run_all or args.summarize:
        summary_df = run_summarize()

    if run_all or args.sample:
        if summary_df is None:
            print(f"Loading summary from {SUMMARY_PATH}")
            summary_df = pd.read_parquet(SUMMARY_PATH)
        sampled_laus_df = run_sample(summary_df)

    if run_all or args.export:
        if summary_df is None:
            print(f"Loading summary from {SUMMARY_PATH}")
            summary_df = pd.read_parquet(SUMMARY_PATH)
        if sampled_laus_df is None:
            print(f"Loading sampled LAUs from {SAMPLED_LAUS_PATH}")
            sampled_laus_df = pd.read_csv(SAMPLED_LAUS_PATH)
        run_export(summary_df, sampled_laus_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
