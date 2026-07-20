#!/usr/bin/env python3

"""
This script processes bus journey data to derive travel and dwell times.
Input files must be pre-sorted by journey identifiers.
"""

import csv
import argparse
import logging
import gzip
from tqdm import tqdm
from typing import Iterator, Tuple, Literal

JOURNEY_IDENTIFIERS = [
    "operatingday",
    "dataownercode",
    "lineplanningnumber",
    "journeynumber",
    "reinforcementnumber",
]

TRAVEL_TIME_FIELDNAMES = [
    "lau",
    "date",
    "line",
    "trip",
    "from_stop",
    "to_stop",
    "from_geometry",
    "to_geometry",
    "from_time",
    "to_time",
    "valid",
]

DWELL_TIME_FIELDNAMES = [
    "lau",
    "date",
    "line",
    "trip",
    "stop",
    "geometry",
    "from_time",
    "to_time",
    "valid",
]

TRAJECTORY_FIELDNAMES = [
    "lau",
    "date",
    "line",
    "trip",
    "geometry",
    "time",
]

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--travel-output", required=True)
parser.add_argument("--dwell-output", required=True)
parser.add_argument("--trajectory-output", required=True)
parser.add_argument("--lau", required=False, help="Only keep records for this LAU")
parser.add_argument("--stop-blacklist", required=False, help="CSV of stops to ignore")
args = parser.parse_args()

logging.getLogger().setLevel(logging.CRITICAL)


def load_stop_blacklist(path: str) -> set:
    """Load blacklisted stops as a set of ``dataownercode:userstopcode``."""
    if not path:
        return set()
    with open(path, newline="") as f:
        return {
            f"{row['dataownercode']}:{row['userstopcode']}"
            for row in csv.DictReader(f)
        }


STOP_BLACKLIST = load_stop_blacklist(args.stop_blacklist)


def derive_times(
    rows: Iterator[dict],
) -> Iterator[Tuple[Literal["travel", "dwell", "trajectory"], dict]]:
    """
    Derives travel times, dwell times, and trajectories from KV6 data.
    """

    last = None
    current_stop = None
    current_psn = None
    types_seen_at_stop = set()
    travel_seq = []
    dwell_seq = []
    valid_sequence = True
    error_message = None
    last_trajectory = None

    def flush_sequences() -> Iterator[Tuple[Literal["travel", "dwell"], dict]]:
        nonlocal travel_seq, dwell_seq, valid_sequence, error_message
        if not (travel_seq or dwell_seq):
            return
        if not valid_sequence:
            logging.error(
                "Skipping invalid sequence "
                f"(operatingday={travel_seq[0]['date'] if travel_seq else dwell_seq[0]['date']}, "
                f"line={travel_seq[0]['line'] if travel_seq else dwell_seq[0]['line']}, "
                f"trip={travel_seq[0]['trip'] if travel_seq else dwell_seq[0]['trip']}, "
                f"reason={error_message})"
            )
        for t in travel_seq:
            yield "travel", {**t, "valid": int(valid_sequence)}
        for d in dwell_seq:
            yield "dwell", {**d, "valid": int(valid_sequence)}
        travel_seq.clear()
        dwell_seq.clear()
        valid_sequence = True
        error_message = None

    for row in rows:
        # clear variables on journey boundary
        if last and any(row[f] != last[f] for f in JOURNEY_IDENTIFIERS):
            yield from flush_sequences()
            last = None
            current_stop = None
            current_psn = None
            types_seen_at_stop.clear()

        # clear duplicate tracker on stop boundary
        if (
            row["userstopcode"] != current_stop
            or row["passagesequencenumber"] != current_psn
        ):
            current_stop = row["userstopcode"]
            current_psn = row["passagesequencenumber"]
            types_seen_at_stop.clear()

        # emit trajectory point for every row
        traj_entry = {
            "lau": row["lau_id"],
            "date": row["operatingday"],
            "line": f"{row['dataownercode']}:{row['lineplanningnumber']}",
            "trip": f"{row['dataownercode']}:{row['journeynumber']}:{row['reinforcementnumber']}",
            "geometry": row["geom"],
            "time": row["timestamp"],
        }
        if traj_entry != last_trajectory:
            yield "trajectory", traj_entry
            last_trajectory = traj_entry

        # bridge over blacklisted stops (e.g. movable-bridge waypoints):
        # keep their trajectory GPS point but treat them as non-stop events so
        # travel/dwell segments span across them as if they were never inserted
        if f"{row['dataownercode']}:{row['userstopcode']}" in STOP_BLACKLIST:
            continue

        # ignore all types which are not handled by the state machine
        if row["type"] not in ["ARRIVAL", "DEPARTURE", "ONSTOP"]:
            continue

        # as per the KV6 spec, ONSTOP without a prior ARRIVAL
        # should be treated as an ARRIVAL
        ct = row["type"]
        if ct == "ONSTOP":
            ct = "ARRIVAL"
        if last:
            lt = last["type"]
            if lt == "ONSTOP":
                lt = "ARRIVAL"

        # skip if we already saw this type at this stop
        if ct in types_seen_at_stop:
            continue
        else:
            types_seen_at_stop.add(ct)

        if last:
            if row["timestamp"] < last["timestamp"]:
                raise ValueError("Timestamps are not sorted")

            if lt == "DEPARTURE" and ct == "ARRIVAL":
                # emit travel time > 0
                travel_seq.append(
                    {
                        "lau": row["lau_id"],
                        "date": row["operatingday"],
                        "line": f"{row['dataownercode']}:{row['lineplanningnumber']}",
                        "trip": f"{row['dataownercode']}:{row['journeynumber']}:{row['reinforcementnumber']}",
                        "from_stop": f"{last['dataownercode']}:{last['userstopcode']}",
                        "to_stop": f"{row['dataownercode']}:{row['userstopcode']}",
                        "from_geometry": last["geom"],
                        "to_geometry": row["geom"],
                        "from_time": last["timestamp"],
                        "to_time": row["timestamp"],
                    }
                )
            elif lt == "DEPARTURE" and ct == "DEPARTURE":
                # emit travel time > 0 and dwell time = 0
                travel_seq.append(
                    {
                        "lau": row["lau_id"],
                        "date": row["operatingday"],
                        "line": f"{row['dataownercode']}:{row['lineplanningnumber']}",
                        "trip": f"{row['dataownercode']}:{row['journeynumber']}:{row['reinforcementnumber']}",
                        "from_stop": f"{last['dataownercode']}:{last['userstopcode']}",
                        "to_stop": f"{row['dataownercode']}:{row['userstopcode']}",
                        "from_geometry": last["geom"],
                        "to_geometry": row["geom"],
                        "from_time": last["timestamp"],
                        "to_time": row["timestamp"],
                    }
                )
                dwell_seq.append(
                    {
                        "lau": row["lau_id"],
                        "date": row["operatingday"],
                        "line": f"{row['dataownercode']}:{row['lineplanningnumber']}",
                        "trip": f"{row['dataownercode']}:{row['journeynumber']}:{row['reinforcementnumber']}",
                        "stop": f"{row['dataownercode']}:{row['userstopcode']}",
                        "geometry": row["geom"],
                        "from_time": row["timestamp"],
                        "to_time": row["timestamp"],
                    }
                )
            elif lt == "ARRIVAL" and ct == "DEPARTURE":
                # emit dwell time > 0
                if last["userstopcode"] == row["userstopcode"]:
                    dwell_seq.append(
                        {
                            "lau": row["lau_id"],
                            "date": row["operatingday"],
                            "line": f"{row['dataownercode']}:{row['lineplanningnumber']}",
                            "trip": f"{row['dataownercode']}:{row['journeynumber']}:{row['reinforcementnumber']}",
                            "stop": f"{row['dataownercode']}:{row['userstopcode']}",
                            "geometry": row["geom"],
                            "from_time": last["timestamp"],
                            "to_time": row["timestamp"],
                        }
                    )
                else:
                    valid_sequence = False
                    error_message = "Departed from wrong stop"
            elif lt == "ARRIVAL" and ct == "ARRIVAL":
                if last["userstopcode"] == row["userstopcode"]:
                    # this is just an update, ignore
                    pass
                else:
                    valid_sequence = False
                    error_message = "Two arrivals in a row"
            else:
                valid_sequence = False
                error_message = f"Unexpected event sequence ({lt} -> {ct})"

        last = row

    if travel_seq or dwell_seq:
        yield from flush_sequences()


with (
    gzip.open(args.input, "rt") as infile,
    gzip.open(args.travel_output, "wt") as travel_out,
    gzip.open(args.dwell_output, "wt") as dwell_out,
    gzip.open(args.trajectory_output, "wt") as traj_out,
):
    reader = csv.DictReader(infile)
    travel_writer = csv.DictWriter(travel_out, fieldnames=TRAVEL_TIME_FIELDNAMES)
    dwell_writer = csv.DictWriter(dwell_out, fieldnames=DWELL_TIME_FIELDNAMES)
    traj_writer = csv.DictWriter(traj_out, fieldnames=TRAJECTORY_FIELDNAMES)

    travel_writer.writeheader()
    dwell_writer.writeheader()
    traj_writer.writeheader()

    for kind, entry in derive_times(iter(tqdm(reader))):
        if args.lau is not None and entry["lau"] != args.lau:
            continue
        if kind == "travel":
            travel_writer.writerow(entry)
        elif kind == "dwell":
            dwell_writer.writerow(entry)
        elif kind == "trajectory":
            traj_writer.writerow(entry)
