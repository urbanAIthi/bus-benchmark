#!/usr/bin/env python3

"""
This script processes bus journey data to derive travel and dwell times.
Input files must be pre-sorted by journey identifiers.
"""

import csv
import argparse
import gzip
from tqdm import tqdm
from typing import Iterator, Optional, Tuple, Literal

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
]

TRAJECTORY_FIELDNAMES = [
    "lau",
    "date",
    "line",
    "trip",
    "geometry",
    "time",
]

# as per the KV6 spec, ONSTOP is an arrival event, repeated while the vehicle stands
# at the stop
STOP_EVENTS = {"ARRIVAL": "ARRIVAL", "ONSTOP": "ARRIVAL", "DEPARTURE": "DEPARTURE"}

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--travel-output", required=True)
parser.add_argument("--dwell-output", required=True)
parser.add_argument("--trajectory-output", required=True)
parser.add_argument("--lau", required=False, help="Only keep records for this LAU")
parser.add_argument("--stop-blacklist", required=False, help="CSV of stops to ignore")
args = parser.parse_args()


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


def trip_fields(row: dict) -> dict:
    """Columns identifying the trip a KV6 event belongs to."""
    return {
        "date": row["operatingday"],
        "line": f"{row['dataownercode']}:{row['lineplanningnumber']}",
        "trip": f"{row['dataownercode']}:{row['journeynumber']}:{row['reinforcementnumber']}",
    }


def group_visits(stop_events: list) -> list:
    """
    Groups a journey's stop events into visits, one per stop passage. A vehicle may
    arrive at and depart from the same passage more than once (KV6 table 24, footnote
    8), so all events of a passage belong to one visit until the next passage starts.
    """
    visits = []
    left = set()
    for event, row in stop_events:
        passage = (row["userstopcode"], row["passagesequencenumber"])
        if visits and visits[-1][0] == passage:
            visits[-1][1].append((event, row))
        elif passage in left:
            # a late message for a passage the vehicle has already left
            continue
        else:
            if visits:
                left.add(visits[-1][0])
            visits.append((passage, [(event, row)]))
    return [events for _, events in visits]


def visit_times(events: list) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Returns the arrival and departure event of a visit. The arrival is the first
    ARRIVAL or ONSTOP and the departure the last DEPARTURE after it: some operators
    send a DEPARTURE as the vehicle pulls up to the stop, before it has halted. A visit
    without an arrival is a pass, at the time of its first DEPARTURE.
    """
    for i, (event, row) in enumerate(events):
        if event == "ARRIVAL":
            departures = [r for e, r in events[i + 1 :] if e == "DEPARTURE"]
            return row, departures[-1] if departures else None
    return None, events[0][1]


def derive_journey_times(
    stop_events: list,
) -> Iterator[Tuple[Literal["travel", "dwell"], dict]]:
    """
    Derives travel and dwell times from the stop events of a single journey.
    """
    last_departure = None
    for events in group_visits(stop_events):
        arrival, departure = visit_times(events)
        reached = arrival if arrival is not None else departure

        # travel time from the previous stop, unless its departure is missing
        if last_departure is not None:
            yield "travel", {
                "lau": reached["lau_id"],
                **trip_fields(reached),
                "from_stop": f"{last_departure['dataownercode']}:{last_departure['userstopcode']}",
                "to_stop": f"{reached['dataownercode']}:{reached['userstopcode']}",
                "from_geometry": last_departure["geom"],
                "to_geometry": reached["geom"],
                "from_time": last_departure["timestamp"],
                "to_time": reached["timestamp"],
            }

        # dwell time, zero if the vehicle passed the stop. A lone departure only counts
        # as a pass after a departure from the previous stop: at the first stop it is
        # the start of the journey, and after a missing departure the arrival is
        # likely missing as well.
        if departure is not None and (arrival is not None or last_departure is not None):
            yield "dwell", {
                "lau": departure["lau_id"],
                **trip_fields(departure),
                "stop": f"{departure['dataownercode']}:{departure['userstopcode']}",
                "geometry": departure["geom"],
                "from_time": reached["timestamp"],
                "to_time": departure["timestamp"],
            }

        last_departure = departure


def derive_times(
    rows: Iterator[dict],
) -> Iterator[Tuple[Literal["travel", "dwell", "trajectory"], dict]]:
    """
    Derives travel times, dwell times, and trajectories from KV6 data.
    """

    journey = None
    stop_events = []
    last_timestamp = None
    last_trajectory = None

    for row in rows:
        # derive the times of a journey once all of its events have been read
        key = tuple(row[f] for f in JOURNEY_IDENTIFIERS)
        if key != journey:
            yield from derive_journey_times(stop_events)
            journey = key
            stop_events = []
            last_timestamp = None

        # emit trajectory point for every row
        traj_entry = {
            "lau": row["lau_id"],
            **trip_fields(row),
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

        # ignore all types which are not arrivals or departures
        event = STOP_EVENTS.get(row["type"])
        if event is None:
            continue

        if last_timestamp is not None and row["timestamp"] < last_timestamp:
            raise ValueError("Timestamps are not sorted")
        last_timestamp = row["timestamp"]
        stop_events.append((event, row))

    yield from derive_journey_times(stop_events)


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
