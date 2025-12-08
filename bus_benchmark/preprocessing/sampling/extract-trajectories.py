import csv
import argparse
import gzip
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--reference", required=True)
parser.add_argument("--raw", required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

reference = set()
with open(args.reference, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        reference.add((row["lau"], row["date"], row["line"], row["trip"]))

TRAJECTORIES_FIELDNAMES = [
    "lau",
    "date",
    "line",
    "trip",
    "geometry",
    "time"
]

with gzip.open(args.raw, 'rt', newline="") as infile, \
     open(args.output, "w", newline="") as travel_out:

    reader = csv.DictReader(infile)
    travel_writer = csv.DictWriter(travel_out, fieldnames=TRAJECTORIES_FIELDNAMES)

    travel_writer.writeheader()

    last_written = None

    for row in iter(tqdm(reader)):
        lau = row["lau_id"]
        date = row["operatingday"]
        line = f"{row['dataownercode']}:{row['lineplanningnumber']}"
        trip = f"{row['dataownercode']}:{row['journeynumber']}:{row['reinforcementnumber']}"

        if (lau, date, line, trip) in reference:
            curr_written = {
                "lau": lau,
                "date": date,
                "line": line,
                "trip": trip,
                "geometry": row["geom"],
                "time": row["timestamp"]
            }
            if last_written != curr_written:
                travel_writer.writerow(curr_written)
                last_written = curr_written
