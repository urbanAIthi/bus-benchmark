"""
Import KV6/KV7 data from trein.fwrite.org and output to CSV files for PostgreSQL COPY.
This creates one CSV file for KV6 data or multiple CSV files for KV7 data (by table type).
"""

from kv6 import read_kv6, write_kv6_to_csv
from kv7 import read_kv7, write_kv7_to_csv
from tqdm import tqdm
import argparse
import os
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, help="Path to .csv.xz file")
parser.add_argument(
    "--mode", choices=["kv6", "kv7"], required=True, help="Interface type"
)
parser.add_argument("--filters", nargs="+", help="Only import these message types")
parser.add_argument("--progress", action="store_true", help="Show progress bar")
parser.add_argument(
    "--output-dir", default="./output", help="Directory for CSV output files"
)
args = parser.parse_args()


def filter_kv6(data, filters):
    if not filters:
        yield from data
        return
    for entry in data:
        if entry["type"] in filters:
            yield entry


def filter_kv7(data, filters):
    if not filters:
        yield from data
        return
    for timestamp, type, entry in data:
        if type in filters:
            yield timestamp, type, entry


if args.mode == "kv6":
    input_filename = os.path.basename(args.file)
    base_name = os.path.splitext(os.path.splitext(input_filename)[0])[0]
    output_file = os.path.join(args.output_dir, base_name + ".csv.gz")

    write_kv6_to_csv(
        tqdm(filter_kv6(read_kv6(args.file), args.filters), disable=not args.progress),
        output_file,
    )
    print(f"KV6 data written to: {output_file}")
elif args.mode == "kv7":
    input_filename = os.path.basename(args.file)
    base_name = os.path.splitext(os.path.splitext(input_filename)[0])[0]
    output_dir = os.path.join(args.output_dir, base_name)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    for table_type in ["USERTIMINGPOINT", "TIMINGPOINT", "LINE"]:
        output_file = os.path.join(output_dir, f"{base_name}_{table_type}.csv.gz")
        write_kv7_to_csv(
            tqdm(
                filter_kv7(read_kv7(args.file), args.filters), disable=not args.progress
            ),
            output_file,
            table_type,
        )
        print(f"KV7 {table_type} data written to: {output_file}")
