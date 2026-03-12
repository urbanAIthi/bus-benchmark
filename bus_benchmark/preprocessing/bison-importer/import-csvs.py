"""
Bulk-import CSV (or .csv.gz) files into a PostgreSQL table using COPY.
The table name is used to look up the expected column list.
"""

import argparse
import configparser
import gzip
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import psycopg2
from psycopg2 import sql
from tqdm import tqdm

TABLE_COLUMNS: dict[str, list[str]] = {
    "kv6_csv": [
        "type",
        "dataownercode",
        "lineplanningnumber",
        "operatingday",
        "journeynumber",
        "reinforcementnumber",
        "timestamp",
        "source",
        "userstopcode",
        "passagesequencenumber",
        "vehiclenumber",
        "blockcode",
        "wheelchairaccessible",
        "numberofcoaches",
        "punctuality",
        "rd",
    ],
    "kv7_line": [
        "timestamp",
        "dataownercode",
        "lineplanningnumber",
        "linepublicnumber",
        "linename",
        "linevetagnumber",
        "transporttype",
        "linecolor",
        "linetextcolor",
    ],
    "kv7_localservicegrouppasstime": [
        "timestamp",
        "dataownercode",
        "localservicelevelcode",
        "lineplanningnumber",
        "journeynumber",
        "fortifyordernumber",
        "userstopcode",
        "userstopordernumber",
        "journeypatterncode",
        "linedirection",
        "destinationcode",
        "targetarrivaltime",
        "targetdeparturetime",
        "sidecode",
        "wheelchairaccessible",
        "journeystoptype",
        "istimingstop",
        "productformulatype",
        "getin",
        "getout",
        "showflexibletrip",
        "linedestcolor",
        "linedesttextcolor",
        "blockcode",
        "sequenceinblock",
        "vehiclejourneytype",
        "quaycode",
        "plannedmonitored",
    ],
    "kv7_timingpoint": [
        "timestamp",
        "dataownercode",
        "timingpointcode",
        "timingpointname",
        "timingpointtown",
        "location",
        "stopareacode",
    ],
    "kv7_usertimingpoint": [
        "timestamp",
        "dataownercode",
        "userstopcode",
        "timingpointdataownercode",
        "timingpointcode",
        "getin",
        "getout",
    ],
}


def _load_db_params() -> dict:
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.ini"))
    return {
        "host": config["database"]["Host"],
        "dbname": config["database"]["Database"],
        "user": config["database"]["User"],
        "password": config["database"]["Password"],
        "port": 5432,
    }


def _build_copy_sql(table: str, columns: list[str]) -> sql.Composed:
    return sql.SQL(
        """
        COPY {table} ({cols})
        FROM STDIN
        WITH (
          FORMAT   csv,
          DELIMITER ',',
          QUOTE    '"',
          ENCODING 'UTF8',
          HEADER   true
        )
        """
    ).format(
        table=sql.Identifier(table),
        cols=sql.SQL(", ").join(map(sql.Identifier, columns)),
    )


def _import_single_csv(path: str, table: str) -> None:
    """Open a (gzipped) CSV, stream it to PostgreSQL, commit, close."""
    conn = psycopg2.connect(**_load_db_params())
    cur = conn.cursor()
    try:
        opener = gzip.open if path.lower().endswith(".gz") else open
        with opener(path, mode="rt", encoding="utf-8") as f:
            cur.copy_expert(_build_copy_sql(table, TABLE_COLUMNS[table]), f)
        conn.commit()
    finally:
        cur.close()
        conn.close()


def import_dataset(table: str, csv_folder: str, max_workers: int = 24) -> None:
    """
    Import every CSV file found in csv_folder into the given table in Postgres.
    """
    if table not in TABLE_COLUMNS:
        known = ", ".join(TABLE_COLUMNS)
        raise SystemExit(
            f"Unknown table '{table}'. Known tables: {known}\n"
            "Pass a custom column list or add the table to TABLE_COLUMNS."
        )

    files = sorted(
        os.path.join(csv_folder, f)
        for f in os.listdir(csv_folder)
        if f.lower().endswith((".csv", ".csv.gz"))
    )
    if not files:
        print(f"No CSV files found in {csv_folder}")
        return

    desc = f"Importing {table}"
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_import_single_csv, f, table): f for f in files}
        with tqdm(total=len(futures), desc=desc) as pbar:
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    print(
                        f"Error importing {os.path.basename(futures[fut])}: {exc}"
                    )
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bulk-import CSV files into PostgreSQL via COPY.",
    )
    parser.add_argument(
        "--table",
        required=True,
        help=f"Target PostgreSQL table. Known tables: {', '.join(TABLE_COLUMNS)}",
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Directory containing .csv or .csv.gz files to import.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Number of parallel worker processes (default: 24).",
    )
    args = parser.parse_args()
    import_dataset(args.table, args.folder, max_workers=args.workers)
