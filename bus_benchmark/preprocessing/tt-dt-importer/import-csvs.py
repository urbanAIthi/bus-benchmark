import argparse
import gzip
import os
import psycopg2
from concurrent.futures import ProcessPoolExecutor, as_completed
from psycopg2 import sql
from dotenv import load_dotenv

load_dotenv()

DB_PARAMS = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

COLUMN_PRESETS = {
    "travel_time": [
        "lau",
        "date",
        "line",
        "trip",
        "route",
        "from_stop",
        "to_stop",
        "from_geometry",
        "to_geometry",
        "from_time",
        "to_time",
    ],
    "dwell_time": [
        "lau",
        "date",
        "line",
        "trip",
        "route",
        "stop",
        "geometry",
        "from_time",
        "to_time",
    ]
}


def create_copy_sql(table, columns):
    return sql.SQL("""
        COPY {table} ({cols})
        FROM STDIN
        WITH (
          FORMAT   csv,
          DELIMITER ',',
          QUOTE    '\"',
          ENCODING 'UTF8',
          HEADER   true
        )
    """).format(
        table=sql.Identifier(table), 
        cols=sql.SQL(", ").join(map(sql.Identifier, columns))
    )


def import_csv(path, copy_sql) -> None:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    try:
        print(f"Importing {os.path.basename(path)}")
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            cur.copy_expert(copy_sql, f)
        conn.commit()
        print(f"Imported {os.path.basename(path)}")
    finally:
        cur.close()
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", 
        required=True,
        help="Path to folder containing CSV files"
    )
    parser.add_argument(
        "--table", 
        required=True,
        help="Target database table name"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["travel_time", "dwell_time"],
        help="Column preset"
    )
    
    args = parser.parse_args()
    columns = COLUMN_PRESETS[args.mode]
    
    copy_sql = create_copy_sql(args.table, columns)
    
    files = sorted(
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if f.lower().endswith(".csv.gz")
    )
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(import_csv, f, copy_sql): f for f in files}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"Error importing {os.path.basename(futures[fut])}: {e}")


if __name__ == "__main__":
    main()
