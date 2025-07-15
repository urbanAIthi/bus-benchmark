import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import psycopg2
from psycopg2 import sql
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

DB_PARAMS = {
    "host": config["database"]["Host"],
    "dbname": config["database"]["Database"],
    "user": config["database"]["User"],
    "password": config["database"]["Password"],
    "port": 5432,
}
CSV_FOLDER = "/mnt/nvme/sql/kv6_validated/travel_time_filtered/"
TABLE = "travel_time_final"
COLUMNS = [
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
]
MAX_WORKERS = 4

copy_sql = sql.SQL("""
    COPY {table} ({cols})
    FROM %s
    WITH (
      FORMAT   csv,
      DELIMITER ',',
      QUOTE    '\"',
      ENCODING 'UTF8',
      HEADER   true
    )
""").format(
    table=sql.Identifier(TABLE), cols=sql.SQL(", ").join(map(sql.Identifier, COLUMNS))
)


def import_csv(path) -> None:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    try:
        print(f"Importing {os.path.basename(path)}")
        cur.execute(copy_sql, [path])
        conn.commit()
        print(f"Imported {os.path.basename(path)}")
    finally:
        cur.close()
        conn.close()


def main() -> None:
    files = sorted(
        os.path.join(CSV_FOLDER, f)
        for f in os.listdir(CSV_FOLDER)
        if f.lower().endswith(".csv")
    )
    if not files:
        print("No CSV files found.")
        return

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(import_csv, f): f for f in files}
        for fut in as_completed(futures):
            try:
                print(fut.result())
            except Exception as e:
                print(f"Error reading {os.path.basename(futures[fut])}: {e}")


if __name__ == "__main__":
    main()
