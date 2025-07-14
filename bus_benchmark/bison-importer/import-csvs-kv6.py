import os
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
import psycopg2
from psycopg2 import sql
from tqdm import tqdm
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
CSV_FOLDER = "/mnt/nvme/benchmark/kv6_converted/"
TABLE = "kv6_csv"
COLUMNS = [
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
]
MAX_WORKERS = 24

copy_sql_stdin = sql.SQL("""
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
    table=sql.Identifier(TABLE), cols=sql.SQL(", ").join(map(sql.Identifier, COLUMNS))
)


def import_csv(path: str) -> None:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    try:
        opener = gzip.open if path.lower().endswith(".gz") else open
        with opener(path, mode="rt", encoding="utf-8") as f:
            cur.copy_expert(copy_sql_stdin, f)
        conn.commit()
    finally:
        cur.close()
        conn.close()


def main() -> None:
    files = sorted(
        os.path.join(CSV_FOLDER, f)
        for f in os.listdir(CSV_FOLDER)
        if f.lower().endswith((".csv", ".csv.gz"))
    )
    if not files:
        print("No files found.")
        return

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(import_csv, f): f for f in files}
        with tqdm(total=len(futures), desc="Importing files") as progress_bar:
            for fut in as_completed(futures):
                try:
                    print(fut.result())
                    progress_bar.update(1)
                except Exception as e:
                    print(f"Error reading {os.path.basename(futures[fut])}: {e}")
                    progress_bar.update(1)


if __name__ == "__main__":
    main()
