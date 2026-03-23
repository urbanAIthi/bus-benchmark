import psycopg2
import gzip
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

LAU_IDS = os.getenv("LAU_IDS", "").split(" ")
KV6_FILTERED = os.getenv("KV6_FILTERED")

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_DATABASE"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

cursor = conn.cursor()

query_template = """
    copy (
        select id, lau_id, timestamp, type, operatingday, dataownercode, lineplanningnumber,
            journeynumber, reinforcementnumber, userstopcode, passagesequencenumber, st_astext(geom) as geom
        from kv6_filtered
        where (operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber) in (
            select distinct operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber
            from kv6_filtered
            where lau_id = %s
        )
        order by operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber, timestamp, id
    ) to stdout delimiter ',' csv header;
"""

for lau_id in tqdm(LAU_IDS):
    query = cursor.mogrify(query_template, [lau_id])
    with gzip.open(f"{KV6_FILTERED}/{lau_id}.csv.gz", "wt") as f:
        cursor.copy_expert(query, f)

cursor.close()
conn.close()
