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
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

cursor = conn.cursor()
cursor.execute("SET work_mem = '1GB'")

events_query = """
    copy (
        select kf.id, kf.lau_id, kf.timestamp, kf.type, kf.operatingday, kf.dataownercode,
            kf.lineplanningnumber, kf.journeynumber, kf.reinforcementnumber, kf.userstopcode,
            kf.passagesequencenumber, st_astext(kf.geom) as geom
        from kv6_journey_lau jl
        join kv6_filtered kf using (operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber)
        where jl.lau_id = %s
        order by kf.operatingday, kf.dataownercode, kf.lineplanningnumber, kf.journeynumber,
            kf.reinforcementnumber, kf.timestamp, kf.id
    ) to stdout delimiter ',' csv header;
"""

routes_query = """
    copy (
        select
            jr.operatingday as date,
            jr.dataownercode || ':' || jr.lineplanningnumber as line,
            jr.dataownercode || ':' || jr.journeynumber || ':' || jr.reinforcementnumber as trip,
            jr.route_id
        from kv6_journey_lau jl
        join kv7_journey_route jr using (operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber)
        where jl.lau_id = %s
    ) to stdout delimiter ',' csv header;
"""

route_dict_query = """
    copy (
        select route_id, planned_route as route
        from kv7_route_dict
        order by route_id
    ) to stdout delimiter ',' csv header;
"""

os.makedirs(f"{KV6_FILTERED}/routes", exist_ok=True)

with gzip.open(f"{KV6_FILTERED}/route_dict.csv.gz", "wt") as f:
    cursor.copy_expert(route_dict_query, f)

for lau_id in tqdm(LAU_IDS):
    query = cursor.mogrify(events_query, [lau_id])
    with gzip.open(f"{KV6_FILTERED}/{lau_id}.csv.gz", "wt") as f:
        cursor.copy_expert(query, f)

    query = cursor.mogrify(routes_query, [lau_id])
    with gzip.open(f"{KV6_FILTERED}/routes/{lau_id}.csv.gz", "wt") as f:
        cursor.copy_expert(query, f)

cursor.close()
conn.close()
