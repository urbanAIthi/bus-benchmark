import psycopg2
import configparser
import gzip
from psycopg2 import sql
from tqdm import tqdm

table = "kv6_filtered"
# output folder on the database server
output_folder = "/mnt/nvme/sql/kv6_filtered_v2"
# list of lau_ids to export
lau_ids = [
    "GM0599",
    "GM0518",
    "GM0344",
    "GM0546",
    "GM0503",
    "GM1930",
    "GM0590",
    "GM1842",
    "GM0281",
    "GM0059",
    "GM0047",
    "GM0629",
    "GM0312",
    "GM1969",
    "GM1731",
    "GM1950",
    "GM1681",
    "GM1690",
]

config = configparser.ConfigParser()
config.read("../../../config.ini")

conn = psycopg2.connect(
    host=config["database"]["Host"],
    database=config["database"]["Database"],
    user=config["database"]["User"],
    password=config["database"]["Password"],
)

cursor = conn.cursor()

query_template = sql.SQL("""
    copy (
        select id, lau_id, timestamp, type, operatingday, dataownercode, lineplanningnumber,
            journeynumber, reinforcementnumber, userstopcode, passagesequencenumber, st_astext(geom) as geom
        from {table}
        where (operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber) in (
            select distinct operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber
            from {table}
            where lau_id = %s
        )
        order by operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber, timestamp, id
    ) to stdout delimiter ',' csv header;
""").format(table=sql.Identifier(table))

for lau_id in tqdm(lau_ids):
    query = cursor.mogrify(query_template.as_string(cursor), [lau_id])
    with gzip.open(f"{output_folder}/{lau_id}.csv.gz", "wt") as f:
        cursor.copy_expert(query, f)

cursor.close()
conn.close()
