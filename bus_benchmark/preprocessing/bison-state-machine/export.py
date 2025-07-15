import psycopg2
import configparser
from tqdm import tqdm

# DO NOT USE UNTRUSTED DATA, TABLE NAME IS NOT ESCAPED
table = "tt_trajectories"
# output folder on the database server
output_folder = "/mnt/nvme/sql/kv6_trajectories"
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
config.read("config.ini")

conn = psycopg2.connect(
    host=config["database"]["Host"],
    database=config["database"]["Database"],
    user=config["database"]["User"],
    password=config["database"]["Password"],
)

cursor = conn.cursor()

for lau_id in tqdm(lau_ids):
    query = f"""
        copy (
            select id, lau_id, timestamp, type, operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber, userstopcode, st_astext(geom) as geom
            from {table}
            where lau_id = %s
            order by operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber, timestamp, id
        ) to %s delimiter ',' csv header;
    """
    cursor.execute(query, [lau_id, f"{output_folder}/{lau_id}.csv"])

cursor.close()
conn.close()
