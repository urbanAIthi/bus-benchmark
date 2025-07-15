from lxml import etree
import csv
import lzma
import sys
import gzip

ns = "{http://bison.connekt.nl/tmi8/kv6/msg}"
parser = etree.XMLParser(recover=True)
csv.field_size_limit(sys.maxsize)


def read_kv6(path):
    """
    Read a .csv.xz file and yield a dict for each event.
    """
    try:
        with lzma.open(path, "rt") as f:
            reader = csv.reader(f)
            for _, _, xml, uuid in reader:
                try:
                    # the parser wants bytes but the csv reader returns strings
                    # convert to bytes and just assume it was utf-8 before
                    root = etree.fromstring(xml.encode("utf-8"), parser=parser)
                    posinfo = root.find(f"{ns}KV6posinfo")
                    if posinfo is None:
                        continue
                    for event in posinfo:
                        data = {"type": event.tag.replace(ns, "")}
                        for child in event:
                            data[child.tag.replace(ns, "")] = child.text
                        yield data
                except etree.XMLSyntaxError:
                    # some messages contain invalid XML, skip them
                    print("Invalid XML")
    except EOFError:
        # some files end unexpectedly, treat this as the end of the file
        print("Unexpected end of file")
    except lzma.LZMAError:
        print("Input data is corrupt")


def insert_kv6(conn, data_iter, chunk_size=100_000):
    """
    Insert data into the database in bulk using executemany.
    """
    insert_sql = """
        INSERT INTO kv6 (
            type, dataownercode, lineplanningnumber, operatingday, journeynumber,
            reinforcementnumber, timestamp, source, userstopcode, passagesequencenumber,
            vehiclenumber, blockcode, wheelchairaccessible, numberofcoaches, punctuality, rd
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    buffer = []
    with conn.cursor() as cursor:
        for entry in data_iter:
            # build geometry if we have valid rd-x and rd-y
            if entry.get("rd-x") not in (None, "-1") and entry.get("rd-y") not in (
                None,
                "-1",
            ):
                point = f"POINT({entry['rd-x']} {entry['rd-y']})"
            else:
                point = None

            values = (
                entry.get("type"),
                entry.get("dataownercode"),
                entry.get("lineplanningnumber"),
                entry.get("operatingday"),
                entry.get("journeynumber"),
                entry.get("reinforcementnumber"),
                entry.get("timestamp"),
                entry.get("source"),
                entry.get("userstopcode"),
                entry.get("passagesequencenumber"),
                entry.get("vehiclenumber"),
                entry.get("blockcode"),
                entry.get("wheelchairaccessible"),
                entry.get("numberofcoaches"),
                entry.get("punctuality"),
                point,
            )

            buffer.append(values)

            # once we hit chunk_size, do a bulk insert
            if len(buffer) >= chunk_size:
                cursor.executemany(insert_sql, buffer)
                buffer.clear()

        # insert any leftover rows
        if buffer:
            cursor.executemany(insert_sql, buffer)

    # commit once at the end, or after each chunk if you prefer
    conn.commit()


def write_kv6_to_csv(data_iter, output_file):
    """
    Write KV6 data to a CSV file compatible with PostgreSQL COPY.
    """
    fieldnames = [
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

    with gzip.open(output_file, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fieldnames)

        for entry in data_iter:
            # build geometry if we have valid rd-x and rd-y
            if entry.get("rd-x") not in (None, "-1") and entry.get("rd-y") not in (
                None,
                "-1",
            ):
                point = f"POINT({entry['rd-x']} {entry['rd-y']})"
            else:
                point = None

            row = [
                entry.get("type"),
                entry.get("dataownercode"),
                entry.get("lineplanningnumber"),
                entry.get("operatingday"),
                entry.get("journeynumber"),
                entry.get("reinforcementnumber"),
                entry.get("timestamp"),
                entry.get("source"),
                entry.get("userstopcode"),
                entry.get("passagesequencenumber"),
                entry.get("vehiclenumber"),
                entry.get("blockcode"),
                entry.get("wheelchairaccessible"),
                entry.get("numberofcoaches"),
                entry.get("punctuality"),
                point,
            ]

            writer.writerow(row)
