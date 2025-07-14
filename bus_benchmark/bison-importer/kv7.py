from ctx import parse_ctx_message
import csv
import lzma
import gzip
from typing import Dict, Generator, Iterable, Tuple, Any

# constants for CSV fieldnames by type
FIELDNAMES_LOCALSERVICEGROUPPASSTIME = [
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
]

FIELDNAMES_USERTIMINGPOINT = [
    "timestamp",
    "dataownercode",
    "userstopcode",
    "timingpointdataownercode",
    "timingpointcode",
    "getin",
    "getout",
]

FIELDNAMES_TIMINGPOINT = [
    "timestamp",
    "dataownercode",
    "timingpointcode",
    "timingpointname",
    "timingpointtown",
    "location",
    "stopareacode",
]

FIELDNAMES_LINE = [
    "timestamp",
    "dataownercode",
    "lineplanningnumber",
    "linepublicnumber",
    "linename",
    "linevetagnumber",
    "transporttype",
    "linecolor",
    "linetextcolor",
]


def read_kv7(path: str) -> Generator[Tuple[str, str, Dict[str, Any]], None, None]:
    """
    Read a .csv.xz file and yield a (timestamp, type, entry) tuple for every event.
    """
    try:
        with lzma.open(path, "rt") as f:
            reader = csv.reader(f)
            for _, _, ctx, _ in reader:
                data = parse_ctx_message(ctx)
                if data["meta"]["label"] != "KV7turbo_planning":
                    print(f"Invalid message type: {data['meta']['label']}")
                    continue
                for table in data["tables"]:
                    for row in table["data"]:
                        yield data["meta"]["res2"], table["meta"]["name"], row
    except EOFError:
        # some files end unexpectedly, treat this as the end of the file
        print("Unexpected end of file")


def write_kv7_to_csv(
    data: Iterable[Tuple[str, str, Dict[str, Any]]], output_file: str, desired_type: str
) -> Dict[str, str]:
    """
    Write KV7 data to separate CSV files by type, compatible with PostgreSQL COPY.
    Returns dictionary of output files created.
    """
    if desired_type == "LOCALSERVICEGROUPPASSTIME":
        fieldnames = FIELDNAMES_LOCALSERVICEGROUPPASSTIME
    elif desired_type == "USERTIMINGPOINT":
        fieldnames = FIELDNAMES_USERTIMINGPOINT
    elif desired_type == "TIMINGPOINT":
        fieldnames = FIELDNAMES_TIMINGPOINT
    elif desired_type == "LINE":
        fieldnames = FIELDNAMES_LINE
    else:
        raise ValueError(f"Unknown desired type: {desired_type}")

    with gzip.open(output_file, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fieldnames)
        for timestamp, type, entry in data:
            if type != desired_type:
                continue

            if type == "LOCALSERVICEGROUPPASSTIME":
                row = [
                    timestamp,
                    entry.get("DataOwnerCode"),
                    entry.get("LocalServiceLevelCode"),
                    entry.get("LinePlanningNumber"),
                    entry.get("JourneyNumber"),
                    entry.get("FortifyOrderNumber"),
                    entry.get("UserStopCode"),
                    entry.get("UserStopOrderNumber"),
                    entry.get("JourneyPatternCode"),
                    entry.get("LineDirection"),
                    entry.get("DestinationCode"),
                    entry.get("TargetArrivalTime"),
                    entry.get("TargetDepartureTime"),
                    entry.get("SideCode"),
                    entry.get("WheelChairAccessible"),
                    entry.get("JourneyStopType"),
                    entry.get("IsTimingStop"),
                    entry.get("ProductFormulaType"),
                    entry.get("GetIn"),
                    entry.get("GetOut"),
                    entry.get("ShowFlexibleTrip"),
                    entry.get("LineDestColor"),
                    entry.get("LineDestTextColor"),
                    entry.get("BlockCode"),
                    entry.get("SequenceInBlock"),
                    None,  # entry.get('VehicleJourneyType')
                    entry.get("QuayCode"),
                    entry.get("PlannedMonitored"),
                ]
            elif type == "USERTIMINGPOINT":
                row = [
                    timestamp,
                    entry.get("DataOwnerCode"),
                    entry.get("UserStopCode"),
                    entry.get("TimingPointDataOwnerCode"),
                    entry.get("TimingPointCode"),
                    entry.get("GetIn"),
                    entry.get("GetOut"),
                ]
            elif type == "TIMINGPOINT":
                if (
                    entry.get("LocationX_EW") is not None
                    and entry.get("LocationY_NS") is not None
                ):
                    point = f"POINT({entry.get('LocationX_EW')} {entry.get('LocationY_NS')})"
                else:
                    point = None
                row = [
                    timestamp,
                    entry.get("DataOwnerCode"),
                    entry.get("TimingPointCode"),
                    entry.get("TimingPointName"),
                    entry.get("TimingPointTown"),
                    point,
                    entry.get("StopAreaCode"),
                ]
            elif type == "LINE":
                row = [
                    timestamp,
                    entry.get("DataOwnerCode"),
                    entry.get("LinePlanningNumber"),
                    entry.get("LinePublicNumber"),
                    entry.get("LineName"),
                    entry.get("LineVeTagNumber"),
                    entry.get("TransportType"),
                    entry.get("LineColor"),
                    entry.get("LineTextColor"),
                ]
            else:
                print(f"Unknown type: {type}")
                continue

            writer.writerow(row)
