CREATE TYPE kv6_type AS enum ('DELAY', 'INIT', 'ARRIVAL', 'ONSTOP', 'DEPARTURE', 'ONROUTE', 'ONPATH', 'OFFROUTE', 'END');
CREATE TYPE kv6_source AS enum ('VEHICLE', 'SERVER');
CREATE TYPE kv6_wheelchairaccessible AS enum ('ACCESSIBLE', 'NOTACCESSIBLE', 'UNKNOWN');

CREATE UNLOGGED TABLE kv6_csv (
    id BIGSERIAL PRIMARY KEY,
    type kv6_type,
    dataownercode VARCHAR(10),
    lineplanningnumber VARCHAR(10),
    operatingday DATE,
    journeynumber INT,
    reinforcementnumber INT,
    timestamp TIMESTAMP WITH TIME ZONE,
    source kv6_source,
    userstopcode VARCHAR(10),
    passagesequencenumber INT,
    vehiclenumber INT,
    blockcode INT,
    wheelchairaccessible kv6_wheelchairaccessible,
    numberofcoaches INT,
    punctuality INT,
    rd GEOMETRY(Point, 28992),
    geom GEOMETRY(Point, 4326) GENERATED ALWAYS AS (ST_Transform(rd, 4326)) STORED
);

CREATE INDEX kv6_csv_type_idx ON kv6_csv USING btree (type);
CREATE INDEX kv6_csv_timestamp_idx ON kv6_csv USING btree (timestamp);
CREATE INDEX kv6_csv_rd_idx ON kv6_csv USING gist (rd);
CREATE INDEX kv6_csv_geom_idx ON kv6_csv USING gist (geom);
CREATE INDEX kv6_csv_operatingday_idx ON kv6_csv USING btree (operatingday);
CREATE INDEX kv6_csv_lineplanningnumber_idx ON kv6_csv USING btree (lineplanningnumber);
CREATE INDEX kv6_csv_journeynumber_idx ON kv6_csv USING btree (journeynumber);
CREATE INDEX kv6_csv_dataownercode_idx ON kv6_csv USING btree (dataownercode);
CREATE INDEX kv6_csv_reinforcementnumber_idx ON kv6_csv USING btree (reinforcementnumber);

CREATE TYPE e9_transporttype AS enum ('TRAIN', 'BUS', 'METRO', 'TRAM', 'BOAT');

CREATE UNLOGGED TABLE kv7_line (
    timestamp TIMESTAMP WITH TIME ZONE,
    dataownercode VARCHAR(10),
    lineplanningnumber VARCHAR(10),
    linepublicnumber VARCHAR(4),
    linename VARCHAR(50),
    linevetagnumber INT,
    transporttype e9_transporttype,
    -- lineicon INT, -- deprecated
    linecolor VARCHAR(6),
    linetextcolor VARCHAR(6)
);

CREATE INDEX kv7_dataownercode ON kv7_line USING btree (dataownercode);
CREATE INDEX kv7_lineplanningnumber ON kv7_line USING btree (lineplanningnumber);
CREATE INDEX kv7_transporttype ON kv7_line USING btree (transporttype);

CREATE TYPE e7_journeystoptype AS ENUM ('FIRST', 'INTERMEDIATE', 'LAST');

-- The columns from ShowFlexibleTrip to VehicleJourneyType are not imported. From
-- 2023-10-24 to 2024-02-20 the OpenOV stream sends them in a different order than its
-- label line states: the ShowFlexibleTrip column holds an unlabelled 1/0, the real
-- ShowFlexibleTrip value follows one column later and the VehicleJourneyType column
-- holds a quay code. Nothing downstream uses them.
CREATE UNLOGGED TABLE kv7_localservicegrouppasstime (
    timestamp TIMESTAMP WITH TIME ZONE,
    dataownercode VARCHAR(10),
    localservicelevelcode VARCHAR(10),
    lineplanningnumber VARCHAR(10),
    journeynumber INT,
    fortifyordernumber INT,
    userstopcode VARCHAR(10),
    userstopordernumber INT,
    journeypatterncode VARCHAR(100),
    linedirection INT,
    -- V10 in the spec, but 2024-08-07 carries codes of up to 13 characters
    destinationcode VARCHAR(20),
    targetarrivaltime VARCHAR(8),
    targetdeparturetime VARCHAR(8),
    sidecode VARCHAR(10),
    wheelchairaccessible kv6_wheelchairaccessible,
    journeystoptype e7_journeystoptype,
    istimingstop BOOLEAN,
    productformulatype INT,
    getin BOOLEAN,
    getout BOOLEAN,
    quaycode VARCHAR(20),
    plannedmonitored BOOLEAN
);

CREATE INDEX kv7_localservicegrouppasstime_timestamp_idx ON kv7_localservicegrouppasstime USING btree (timestamp);
CREATE INDEX kv7_localservicegrouppasstime_dataownercode_idx ON kv7_localservicegrouppasstime USING btree (dataownercode);
CREATE INDEX kv7_localservicegrouppasstime_lineplanningnumber_idx ON kv7_localservicegrouppasstime USING btree (lineplanningnumber);
CREATE INDEX kv7_localservicegrouppasstime_journeynumber_idx ON kv7_localservicegrouppasstime USING btree (journeynumber);
CREATE INDEX kv7_localservicegrouppasstime_lookup_idx ON kv7_localservicegrouppasstime USING btree (dataownercode, lineplanningnumber, journeynumber, timestamp);

-- Which of a journey's timetable variants applies on a given operating day. Lives in the
-- KV7turbo_calendar feed rather than KV7turbo_planning. Operators publish their calendar
-- every night for roughly the next 30 days, starting at the publication day (now and then
-- a day or two before it), and smaller publications in between repeat only some service
-- levels. Later publications do reassign dates between service levels, so readers must
-- take the newest publication of a service level preceding the day in question rather
-- than the union of every publication.
CREATE UNLOGGED TABLE kv7_localservicegroupvalidity (
    timestamp TIMESTAMP WITH TIME ZONE,
    dataownercode VARCHAR(10),
    localservicelevelcode VARCHAR(10),
    operationdate DATE
);

CREATE INDEX kv7_localservicegroupvalidity_level_idx ON kv7_localservicegroupvalidity USING btree (dataownercode, localservicelevelcode, timestamp DESC);
CREATE INDEX kv7_localservicegroupvalidity_lookup_idx ON kv7_localservicegroupvalidity USING btree (dataownercode, localservicelevelcode, operationdate, timestamp DESC);
