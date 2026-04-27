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

CREATE UNLOGGED TABLE kv7_usertimingpoint (
    timestamp TIMESTAMP WITH TIME ZONE,
    dataownercode VARCHAR(10),
    userstopcode VARCHAR(10),
    timingpointdataownercode VARCHAR(10),
    timingpointcode VARCHAR(10),
    getin BOOLEAN,
    getout BOOLEAN
);

CREATE UNLOGGED TABLE kv7_timingpoint (
    timestamp TIMESTAMP WITH TIME ZONE,
    dataownercode VARCHAR(10),
    timingpointcode VARCHAR(10),
    timingpointname VARCHAR(50),
    timingpointtown VARCHAR(50),
    location GEOMETRY(Point, 28992),
    stopareacode VARCHAR(10),
    location_wgs84 GEOMETRY(Point, 4326) GENERATED ALWAYS AS (ST_Transform(location, 4326)) STORED
);

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
CREATE TYPE e21_showflexibletrip AS ENUM ('TRUE', 'FALSE', 'REALTIME');
CREATE TYPE e22_monitoringerror AS ENUM ('GPS', 'GPRS', 'Radio', 'General', 'NoSystem', 'other', 'unknown');

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
    destinationcode VARCHAR(10),
    targetarrivaltime VARCHAR(8),
    targetdeparturetime VARCHAR(8),
    sidecode VARCHAR(10),
    wheelchairaccessible kv6_wheelchairaccessible,
    journeystoptype e7_journeystoptype,
    istimingstop BOOLEAN,
    productformulatype INT,
    getin BOOLEAN,
    getout BOOLEAN,
    showflexibletrip e21_showflexibletrip,
    linedestcolor VARCHAR(6),
    linedesttextcolor VARCHAR(6),
    blockcode INT,
    sequenceinblock INT,
    vehiclejourneytype e22_monitoringerror,
    quaycode VARCHAR(20),
    plannedmonitored BOOLEAN
);

CREATE INDEX kv7_localservicegrouppasstime_timestamp_idx ON kv7_localservicegrouppasstime USING btree (timestamp);
CREATE INDEX kv7_localservicegrouppasstime_dataownercode_idx ON kv7_localservicegrouppasstime USING btree (dataownercode);
CREATE INDEX kv7_localservicegrouppasstime_lineplanningnumber_idx ON kv7_localservicegrouppasstime USING btree (lineplanningnumber);
CREATE INDEX kv7_localservicegrouppasstime_journeynumber_idx ON kv7_localservicegrouppasstime USING btree (journeynumber);
