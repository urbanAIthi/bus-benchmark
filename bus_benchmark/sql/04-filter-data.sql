create unlogged table kv6_filtered as (
    with lines as (
		select distinct dataownercode, lineplanningnumber
        from kv7_line
        where transporttype = 'BUS'
    ), stops as (
        select stops.dataownercode, stops.userstopcode, stops.geom, lau.lau_id
        from kv6_median_stop_locations stops
        join lau_rg_01m_2023_4326 lau
        on st_intersects(stops.geom, lau.geom)
        where cntr_code = 'NL'
    )
    select
        kv6.id,
        stops.lau_id,
        kv6.timestamp,
        kv6.type,
        kv6.operatingday,
        kv6.dataownercode,
        kv6.lineplanningnumber,
        kv6.journeynumber,
        kv6.reinforcementnumber,
        kv6.userstopcode,
        kv6.passagesequencenumber,
        kv6.geom,
        st_distance(kv6.geom::geography, stops.geom::geography) as distance_from_stop
    from lines
    join kv6_csv kv6 using (dataownercode, lineplanningnumber)
    join stops using (dataownercode, userstopcode)
);

create index kv6_filtered_lau_id_idx
    on kv6_filtered using btree (lau_id);
create index kv6_filtered_dataownercode_lineplanningnumber_idx
    on kv6_filtered using btree (dataownercode, lineplanningnumber);
create index kv6_filtered_dataownercode_userstopcode_idx
    on kv6_filtered using btree (dataownercode, userstopcode);
create index kv6_filtered_operatingday_idx
    on kv6_filtered using btree (operatingday);
create index kv6_filtered_geom_idx
    on kv6_filtered using gist (geom);
create index kv6_filtered_journey_key_idx
    on kv6_filtered using btree (operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber);

create unlogged table kv6_journey_lau as
select distinct
    operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber, lau_id
from kv6_filtered;

create index kv6_journey_lau_lau_id_idx
    on kv6_journey_lau using btree (lau_id);
create index kv6_journey_lau_journey_key_idx
    on kv6_journey_lau using btree (operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber);

create unlogged table kv6_journey_first_ts as
select
    operatingday,
    dataownercode,
    lineplanningnumber,
    journeynumber,
    reinforcementnumber,
    min(timestamp) as first_kv6_ts
from kv6_filtered
group by operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber;

create index kv6_journey_first_ts_lookup_idx
    on kv6_journey_first_ts using btree (dataownercode, lineplanningnumber, journeynumber);

create unlogged table kv7_journey_snapshot as
select
    jf.operatingday,
    jf.dataownercode,
    jf.lineplanningnumber,
    jf.journeynumber,
    jf.reinforcementnumber,
    ks.snapshot_ts
from kv6_journey_first_ts jf
join lateral (
    select kv7.timestamp as snapshot_ts
    from kv7_localservicegrouppasstime kv7
    where kv7.dataownercode = jf.dataownercode
      and kv7.lineplanningnumber = jf.lineplanningnumber
      and kv7.journeynumber = jf.journeynumber
      and kv7.timestamp < jf.first_kv6_ts
    order by kv7.timestamp desc
    limit 1
) ks on true;

create index kv7_journey_snapshot_lookup_idx
    on kv7_journey_snapshot using btree (dataownercode, lineplanningnumber, journeynumber, snapshot_ts);

create unlogged table kv7_planned_route as
with kv7_dedup as (
    select distinct
        kv7.dataownercode,
        kv7.lineplanningnumber,
        kv7.journeynumber,
        kv7.timestamp,
        kv7.userstopordernumber,
        kv7.userstopcode
    from kv7_localservicegrouppasstime kv7
    join (
        select distinct dataownercode, lineplanningnumber, journeynumber, snapshot_ts
        from kv7_journey_snapshot
    ) used
        on kv7.dataownercode = used.dataownercode
        and kv7.lineplanningnumber = used.lineplanningnumber
        and kv7.journeynumber = used.journeynumber
        and kv7.timestamp = used.snapshot_ts
)
select
    dataownercode,
    lineplanningnumber,
    journeynumber,
    timestamp as snapshot_ts,
    string_agg(
        dataownercode || ':' || userstopcode,
        '>'
        order by userstopordernumber
    ) as planned_route
from kv7_dedup
group by dataownercode, lineplanningnumber, journeynumber, timestamp;

create index kv7_planned_route_lookup_idx
    on kv7_planned_route using btree (dataownercode, lineplanningnumber, journeynumber, snapshot_ts);

create unlogged table kv7_route_dict as
select
    (row_number() over (order by planned_route) - 1)::int as route_id,
    planned_route
from (
    select distinct planned_route
    from kv7_planned_route
    where planned_route is not null
) d;

create index kv7_route_dict_route_id_idx
    on kv7_route_dict using btree (route_id);
create index kv7_route_dict_planned_route_idx
    on kv7_route_dict using btree (planned_route);

create unlogged table kv7_journey_route as
select
    s.operatingday,
    s.dataownercode,
    s.lineplanningnumber,
    s.journeynumber,
    s.reinforcementnumber,
    d.route_id
from kv7_journey_snapshot s
join kv7_planned_route r
    on r.dataownercode = s.dataownercode
    and r.lineplanningnumber = s.lineplanningnumber
    and r.journeynumber = s.journeynumber
    and r.snapshot_ts = s.snapshot_ts
join kv7_route_dict d
    on d.planned_route = r.planned_route;

create index kv7_journey_route_key_idx
    on kv7_journey_route using btree (operatingday, dataownercode, lineplanningnumber, journeynumber, reinforcementnumber);
