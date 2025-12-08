create table kv6_filtered as (
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
