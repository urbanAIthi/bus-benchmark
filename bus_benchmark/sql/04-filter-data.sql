create table kv6_filtered as (
    with latest_line as (
		select distinct on (dataownercode, lineplanningnumber)
            dataownercode,
            lineplanningnumber
        from kv7_line
        where transporttype = 'BUS'
        order by dataownercode, lineplanningnumber, "timestamp" desc
    )
    select
        kv6.id,
        lau.lau_id,
        kv6.timestamp,
        kv6.type,
        kv6.operatingday,
        kv6.dataownercode,
        kv6.lineplanningnumber,
        kv6.journeynumber,
        kv6.reinforcementnumber,
        kv6.userstopcode,
        kv6.geom
    from kv6_csv kv6
    join latest_line lines
        on kv6.dataownercode = lines.dataownercode
        and kv6.lineplanningnumber = lines.lineplanningnumber
    left join kv7_stop_locations stops
        on kv6.dataownercode = stops.dataownercode
        and kv6.userstopcode = stops.userstopcode
    left join lau_rg_01m_2023_4326 lau
        on st_contains(lau.geom, coalesce(kv6.geom, stops.geom))
        and lau.cntr_code = 'NL'
    where kv6.type in ('ARRIVAL', 'DEPARTURE')
    order by
        kv6.operatingday,
        kv6.dataownercode,
        kv6.lineplanningnumber,
        kv6.journeynumber,
        kv6.reinforcementnumber,
        kv6.timestamp,
        kv6.id
);

create index kv6_filtered_lau_id_idx
    on kv6_filtered using btree (lau_id);
create index kv6_filtered_dataownercode_idx
    on kv6_filtered using btree (dataownercode);
create index kv6_filtered_operatingday_idx
    on kv6_filtered using btree (operatingday);
create index kv6_filtered_geom_idx
    on kv6_filtered using gist (geom);
