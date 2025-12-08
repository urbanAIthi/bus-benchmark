-- by default the query planner attempts to use hash aggregation
-- which in combination with st_geometricmedian leads to an out of memory error
set enable_hashagg=off;

create table kv6_median_stop_locations as (
    select
        dataownercode,
        userstopcode,
        st_transform(
            st_geometricmedian(
                st_collect(rd)
            ),
            4326
        )::geometry(point, 4326) as geom,
        count(*) as stop_count
    from kv6_csv
    where rd is not null and type in ('ARRIVAL', 'DEPARTURE')
    group by dataownercode, userstopcode
);

alter table kv6_median_stop_locations
add primary key (dataownercode, userstopcode);

create index kv6_median_stop_locations_geom_gix
on kv6_median_stop_locations using gist (geom);
