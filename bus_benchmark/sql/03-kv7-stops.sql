create table public.kv7_stop_locations as (
	with latest_usertimingpoint as (
		select distinct on (dataownercode, userstopcode)
			dataownercode,
			userstopcode,
			timingpointdataownercode,
			timingpointcode
		from public.kv7_usertimingpoint
		order by dataownercode, userstopcode, "timestamp" desc
	),
	latest_timingpoint as (
		select distinct on (dataownercode, timingpointcode)
			dataownercode,
			timingpointcode,
			location
		from public.kv7_timingpoint
		order by dataownercode, timingpointcode, "timestamp" desc
	)
	select 
		ut.dataownercode,
		ut.userstopcode,
		st_transform(tp.location, 4326) as geom
	from latest_usertimingpoint ut
	join latest_timingpoint tp
		on ut.timingpointdataownercode = tp.dataownercode
		and ut.timingpointcode = tp.timingpointcode
	where tp.location is not null
);