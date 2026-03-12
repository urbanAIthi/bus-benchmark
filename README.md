# A benchmark dataset for bus travel and dwell time prediction

This repository contains the dataset for the paper "A Benchmark Dataset for Bus Travel and Dwell Time Prediction", written by Alexander Horn, Philip-Roman Adam and Stefanie Schmidtner and presented at IEEE ITSC 2025. The `v1` branch contains the code used to generate the version of the dataset ([10.5281/zenodo.15839004](https://doi.org/10.5281/zenodo.15839004)) that was published along the paper.

## Abstract

The prediction of bus travel and dwell times using machine learning has been extensively studied, resulting in many different approaches. However, due to the absence of standardized benchmarks, the field currently lacks meaningful comparison of model performance. We compile and release a benchmark based on three years of automated vehicle location (AVL) data covering the Dutch public transport network. This includes data excerpts representative of different evaluation scenarios (rural, urban, small and large cities) as well as a methodology for calculating metrics in a reproducible manner. The dataset is available on Zenodo ([10.5281/zenodo.15839003](https://doi.org/10.5281/zenodo.15839003)).

## Acknowledgements

The dataset contains ["Unofficial archive of travel information Dutch Public Transport"](https://trein.fwrite.org/) by Adriaan van Natijne, which is licensed under the Creative Commons Attribution 4.0 license.

This research was funded by the Ingolstadt public transit authority (Verkehrsverbund Großraum Ingolstadt, VGI) with funds from the German Federal Ministry of Transport (Bundesministerium für Verkehr, BMV) as part of the research project VGI newMIND.

## Generating dataset

Generate the LAU table:
```
wget https://gisco-services.ec.europa.eu/distribution/v2/lau/shp/LAU_RG_01M_2023_4326.shp.zip
unzip LAU_RG_01M_2023_4326.shp.zip
shp2pgsql -I -s 4326 LAU_RG_01M_2023_4326.shp lau_rg_01m_2023_4326 > sql/01-import-laus.sql
```

Run the following scripts in order:
* Convert XML/CTX files to CSV:  
`bus_benchmark/preprocessing/bison-importer/import-*-bulk.sh`
* Import CSV files into PostgreSQL:  
`bus_benchmark/preprocessing/bison-importer/import-csvs-*.py`
* Match tables with LAUs:  
`sql/*.sql`
* Export PostgreSQL tables to CSV files:  
`bus_benchmark/preprocessing/bison-state-machine/export.py`
* Convert KV6 logs into travel and dwell times:  
`bus_benchmark/preprocessing/bison-state-machine/process-all.sh`
* Validate travel and dwell times:  
`bus_benchmark/preprocessing/sampling/validate-all.sh`
* Sample LAUs and export final travel and dwell time CSV files:  
`notebooks/sampling.ipynb`

## CSV schema

### Common columns

* **lau**:   
    [LAU](https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/local-administrative-units) which this rows geometry belongs to.
* **date** (ISO 8601):  
    Service day. This is not necessarily equal to the calendar day, see [GTFS specification](https://gtfs.org/documentation/schedule/reference/).
* **line**:  
    Bus line identifier. This usually contains, but is not necessarily equivalent to the line identifiers communicated to the passengers.
* **trip**:  
    Trip identifier. This can be used to group entries together into a contiguous sequence of stops. It is only unique for every service day.
* **route**:  
    Route identifier. This can be used to identify a unique sequence of stops across all service days.

### Travel times

All common columns, plus:
* **from_stop**, **to_stop**:  
    Origin and destination stop identifier. This identifies a unique bus stop. Separate platforms or multiple identically named stop points are considered to be separate stops.  
* **from_geometry**, **to_geometry** (WKT):  
    Origin and destination coordinates of the bus stops. The used coordinate system is WGS 84 (EPSG:4326).
* **from_time**, **to_time** (ISO 8601):  
    Departure and arrival time stamps. This can be used to calculate the travel time.

### Dwell times

All common columns, plus:
* **stop**:  
    Stop identifier. The definition is the same as above.  
* **geometry** (WKT):  
    Coordinates of the bus stop. The definition is the same as above.
* **from_time**, **to_time** (ISO 8601):  
    Arrival and departure timestamps. This can be used to calculate the dwell time.

### Trajectories

All common columns, plus:
* **geometry** (WKT):  
    Coordinates of the bus at a particular moment.
* **time** (ISO 8601):  
    Timestamp of this record.
