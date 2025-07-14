# A benchmark dataset for bus travel and dwell time prediction

This repository contains the dataset for the paper "A Benchmark Dataset for Bus Travel and Dwell Time Prediction", to be presented at IEEE ITSC 2025.

## Abstract

The prediction of bus travel and dwell times using machine learning has been extensively studied, resulting in many different approaches. However, due to the absence of standardized benchmarks, the field currently lacks meaningful comparison of model performance. We compile and release a benchmark based on three years of automated vehicle location (AVL) data covering the Dutch public transport network. This includes data excerpts representative of different evaluation scenarios (rural, urban, small and large cities) as well as a methodology for calculating metrics in a reproducible manner.

The dataset is available on [Zenodo](https://zenodo.org/records/15839004).

## Acknowledgements

The dataset contains "Unofficial archive of travel information Dutch Public Transport" by Adriaan van Natijne, which is licensed under the Creative Commons Attribution 4.0 license.

This research was funded by the Ingolstadt public transit authority (Verkehrsverbund Großraum Ingolstadt, VGI) with funds from the German Federal Ministry of Transport (Bundesministerium für Verkehr, BMV) as part of the research project VGI newMIND.

## Regenerating dataset

First, import the LAU dataset:
(See: https://postgis.net/workshops/postgis-intro/loading_data.html)
```
wget https://gisco-services.ec.europa.eu/distribution/v2/lau/shp/LAU_RG_01M_2023_4326.shp.zip
unzip LAU_RG_01M_2023_4326.shp.zip
shp2pgsql -I -s 4326 LAU_RG_01M_2023_4326.shp lau_rg_01m_2023_4326 > sql/01-import-laus.sql
```

Then, import the KV6/KV7 data and execute all SQL files in order.

## Usage

See `notebooks/get_results.ipynb`

## CSV schema

### Common columns

* **lau**:   
    [LAU](https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/lau) which this rows geometry belongs to.
* **date** (ISO 8601):  
    Service day. This is not necessarily equal to the calendar day, see [GTFS specification](https://gtfs.org/documentation/schedule/reference/).
* **line**:  
    Bus line identifier. This usually contains, but is not necessarily equivalent to the line identifiers communicated to the passengers.
* **trip**:  
    Trip identifier. This can be used to group entries together into a contiguous sequence of stops. It is only unique for every service day.
* **route**:  
    Route identifier. This can be used to identify a unique sequence of stops across all service days.
* **outlier**:  
    Outlier or not. Identifies whether a row should be considered an outlier.

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
