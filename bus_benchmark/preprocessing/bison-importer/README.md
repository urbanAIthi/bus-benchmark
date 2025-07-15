# BISON Importer

Imports BISON public transport data into a PostgreSQL database.

## Supported data

### KV6 Current trip punctuality and vehicle information

Specification: https://bison.dova.nu/standaarden/kv6-actuele-ritpunctualiteit-en-voertuiginformatie

Download: https://trein.fwrite.org/idx/dedup_KV6posinfo.html

### KV78turbo Efficient travel information at stop level

Specification: https://bison.dova.nu/standaarden/kv78turbo-efficiente-reisinformatie-op-halteniveau

Download: https://trein.fwrite.org/idx/dedup_KV7planning.html

## Usage

Add database credentials to `config.ini`.

```
usage: import.py [-h] --file FILE --mode {kv6,kv7} [--filters FILTERS [FILTERS ...]] [--progress]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Path to .csv.xz file
  --mode {kv6,kv7}      Interface type
  --filters FILTERS [FILTERS ...]
                        Only import these message types
  --progress            Show progress bar
```

### Importing KV6
```
python ./import.py --mode kv6 --filters ARRIVAL DEPARTURE --file KV6posinfo_2021-01-01.csv.xz
```

### Importing KV7
```
python ./import.py --mode kv7 --filters USERTIMINGPOINT TIMINGPOINT LINE --file KV7planning_2022-01-01.csv.xz
```
