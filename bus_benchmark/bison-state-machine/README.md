# State machine

As per the [BISON KV6 documentation](https://bison.dova.nu/sites/default/files/bestanden/tmi8_actuele_ritpunctualiteit_en_voertuiginformatie_kv_6_v8.1.3.1_release.pdf), we implement the following state machine to process the logs:

| Previous state | Next state | Condition           | Action                                              |
|----------------|------------|---------------------|-----------------------------------------------------|
| ARRIVAL        | ARRIVAL    |                     | Mark sequence as invalid (arrived twice)            |
| ARRIVAL        | DEPARTURE  | `last_stop == stop` | Emit dwell time                                     |
| ARRIVAL        | DEPARTURE  | `last_stop != stop` | Mark sequence as invalid (departed from wrong stop) |
| DEPARTURE      | ARRIVAL    | `last_stop == stop` | Emit warning (arrived at same stop)                 |
| DEPARTURE      | ARRIVAL    | `last_stop != stop` | Emit travel time                                    |
| DEPARTURE      | DEPARTURE  | `last_stop == stop` | Emit warning (departed from same stop twice)        |
| DEPARTURE      | DEPARTURE  | `last_stop != stop` | Emit travel time, emit empty dwell time             |
