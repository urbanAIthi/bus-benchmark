# State machine

Derives travel and dwell times from the logs as per the [BISON KV6 documentation](https://bison.dova.nu/sites/default/files/bestanden/tmi8_actuele_ritpunctualiteit_en_voertuiginformatie_kv_6_v8.1.3.1_release.pdf).

Only `ARRIVAL`, `ONSTOP` and `DEPARTURE` messages are used, with `ONSTOP` counting as an arrival. Each journey's messages are grouped into visits, one per stop passage (`userstopcode`, `passagesequencenumber`). The spec allows a vehicle to arrive at and depart from the same passage more than once, so all messages of a passage belong to one visit until the next passage starts. Messages for a passage the vehicle has already left are ignored.

| Visit contains                   | Arrival         | Departure                          | Dwell time             |
|----------------------------------|-----------------|------------------------------------|------------------------|
| An arrival and a later departure | First arrival   | Last departure after that arrival  | Departure − arrival    |
| An arrival, no later departure   | First arrival   | Missing                            | None                   |
| Departures only (vehicle passed) | First departure | First departure                    | Zero, if the previous visit has a departure |

A travel time is emitted from each visit's departure to the next visit's arrival. If a visit's departure is missing, no travel time is emitted from that stop. Departures before the first arrival of a visit are ignored: some operators send one as the vehicle pulls up to the stop, before it has halted.
