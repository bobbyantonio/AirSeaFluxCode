# Changes

Differences between version 1.0.0 and version 1.1.0.

## Bug Fixes

* Humidity units are now consistently `g/kg` throughout ([#2](https://github.com/NOCSurfaceProcesses/AirSeaFluxCode/pull/2)).
* Correct function call to `delta` in `cs_wl_subs.cs` ([#2](https://github.com/NOCSurfaceProcesses/AirSeaFluxCode/pull/2)).
* Input RH values < 1%, and humidity units `g/kg` < 1 now display warning rather than raising error ([#3](https://github.com/NOCSurfaceProcesses/AirSeaFluxCode/pull/3)).

## Documentation

* Formatting of variables, types, and units are now consistent ([#2](https://github.com/NOCSurfaceProcesses/AirSeaFluxCode/pull/2)).

## Optimisations

* Minor refactors to simplify `hum_subs.VapourPressure` ([#2](https://github.com/NOCSurfaceProcesses/AirSeaFluxCode/pull/2)).

## Contributors

* jtsiddons
* rcornes
* SCChan21
