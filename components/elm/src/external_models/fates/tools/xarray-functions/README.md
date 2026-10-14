# FATES function library for `xarray`

This repository provides a number of python functions for use with [xarray](https://docs.xarray.dev/en/stable/).

## FATES History Output

FATES history output is stored using the [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) format.  The output supports four dimensions, the first three of which are always: `time`, `lat`, and `lon`.  The fourth dimension will be a FATES-defined dimensions.  Some of the output variables require association with multiple FATES-specific dimensions.  To enable this, FATES will stack these dimensions, or "multiplex" them, together.  To facilitate the user, we've developed a number of python functions to unpack, or "deduplex", the dimensionality when using `xarray`.
