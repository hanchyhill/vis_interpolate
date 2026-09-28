"""Serialize netCDF4 I/O shared by business worker threads."""

from threading import RLock


NETCDF_IO_LOCK = RLock()
