#!/g/data3/hh5/public/apps/miniconda3/envs/analysis3-19.04/bin/python
#PBS -q express
#PBS -l ncpus=16
#PBS -l mem=64gb
#PBS -l walltime=0:30:00
#PBS -l jobfs=400gb
#PBS -l wd
#PBS -j oe
#PBS -m abe

from dask.distributed import Client, progress
import tempfile
import xarray
import dask
import bottleneck
import numpy
import time
import os


def dask_percentile(array, axis, q):
    """
    Wrapper around numpy.percentile to operate on Dask arrays
    """
    array = array.rechunk({axis: -1})
    return array.map_blocks(numpy.percentile, axis=axis, q=q, dtype=array.dtype, drop_axis=axis)


def rolling_maximum(dataset):
    """
    Preprocess hourly max temperatures to 15 day rolling mean of daily max temperature
    """

    # Group by day (24 samples) then get the max of each day
    daily_time = dataset.time.data.reshape((-1, 24))[:,0]
    daily_max_data = dataset.data.reshape((-1, 24, dataset.shape[1], dataset.shape[2])).max(axis=1)
    
    # Currently there is a bug in xarray.DataArray.rolling, this is a manual implementation
    # Add a halo to each Dask chunk, then calculate a moving mean over 15 samples in the time axis
    rolling_data = dask.array.overlap.map_overlap(
        daily_max_data,
        func=bottleneck.move_mean,
        window=15,
        axis=0,
        depth=(14,0,0),
        boundary='reflect',
        trim=True,
        dtype=daily_max_data.dtype)

    # The moving mean is trailing - so it has the current time plus the 14
    # previous times. Correct the date to be in the middle of the window
    rolling_time = daily_time - numpy.timedelta64(7, 'D')
    
    # Convert the Dask array back into a DataArray
    rolling = xarray.DataArray(rolling_data,
                             dims = dataset.dims,
                             coords = {
                                 'time': ('time', rolling_time),
                                 'latitude': dataset.latitude,
                                 'longitude': dataset.longitude,
                             })
    
    return rolling


if __name__ == '__main__':
    # Get the number of CPUS in the job and start a dask.distributed cluster
    cores = int(os.environ.get('PBS_NCPUS','4'))
    client = Client(n_workers=cores, threads_per_worker=1, memory_limit='4gb', local_dir=tempfile.mkdtemp())

    start = time.perf_counter()

    # Read in ERA-5 data
    ds = xarray.open_mfdataset('/g/data/ub4/era5/netcdf/surface/MX2T/*/MX2T_era5_global_*.nc',
            chunks={'latitude': 91*1, 'longitude': 180*2})

    # Trim the input to a bit larger than the target period for the rolling
    # average, making sure we have full days
    ds = ds.sel(time=slice('19791201','20100131T2300'))

    print("Analysing %.2f GB"%(ds.mx2t.nbytes/(1024**3)))

    # Pre-process the input timeseries, then trim to the target date range
    rolled = rolling_maximum(ds.mx2t).sel(time=slice('19800101','20100101'))

    # Run a percentile on each day of the year
    doy_p90 = (rolled.groupby('time.dayofyear')
                     .reduce(dask_percentile, dim='time', q=90, allow_lazy=True))

    # Convert to a Dataset and save the output
    doy_p90 = doy_p90.to_dataset(name='mx2t_doy_p90')
    future = client.persist(doy_p90.to_netcdf('mx2t_doy_p90.nc', compute=False))

    # Uncomment for a progress bar:
    # progress(future)
    future.compute()

    end = time.perf_counter()
    print()
    print("time", end-start)

    client.close()
