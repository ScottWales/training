# Xarray + Dask for climate analysis

Requirements:
 - xarray
 - dask

We recommend you use [NCI's VDI virtual desktops](https://opus.nci.org.au/display/Help/VDI+User+Guide), as these have direct access to the data

You can then access our Conda environment with

    module use /g/data3/hh5/public/modules
    module load conda/analysis3

## Part 1: Calculating a climate index

Use Xarray to calculate the ENSO 3.4 index for CMIP5 datasets

Topics:
 - Multi-file datasets
 - Constraining areas
 - Quick plots
 - Saving to NetCDF

 - [Slides]()
 - [Exercises]()

## Part 2: Finding extreme events

Use Xarray to find heatwaves in CMIP5 datasets

Topics:
 - Dask and chunking
 - Split-apply-combine operations
 - Creating climatologies
 - Rolling windows and custom filters

 - [Slides]()
 - [Exercises]()
