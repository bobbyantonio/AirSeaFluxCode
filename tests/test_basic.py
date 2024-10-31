import time

import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import xarray_regrid
import numpy as np
import pandas as pd
import datetime
from glob import glob
from tabulate import tabulate
from src import AirSeaFluxCode

def get_dummy_global_dataset(resolution):
    
    return xr.Dataset(
        {
            'latitude': (['latitude'], np.arange(-90, 90 + resolution, resolution)),
            'longitude': (['longitude'], np.arange(0, 360, resolution)),
        }
    )
    
if __name__=="__main__":

    ## Get data ready
    fps = glob('/Users/bobbyantonio/repos/AirSeaFluxCode/sample_era5_data/*20070203.nc')

    var_name_lookup = {'z': 'geopotential', 'q': 'specific_humidity'}

    surface_fps = [fp for fp in fps if 'hPa' not in fp]
    surface_ds = xr.merge([xr.load_dataset(fp) for fp in surface_fps])

    plevel_fps = [fp for fp in fps if 'hPa' in fp]

    plevel_ds = []
    for var in ['z', 'q']:
        tmp_fps = [fp for fp in plevel_fps if var_name_lookup[var] in fp]
        tmp_ds = xr.concat([xr.load_dataset(fp).drop_vars('expver') for fp in sorted(tmp_fps)], dim='pressure_level')

        plevel_ds.append(tmp_ds)
    plevel_ds = xr.merge(plevel_ds)

    ds = xr.merge([surface_ds, plevel_ds]).sel(time=pd.date_range(start=datetime.datetime(2007,2,3), end=datetime.datetime(2007,2,3,6), freq='1h')).isel(time=0)
    plevels = ds['pressure_level'].values
    ds['q'] = ds['q'] * 1000 # Convert to g/kg

    # Expand pressure levels out
    for var in ['z', 'q']:
        for p in plevels:
            ds[f'{var}_{int(p)}'] = ds[var].sel(pressure_level=p)
            
    ds = ds.drop_vars('expver').drop_vars('number').drop_vars('q').drop_vars('z').drop_vars('pressure_level')

    ds['wind_speed'] = np.sqrt(ds['u10']**2 + ds['v10']**2)

    ds['msl'] = ds['msl'] / 100 # Convert to hPA
    ds['ewss'] = ds['ewss'] / (60*60) # Convert from integrated stress to mean stress
    ds['ewss'] = ds['nsss'] / (60*60)
    ds['surface_stress_magnitude'] = np.sqrt(ds['ewss']**2 + ds['nsss']**2)
    # ds['q'] = ds['q']*1000 

    # Interpolate humidity to surface
    ds['q_surface'] = ds['q_1000'] + (ds['q_1000'] - ds['q_975']) * (ds['msl'] - 1000) / 25
    df = ds.to_dataframe().reset_index()
    

    start = time.time()
    ### Perform iterations for low resoltion data first
    # # Createa a 2 degrees version. Just using bilinear for now to see speed
    target_dataset =get_dummy_global_dataset(2.0)

    ds_low_res = ds.unify_chunks().regrid.linear(target_dataset)

    df_low_res = ds_low_res.to_dataframe().reset_index()

    df_low_res = df_low_res[~np.isnan(df_low_res['sst'])].reset_index()
    df_low_res.head()
    
    start = time.time()
    warm_start_df = AirSeaFluxCode(spd=df_low_res['wind_speed'].to_numpy(), 
                    T=df_low_res['t2m'].to_numpy(), 
                    SST=df_low_res['skt'].to_numpy(), 
                    SST_fl="skin", 
                    meth="ecmwf", 
                    lat=df_low_res['latitude'].to_numpy(),
                    hin=np.array([10, 2]), 
                    hum=('q', df_low_res['q_surface'].to_numpy()), 
                    hout=10,
                    maxiter=50,
                    P=df_low_res['msl'].to_numpy(), 
                    cskin=0, 
                    Rs=df_low_res['msdwswrf'],
                    tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1], 
                    L="tsrv", 
                    wl=1,
                    out_var = ("tau", "sensible", "latent", "monob", "cd", "cd10n",
                        "ct", "ct10n", "cq", "cq10n", "tsrv", "tsr", "qsr",
                        "usr", "psim", "psit", "psiq", "psim_ref", "psit_ref",
                        "psiq_ref", "u10n", "t10n", "q10n", "zo", "zot", "zoq",
                        "uref", "tref", "qref", "qair", "qsea", "Rb", "rh",
                        "rho", "cp", "lv", "theta"))
    warm_start_df = warm_start_df.drop('flag', axis=1)
    # TODO: this filling of NaN values needs to be better
    warm_start_df = warm_start_df.fillna(method='bfill')
    warm_start_df = pd.concat([df_low_res[['latitude', 'longitude']], warm_start_df], axis=1)
    warm_start_ds = warm_start_df.set_index(['latitude', 'longitude']).to_xarray()
    print(f'Iterations took {time.time() - start} seconds')
    
    start = time.time()
    hi_res_dummy_dataset = get_dummy_global_dataset(0.25)
    warm_start_ds_hi_res = warm_start_ds.regrid.linear(hi_res_dummy_dataset)
    warm_start_df = warm_start_ds_hi_res.to_dataframe().reset_index()
    
    warm_start_df = warm_start_df[~np.isnan(df['sst'])].reset_index()
    warm_start_df = warm_start_df.fillna(method='bfill')
    df = df[~np.isnan(df['sst'])].reset_index()
    
    print(f'Regridding took {time.time() - start} seconds')

    start = time.time()
    res_ws = AirSeaFluxCode(spd=df['wind_speed'].to_numpy(), 
                    T=df['t2m'].to_numpy(), 
                    SST=df['skt'].to_numpy(), 
                    SST_fl="skin", 
                    meth="ecmwf", 
                    lat=df['latitude'].to_numpy(),
                    hin=np.array([10, 2]), 
                    hum=('q', df['q_surface'].to_numpy()), 
                    hout=10,
                    maxiter=50,
                    P=df['msl'].to_numpy(), 
                    cskin=0, 
                    Rs=df['msdwswrf'],
                    tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1], 
                    L="tsrv", 
                    wl=1,
                    out_var = ("tau", "sensible", "latent"),
                    warm_start_parameters=None)
    print(f'Iterations without warm start took {time.time() - start} seconds')
    
    start = time.time()
    res_no_ws = AirSeaFluxCode(spd=df['wind_speed'].to_numpy(), 
                    T=df['t2m'].to_numpy(), 
                    SST=df['skt'].to_numpy(), 
                    SST_fl="skin", 
                    meth="ecmwf", 
                    lat=df['latitude'].to_numpy(),
                    hin=np.array([10, 2]), 
                    hum=('q', df['q_surface'].to_numpy()), 
                    hout=10,
                    maxiter=50,
                    P=df['msl'].to_numpy(), 
                    cskin=0, 
                    Rs=df['msdwswrf'],
                    tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1], 
                    L="tsrv", 
                    wl=1,
                    out_var = ("tau", "sensible", "latent"),
                    warm_start_parameters=warm_start_df)
    print(f'Iterations with warm start took {time.time() - start} seconds')
    
    t=1
   