# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: airseaflux
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Introduction
#
# AirSeaFluxCode is developed to provide an easy and accessible way to calculate turbulent surface fluxes (TSFs) from a small number of bulk variables and for a viariety of bulk algorithms. 
#
# By running AirSeaFluxCode you can compare different bulk algorithms and to also investigate the effect choices within the implementation of each parameterisation have on the TSFs estimates. 
#

# %% [markdown]
# ### Getting started
#
# Let's first import the basic python packages we will need for reading in our input data, to perform basic statistics  and plotting

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# first import all packages you might need
# %matplotlib inline
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import xarray_regrid
import numpy as np
import pandas as pd
import datetime
from codetiming import Timer
from glob import glob
from tabulate import tabulate
from scipy import ndimage
from src import AirSeaFluxCode

def np_multiply_many(list_of_arrays: list):
    """
        Performs np.multiply iteratively for a list of arrays (assumed to be the same shape)

    Args:
        list_of_arrays (list): list of arrays of the same shape

    Returns:
        np.ndarray : element-wise product of arrays in the list
    """
    
    for n, arr in enumerate(list_of_arrays):
        if n == 0:
            result = arr.copy()
        else:
            result = np.multiply(arr, result)
    
    return result

def get_dummy_global_dataset(resolution):
    
    return xr.Dataset(
        {
            'latitude': (['latitude'], np.arange(-90, 90 + resolution, resolution)),
            'longitude': (['longitude'], np.arange(0, 360, resolution)),
        }
    )
    
def potential_temperature(temperature: np.array, pressure: np.array, 
                          pressure_ref: float= 1014.0, 
                          kappa: float=0.286):

    return np.multiply(temperature, np.power(pressure_ref/pressure, kappa))

def potential_temperature_to_temperature(potential_temperature: np.array, pressure: np.array, 
                          pressure_ref: float= 1014.0, 
                          kappa: float=0.286):

    return np.multiply(potential_temperature, np.power(pressure/pressure_ref, kappa))

def calculate_density(pressure, temperature):
    # Ideal gas formula
    return pressure / (287*temperature)

def dataset_to_flux_dataframe(ds: xr.Dataset,
                              plevels: list=None):
    """Converts a dataset into a pandas dataset, in order to pass it to the air-sea flux calculation

    Args:
        ds (xr.Dataset): ERA5-like dataset to convert into a Pandas dataframe

    Returns:
        pd.DataFrame: Pandas dataframe
    """
    
    # It seems that AirSeaFluxCode expects downwards (not net) radiation fluxes
    flux_input_variables = ['2m_temperature', 'mean_sea_level_pressure', 
                            'specific_humidity','10m_u_component_of_wind', 
                            '10m_v_component_of_wind',
                            'mean_surface_downward_short_wave_radiation_flux',
                            'mean_surface_downward_long_wave_radiation_flux']
    
    if plevels is None: 
        plevels = [975, 1000]

    ds = ds[flux_input_variables].sel(level=plevels).copy()
    ds['mean_sea_level_pressure'] = ds['mean_sea_level_pressure'] / 100 # Convert to hPA
    ds['specific_humidity'] = ds['specific_humidity'] * 1000 # Convert to g/kg
    ds['wind_speed'] = np.sqrt(ds['10m_u_component_of_wind']**2 + ds['10m_v_component_of_wind']**2)
       
    # Interpolate humidity to surface
    ds['specific_humidity_surface'] = ds['specific_humidity'].sel(level=1000) + (ds['specific_humidity'].sel(level=1000) - ds['specific_humidity'].sel(level=975)) * (ds['mean_sea_level_pressure'] - 1000) / 25
    
    # Drop any vars with level, and drop level, to allow creation of dataframe without level as an index
    ds = ds.drop_vars('specific_humidity').drop_vars('level')


    df = ds.to_dataframe().reset_index()

    return df

def heuristic_boundary_layer_height(ds: xr.Dataset, 
                                    threshold: float=1.0,
                                    pressure_levels: list=None):
    
    """Heuristic method to calculate boundary layer height, based on the location that 
    potential temperature changes significantly
    
    Args:
        ds (xr.Dataset): Dataset containing temperature, geopotential, 2m_temperature, on pressure levels containing
        1000hPa, 975hPa, 950hPa, +...

    Returns:
        height_da, pressure_da: DataArrays containing boundary layer height on the grid, and the pressure level of the boundary height (to be used to update)
    """
    
    ds = ds.sortby('level', ascending=False)
    
    if pressure_levels is None:
        pressure_levels = [1000, 975, 950, 925, 900, 875, 850, 825, 800]
    else:
        pressure_levels = sorted(pressure_levels, reverse=True)
    
    da_pt = ds[f'temperature'].copy() 
    for p in pressure_levels:
        da_pt.loc[{'level': p}] = potential_temperature(temperature=ds['temperature'].sel(level=p), pressure=p)

    # Calculate boundary layer heights
    
    height_da = xr.ones_like(ds['2m_temperature']) * np.nan
    pressure_da = xr.ones_like(ds['2m_temperature']) * np.nan

    for n, p in enumerate([1014] + list(pressure_levels)):
        if n == 0:
            pressure_level_da = xr.ones_like(ds['2m_temperature']) * p
        else:
            pressure_level_da = xr.ones_like(ds['2m_temperature']) * p
            
        if n == 0:
            
            large_diff_2m_1000 = np.abs(ds['2m_temperature'] - da_pt.sel(level=1000)) > threshold
            large_diff_2m_975 = np.abs(ds['2m_temperature'] - da_pt.sel(level=975)) > threshold
            large_diff_2m_950 = np.abs(ds['2m_temperature'] - da_pt.sel(level=975)) > threshold
            
            large_diff_1000_975 = np.abs(da_pt.sel(level=1000) - da_pt.sel(level=975)) > threshold

            large_diff_975_950 = np.abs(da_pt.sel(level=975) - da_pt.sel(level=950)) > threshold
            large_diff_950_925 = np.abs(da_pt.sel(level=950) - da_pt.sel(level=925)) > threshold
            
            check_1000hpa_positive = ds[f'geopotential'].sel(level=1000) > 0
            check_975hpa_positive = ds[f'geopotential'].sel(level=975) > 0
            
            logical_condition_1 = np.logical_and(check_1000hpa_positive, np.logical_and(large_diff_2m_1000, large_diff_1000_975))
            logical_condition_2 = np.logical_and(check_975hpa_positive, np.logical_and(large_diff_2m_975, large_diff_975_950))
            logical_condition_3 = np.logical_and(large_diff_2m_950, large_diff_950_925)
            
            # Use height halfway between 10m and the level of the first positive geopotential
            height_to_use = xr.where(check_1000hpa_positive, 0.5*(10.0 + ds['geopotential'].sel(level=1000)), xr.where(check_975hpa_positive, 0.5*(10.0 + ds['geopotential'].sel(level=975)), 0.5*(10.0 + ds['geopotential'].sel(level=950))))
            # height_to_use = 100
            
            height_da = xr.where(np.logical_or(np.logical_or(logical_condition_1, logical_condition_2), logical_condition_3), height_to_use, np.nan)
            pressure_da = xr.where(np.logical_or(np.logical_or(logical_condition_1, logical_condition_2), logical_condition_3), pressure_level_da, np.nan)
            
        elif n == len(pressure_levels):
            values_are_null = height_da.isnull()
            height_da = xr.where(values_are_null, ds['geopotential'].sel(level=p) / 9.81, height_da)
            pressure_da = xr.where(values_are_null, pressure_level_da, pressure_da)
        else:
            
            large_diff_here = np.abs(da_pt.sel(level=p) - da_pt.sel(level=pressure_levels[n])) > threshold
            check_positive = ds['geopotential'].sel(level=p) > 0

            values_are_null = height_da.isnull()
            
            logical_condition =  np.logical_and(check_positive, np.logical_and(large_diff_here, values_are_null))
            
            height_da = xr.where(logical_condition, 0.5*(ds['geopotential'].sel(level=p) + ds['geopotential'].sel(level=pressure_levels[n]))/ 9.81, height_da)
            # height_da = xr.where(logical_condition, ds['geopotential'].sel(level=p)/ 9.81, height_da)
            pressure_da = xr.where(logical_condition, pressure_level_da, pressure_da)

    return height_da, pressure_da


    

# %% [markdown]
# ## Air sea flux code on gridded data

# %%
# TODO: precip needs to be aggregated

# %%
# 1 second per 6 hours in SYPD
secs_per_6hr_step = 2
secs_per_day = secs_per_6hr_step * 4
secs_per_year = secs_per_day*365
print(secs_per_year / 60)
sypd = (24*60*60) / secs_per_year
print(sypd)

# %%
year = 2016
month = 1
day = 1
hr = 12

dt = datetime.datetime(year,month,day, hour=hr)

# %%
sst_da = xr.load_dataarray(f"/Users/bobbyantonio/repos/AirSeaFluxCode/sample_era5_data/era5_sea_surface_temperature_{dt.strftime('%Y%m%d')}.nc").isel(time=0)

# Regrid to 2 degrees
target_2_degree_ds = get_dummy_global_dataset(2.0)
# sst_da = sst_da.regrid.linear(target_2_degree_ds).sel(latitude=np.arange(-90, 90,2)) # Subselecting latitude to be consistent with saved data

latitude_vals = sst_da['latitude'].values
longitude_vals = sst_da['longitude'].values

sst_array = sst_da.values
sst_shape = sst_array.shape

latitude_array = np.broadcast_to(np.array(latitude_vals).reshape(-1,1), sst_shape)
longitude_array = np.broadcast_to(np.array(longitude_vals).reshape(1,-1), sst_shape)

sea_mask = xr.where(~sst_da.isnull(), True, False)
sst_mask = ~np.isnan(sst_array)
sea_mask_df = sea_mask.to_dataframe().reset_index()[['sst']]

lat_vals = latitude_array[sst_mask]
lon_vals = longitude_array[sst_mask]

sea_points = np.array(list(zip(lat_vals, lon_vals)))

# %%

glob_str = f"/Users/bobbyantonio/repos/AirSeaFluxCode/sample_era5_data/*{dt.strftime('%Y%m%d')}.nc"
fps = glob(glob_str)

var_name_lookup = {'z': 'geopotential', 'q': 'specific_humidity', 't': 'temperature'}

surface_fps = [fp for fp in fps if 'hPa' not in fp]
surface_ds = xr.merge([xr.load_dataset(fp) for fp in surface_fps])


# %%

# Means of 
surface_ds['surface_wind_magnitude'] = np.sqrt(surface_ds['u10']**2 + surface_ds['v10']**2)
surface_ds['surface_stress_magnitude'] = np.sqrt(surface_ds['iews']**2 + surface_ds['inss']**2)

for var in ['msshf', 'mslhf', 'surface_stress_magnitude']:
    surface_ds[f'{var}_6hr'] = surface_ds[var].sel(time=[dt + pd.Timedelta(n, 'h') for n in range(1, 7)]).groupby(['latitude', 'longitude']).mean(...).expand_dims(dim={'time': 1}).assign_coords({'time': [dt]}).compute()

surface_ds['msnswrf'] = surface_ds['msnswrf'].fillna(0)
surface_ds_t0 = surface_ds.sel(time=dt)

plevel_fps = [fp for fp in fps if 'hPa' in fp]

plevel_ds = []
for var in ['z', 'q', 't']:
    tmp_fps = [fp for fp in plevel_fps if var_name_lookup[var] in fp]
    if len(tmp_fps) > 0:
        tmp_ds = xr.concat([xr.load_dataset(fp).drop_vars('expver') for fp in sorted(tmp_fps)], dim='pressure_level')

        plevel_ds.append(tmp_ds)
plevel_ds = xr.merge(plevel_ds)

ds = xr.merge([surface_ds_t0, plevel_ds]).sel(time=dt)
                                           
                                           
# ds['ewss'] = ds['ewss'] / (60*60) # Convert from integrated stress to mean stress
# ds['ewss'] = ds['nsss'] / (60*60)

ds = ds.rename({'u10': '10m_u_component_of_wind',
                'v10': '10m_v_component_of_wind',
                'z': 'geopotential',
                'q': 'specific_humidity',
                'msl': 'mean_sea_level_pressure',
                't2m': '2m_temperature',
                'skt': 'skin_temperature',
                't': 'temperature',
                'sst': 'sea_surface_temperature',
                'msnswrf': 'mean_surface_net_short_wave_radiation_flux',
                'msdwswrf': 'mean_surface_downward_short_wave_radiation_flux',
                'msdwlwrf': 'mean_surface_downward_long_wave_radiation_flux',
                'pressure_level': 'level'}).drop_vars('expver').drop_vars('number')

sea_surface_ds = ds[['skin_temperature', 'sea_surface_temperature']]


# %%
# # Createa a 2 degrees version. Just using bilinear for now to see speed
target_dataset = xr.Dataset(
    {
        'latitude': (['latitude'], np.arange(-90, 90 + 2, 2)),
        'longitude': (['longitude'], np.arange(0, 360, 2)),
    }
)

ds_2deg = ds.unify_chunks().regrid.linear(target_dataset)

df_2deg = ds_2deg.to_dataframe().reset_index()

df_2deg = df_2deg[~np.isnan(df_2deg['sea_surface_temperature'])].reset_index()
df_2deg.head()

# %%
df

# %%
df = dataset_to_flux_dataframe(ds)
sea_surface_df = sea_surface_ds['sea_surface_temperature'].to_dataframe().reset_index()
sea_surface_df = sea_surface_df[sea_mask_df['sst']]
sea_surface_df = sea_surface_df

flux_df = df[sea_mask_df['sst']].reset_index()

res_sst = AirSeaFluxCode.AirSeaFluxCode(spd=flux_df['wind_speed'].copy().to_numpy(), 
                     T=flux_df['2m_temperature'].copy().to_numpy(), 
                     SST=sea_surface_df['sea_surface_temperature'].to_numpy(), 
                     SST_fl="bulk", 
                     meth="ecmwf", 
                     lat=flux_df['latitude'].to_numpy(),
                     hin=np.array([10, 2]), 
                     hum=('q', flux_df['specific_humidity_surface'].copy().to_numpy()), 
                     hout=10,
                     maxiter=10,
                     P=flux_df['mean_sea_level_pressure'].copy().to_numpy(), 
                     cskin=1, 
                     Rs=flux_df['mean_surface_downward_short_wave_radiation_flux'].copy().to_numpy(),
                     Rl=flux_df['mean_surface_downward_long_wave_radiation_flux'].copy().to_numpy(),
                     tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1], 
                     L="tsrv", 
                     wl=1,
                     gust=[4, 1.2, 600, 0.01],
                     out_var = ("tau", "sensible", "latent", "cd", "cp", "ct", "rho", 'dter', 'dqer', 'dtwl'),
                     out=1)

res_sst_df = pd.concat([flux_df[['latitude', 'longitude']], res_sst], axis=1)
full_sst_df = df[['latitude', 'longitude']].merge(res_sst_df, on=['latitude', 'longitude'], how='left')
res_sst_ds = full_sst_df.set_index(['latitude', 'longitude']).to_xarray()

# TODO: try using ffill instead (bit slower)
for var in ["tau", "sensible", "latent", "cd", "cp", "rho"]:

    if var in  ["tau", "sensible", "latent"]:
        # Fill NaNs, set land points to 0 flux, and set sea-ice points to 0 flux
        res_sst_ds[var] = xr.where(sea_mask, res_sst_ds[var].fillna(res_sst_ds[var].mean().item()), 0)
    else:
        res_sst_ds[var] = res_sst_ds[var].fillna(res_sst_ds[var].mean().item())
             
print('Num NaN tau in output dataframe: ', res_sst_df['tau'].isna().sum(), ' / ', len(res_sst_df['tau']))  
print('Num NaN tau in final dataset: ', res_sst_ds['tau'].isnull().sum().item(), ' / ', res_sst_ds['tau'].size)

# %%

# df = dataset_to_flux_dataframe(ds)
# sea_surface_df = sea_surface_ds['skin_temperature'].to_dataframe().reset_index()
# sea_surface_df = sea_surface_df[sea_mask_df['sst']]

# flux_df = df[sea_mask_df['sst']].reset_index()


# res_ssst = AirSeaFluxCode.AirSeaFluxCode(spd=flux_df['wind_speed'].to_numpy(), 
#                      T=flux_df['2m_temperature'].to_numpy(), 
#                      SST=sea_surface_df['skin_temperature'].to_numpy(), 
#                      SST_fl="skin", 
#                      meth="ecmwf", 
#                      lat=flux_df['latitude'].to_numpy(),
#                      hin=np.array([10, 2]), 
#                      hum=('q', flux_df['specific_humidity_surface'].to_numpy()), 
#                      hout=10,
#                      maxiter=20,
#                      P=flux_df['mean_sea_level_pressure'].to_numpy(), 
#                      cskin=0, 
#                     #  Rs=flux_df['msdwswrf'].to_numpy(),
#                      tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1], 
#                      L="tsrv", 
#                      wl=0,
#                      out_var = ("tau", "sensible", "latent", "cd", "cp", "rho", "usr", "ct", "ct10n"))
# res_ssst_df = pd.concat([flux_df[['latitude', 'longitude']], res_ssst], axis=1)
# full_ssst_df = df[['latitude', 'longitude']].merge(res_ssst_df, on=['latitude', 'longitude'], how='left')
# res_ssst_ds = full_ssst_df.set_index(['latitude', 'longitude']).to_xarray()

# # TODO: try using ffill instead (bit slower)
# for var in ["tau", "sensible", "latent", "cd", "cp", "rho"]:

#     if var in  ["tau", "sensible", "latent"]:
#         # Fill NaNs, set land points to 0 flux, and set sea-ice points to 0 flux
#         res_ssst_ds[var] = xr.where(sea_mask, res_ssst_ds[var].fillna(res_ssst_ds[var].mean().item()), 0)
#     else:
#         res_ssst_ds[var] = res_ssst_ds[var].fillna(res_ssst_ds[var].mean().item())
             
# print('Num NaN tau in output dataframe: ', res_ssst_df['tau'].isna().sum(), ' / ', len(res_ssst_df['tau']))  
# print('Num NaN tau in final dataset: ', res_ssst_ds['tau'].isnull().sum().item(), ' / ', res_ssst_ds['tau'].size)

# %%
# Sanity check results compared to ERA5
funcs = {'mean': lambda x: x.mean().item(), 'max': lambda x: x.max().item(), 'min': lambda x: x.min().item()}

for k, func in funcs.items():
    
    print(f"{k} for ERA5: {func(ds['msshf'])}")
    print(f"{k} for ERA5 6hr: {func(ds['msshf_6hr'])}")
    print(f"{k} for AirSeaFlux: {func(res_sst_ds['sensible'])}")

# %%
res_2deg = AirSeaFluxCode(spd=df_2deg['wind_speed'].to_numpy(), 
                     T=df_2deg['t2m'].to_numpy(), 
                     SST=df_2deg['skt'].to_numpy(), 
                     SST_fl="skin", 
                     meth="ecmwf", 
                     lat=df_2deg['latitude'].to_numpy(),
                     hin=np.array([10, 2]), 
                     hum=('q', df_2deg['q_surface'].to_numpy()), 
                     hout=10,
                     maxiter=50,
                     P=df_2deg['msl'].to_numpy(), 
                     cskin=0, 
                     Rs=df_2deg['msdwswrf'],
                     tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1], 
                     L="tsrv", 
                     wl=1,
                     out_var = ("tau", "sensible", "latent", "cd", "rho", "uref"))
res_2deg = pd.concat([df_2deg[['latitude', 'longitude']], res_2deg], axis=1)
res_2deg_ds = res_2deg.set_index(['latitude', 'longitude']).to_xarray()

print('Num nan tau on sea: ', res_2deg_ds['tau'].isna().sum(), ' / ', len(res_2deg))

# %%
num_rows = 4
num_cols = 2

fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*5))
res_sst_ds['sensible'].plot.imshow(cmap='RdBu_r', ax=ax[0,0], vmin=-400, vmax=400)
ax[0,0].set_title('Calculated Sensible')
sshf_da = ds['msshf'].where(sea_mask)
sshf_da.plot.imshow(ax=ax[0,1], vmin=-400, vmax=400, cmap='RdBu_r',)
ax[0,1].set_title('ERA5 Sensible')

res_sst_ds['tau'].plot.imshow(ax=ax[1,0], vmin=0, vmax=4)
tau_da = ds['surface_stress_magnitude'].where(sea_mask)
tau_da.plot.imshow(ax=ax[1,1], vmin=0, vmax=4)

res_sst_ds['latent'].plot.imshow(ax=ax[2,0], vmin=-600, vmax=600, cmap='RdBu_r',)
tau_da = ds['mslhf'].where(sea_mask)
tau_da.plot.imshow(ax=ax[2,1], vmin=-600, vmax=600, cmap='RdBu_r',)

ds['siconc'].plot.imshow(ax=ax[3,0])


# %% [markdown]
# ## Sensible heat flux

# %%
num_rows = 2
num_cols = 2
entrainment_coefficient = 0.2
delta_t = 6*60*60
height_da, pressure_da = heuristic_boundary_layer_height(ds)
height_da = np.clip(height_da, a_min=200, a_max=None)

fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*5))


xr.where(sea_mask, height_da, np.nan).plot(ax=axs[0,0], vmin=0, vmax=2000)
xr.where(sea_mask, ds['blh'], np.nan).plot(ax=axs[0,1], vmin=0, vmax=2000)

exponential_factor_with_bl = np.exp( -  (1 + entrainment_coefficient) * delta_t * np.divide(np.multiply(res_sst_ds['ct'], ds['surface_wind_magnitude']) , height_da)  )
exponential_factor = np.exp( -  (1 + entrainment_coefficient) * delta_t * np.divide(np.multiply(res_sst_ds['ct'], ds['surface_wind_magnitude']) , ds['blh'])  )


exponential_factor_with_bl.plot.imshow(ax=axs[1,0], vmin=0, vmax=2.5)
exponential_factor.plot(ax=axs[1,1], vmin=0, vmax=2.5)

axs[0,0].set_title('Calculated boundary height')
axs[0,1].set_title('ERA5 boundary height')
axs[1,0].set_title('Exp factor with calc h')
axs[1,1].set_title('Exp factor with ERA5 h')

# %%
num_rows = 2
num_cols = 2
entrainment_coefficient = 0.2
delta_t = 6*60*60 # 6 hours
height_da, pressure_da = heuristic_boundary_layer_height(ds)
height_da = np.clip(height_da, a_min=200, a_max=None)


fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*5))

# plot boundary layer height distribution
xr.where(sea_mask, height_da, np.nan).plot.hist(ax=axs[0,0], bins=10, density=True)
axs[0,0].set_xlim([0,3000])
xr.where(sea_mask, ds['blh'], np.nan).plot.hist(bins=20, ax=axs[0,1], density=True)
axs[0,1].set_xlim([0,3000])


xr.where(sea_mask, ds['blh'], np.nan).plot(ax=axs[1,0], vmin=0, vmax=3000)
xr.where(sea_mask, height_da, np.nan).plot(ax=axs[1,1], vmin=0, vmax=3000)

# %%
num_rows = 2
num_cols = 2

entrainment_coefficient = 0.2
delta_t = 6*60*60
mixed_layer_depth = 20 # To be replaced with NEMO MLD
sea_density = 1.03 # To be replaced, use potential density?
ocean_heat_capacity = 4200
cp = res_sst_ds['cp']
skin_temp = sea_surface_ds['sea_surface_temperature'] -  res_sst_ds['dter'] - res_sst_ds['dtwl'] #TODO: Get NEMO to pass skin temperature instead

# Calculate equilibrium temperature
ocean_eqm_factor = sea_density * ocean_heat_capacity * mixed_layer_depth
atm_eqm_factor = np_multiply_many([height_da ,res_sst_ds['rho'], res_sst_ds['cp']])

t_eqm_numerator = np.multiply(ocean_eqm_factor , skin_temp) + np.multiply(atm_eqm_factor , ds['2m_temperature'])
t_eqm_denominator = ocean_eqm_factor + atm_eqm_factor

t_eqm = np.divide(t_eqm_numerator, t_eqm_denominator)

# Calculate exponential coefficients
# Currently making lots of simplifications of the ocean which should be checked!
# But since it is to calculate an equilibrium temperature perhaps it is not too important
exp_coeff_atm = (1 + entrainment_coefficient) * np.divide(np.multiply(res_sst_ds['ct'], ds['surface_wind_magnitude']) , height_da)
exp_coeff_oce = np_multiply_many([res_sst_ds['cp'], ds['surface_wind_magnitude'], res_sst_ds['ct'], res_sst_ds['rho']]) / (sea_density * ocean_heat_capacity * mixed_layer_depth)


# %%
# Calculate new temperature, and temperature change
new_t = (ds['2m_temperature'] - t_eqm) * np.exp( - (exp_coeff_atm + exp_coeff_oce) * delta_t) + t_eqm
update = new_t - ds['2m_temperature']

# %%

# multiply by time...

fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*5))

temp_diff = ds['2m_temperature'] - skin_temp

update.plot(ax=axs[0,0], vmin=-10, vmax=10, cmap='RdBu_r')

xr.where(sea_mask, -temp_diff, np.nan).plot(ax=axs[0,1], vmin=-10, vmax=10, cmap='RdBu_r')
xr.where(sea_mask, surface_ds['t2m'].sel(time=dt+datetime.timedelta(hours=6)) - surface_ds['t2m'].sel(time=dt), np.nan).plot(ax=axs[1,0], vmin=-10, vmax=10, cmap='RdBu_r')

axs[0,0].set_title('Predicted change due to sensible heat flux')
axs[0,1].set_title('Skin temperature - t2m')
axs[1,0].set_title('Actual t2m change')

# %%
figure, axs = plt.subplots(3,1)

skin_temp.plot(ax=axs[0], vmax=310, vmin=255)
sea_surface_ds['sea_surface_temperature'].plot(ax=axs[1], vmax=310, vmin=255)
(skin_temp - sea_surface_ds['sea_surface_temperature']).plot(ax=axs[2], vmin=-2, vmax=2, cmap='RdBu_r')

# %%
figure, axs = plt.subplots(4,1)

t_eqm.plot(ax=axs[0], vmax=310, vmin=255)
(t_eqm - ds['2m_temperature']).plot(ax=axs[1],  cmap='RdBu_r', vmax=-2, vmin=2)
(t_eqm - skin_temp).plot(ax=axs[2], vmin=-2, vmax=2, cmap='RdBu_r')
xr.where( np.logical_or(np.logical_and(t_eqm > ds['2m_temperature'], t_eqm > skin_temp), np.logical_and(t_eqm < ds['2m_temperature'], t_eqm < skin_temp)), 1,0).plot(ax=axs[3])

# %% [markdown]
# ## Latent heat flux

# %%
# First need to calculate q_s the saturation specific humidity of air

# %%
from src import hum_subs

# Using Buck method for water as it seems to align with what the IFS does
# However the Alduchov Eskridge method that IFS uses for ice doesn't seem to be available. So use GoffGratch which is said to be standard according to AirSeaFluxCode
vapor_pressure_water = hum_subs.VaporPressure(skin_temp.values, ds['mean_sea_level_pressure'].values, 'liquid', meth='Buck')
vapor_pressure_ice = hum_subs.VaporPressure(skin_temp.values, ds['mean_sea_level_pressure'].values, 'ice', meth='GoffGratch')

# %%
saturation_vapor_press = np.multiply(1 - ds['siconc'], vapor_pressure_water) + np.multiply(ds['siconc'], vapor_pressure_ice) 

# %%
hum_subs.qsat_air(skin_temp, ds['mean_sea_level_pressure'], )

# %%
# From IFS documentation:
# q_sat = ( (R_dry / R_vap) * e_sat ) / (  pressure - (1  -  (R_dry / R_vap) ) * e_sat)

# where

# R_dry gas constant for dry air  = 
# R_vap gas constant for water vapour

# From AeroBulk:
# R_dry = 287.05       # Specific gas constant for dry air              [J/K/kg]
# R_vap = 461.495      # Specific gas constant for water vapor          [J/K/kg]

# %% [markdown]
# ## Momentum flux

# %%
num_rows = 2
num_cols = 2

fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*5))
xr.where(sea_mask, ds['surface_stress_magnitude_6hr'], np.nan).plot(ax=ax[0,0], vmin=0, vmax=8)
xr.where(sea_mask, ds['surface_stress_magnitude'], np.nan).plot(ax=ax[0,1], vmin=0, vmax=8)

delta_t = 6*60*60 # 6 hours
mmntm_numerator = np.multiply(ds['surface_wind_magnitude'] , res_ssst_ds['cd']) * delta_t
mmntm_denominator = height_da + np.multiply(ds['surface_wind_magnitude'] , res_ssst_ds['cd']) * delta_t

approx_6hr_momentum_flux = np.divide(mmntm_numerator, mmntm_denominator)

xr.where(sea_mask, approx_6hr_momentum_flux, np.nan).plot(ax=ax[1,0], vmin=0, vmax=8)

change_in_wind_speed_era5 = surface_ds['surface_wind_magnitude'].sel(time=datetime.datetime(2016, 1, 1, 18, 0)) - surface_ds['surface_wind_magnitude'].sel(time=datetime.datetime(2016, 1, 1, 12, 0))

xr.where(sea_mask, change_in_wind_speed_era5, np.nan).plot(ax=ax[1,1], vmin=0, vmax=8)


# %%

surface_ds['surface_wind_magnitude'].sel(time=datetime.datetime(2016, 1, 1, 18, 0)) - surface_ds['surface_wind_magnitude'].sel(time=datetime.datetime(2016, 1, 1, 12, 0))

# %%
# Scatter plot between instantaneous heat flux and 

# %%
num_rows = 4
num_cols = 2

fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*5))
res_2deg_ds['sensible'].plot.imshow(cmap='RdBu_r', ax=ax[0,0], vmin=-400, vmax=400)
sshf_da = ds_2deg['msshf'].where(~ds_2deg['sst'].isnull())
sshf_da.plot.imshow(ax=ax[0,1], vmin=-400, vmax=400, cmap='RdBu_r',)

res_2deg_ds['tau'].plot.imshow(ax=ax[1,0], vmin=0, vmax=4)
tau_da = ds_2deg['surface_stress_magnitude'].where(~ds_2deg['sst'].isnull())
tau_da.plot.imshow(ax=ax[1,1], vmin=0, vmax=4)

res_2deg_ds['latent'].plot.imshow(ax=ax[2,0], vmin=-600, vmax=600, cmap='RdBu_r',)
tau_da = ds_2deg['mslhf'].where(~ds_2deg['sst'].isnull())
tau_da.plot.imshow(ax=ax[2,1], vmin=-600, vmax=600, cmap='RdBu_r',)

ds_2deg['siconc'].plot.imshow(ax=ax[3,0])

# %%
ds['surface_stress_magnitude'].where(~ds['sst'].isnull()).plot.hist()

# %%
res['tau'].max()

# %%
ds['surface_stress_magnitude'].where(~ds['sst'].isnull()).max()

# %% [markdown]
# ### AirSeaFluxCode examples
#
# AirSeaFluxCode is set up to run in its default setting with a minimum number of input variables, namely wind speed; air temperature; and sea surface temperature. Let's load the code, import some real data composed for testing it (Research vessel data) and run AirSeaFluxCode with default settings (output height 10m, cool skin/warm layer corrections turned off, bulk algorithm Smith 1988, gustiness on, saturation vapour pressure following Buck (2012), tolerance limits set for both flux estimates and height adjusted variables (['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1]), number of iterations are ten, non converged points are set to missing and Monin-Obukhov stability length is calculated following the ECMWF implementation.

# %%
inDt = pd.read_csv("Test_Data/data_all.csv")

# %%
inDt = inDt[inDt['Date'] == 20070203]

# %%
inDt = pd.read_csv("Test_Data/data_all.csv")
date = np.asarray(inDt["Date"])
lon = np.asarray(inDt["Longitude"])
lat = np.asarray(inDt["Latitude"])
spd = np.asarray(inDt["Wind speed"])
t = np.asarray(inDt["Air temperature"])
sst = np.asarray(inDt["SST"])
rh = np.asarray(inDt["RH"])
p = np.asarray(inDt["P"])
sw = np.asarray(inDt["Rs"])
hu = np.asarray(inDt["zu"])
ht = np.asarray(inDt["zt"])
hin = np.array([hu, ht, ht])
del inDt
outvar = ("tau", "sensible", "latent", "u10n", "t10n", "q10n")
# run AirSeaFluxCode

increase_factor=100
res_baseline = AirSeaFluxCode(spd, 
                     t, 
                     sst, 
                     "bulk", 
                     meth="UA", 
                     lat=lat, 
                     hin=hin, 
                     hum=["rh", rh], 
                     hout=10.3,
                     P=p, 
                     cskin=0, 
                     Rs=sw,
                     tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1], L="tsrv", 
                     out_var = outvar)

# %%
t

# %%
np.abs(res['sensible']).max()

# %%
hu

# %%
ht

# %%
increase_factor = 100
res = AirSeaFluxCode(np.concat([spd]*increase_factor), 
                     np.concat([t]*increase_factor), 
                     np.concat([sst]*increase_factor), 
                     "bulk", 
                     meth="UA", 
                     lat=np.concat([lat]*increase_factor), 
                     hin=np.concat([hin]*increase_factor, axis=1), 
                     hum=["rh", np.concat([rh]*increase_factor)], 
                     P=np.concat([p]*increase_factor),
                     cskin=0, 
                     Rs=np.concat([sw]*increase_factor),
                     tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1], L="tsrv", 
                     out_var = outvar)

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# res is the output of AirSeaFluxCode which is a dataFrame with keys: "tau", "sensible", "latent", "u10n", "t10n", "q10n". Let's plot the flux estimates.

# %%
fig, ax = plt.subplots(3, 1, sharex=True, sharey=False)
fig.tight_layout()
ax[0].plot(res["tau"], "-", color="grey", linewidth=1, alpha = 0.8)
ax[1].plot(res["sensible"], "-", color="grey", linewidth=1, alpha = 0.8)
ax[2].plot(res["latent"], "-", color="grey", linewidth=1, alpha = 0.8)
ax[0].set_ylabel('tau (Nm$^{-2}$)')
ax[1].set_ylabel('shf (Wm$^{-2}$)')
ax[2].set_ylabel('lhf (Wm$^{-2}$)')
plt.show()

# %% [markdown]
# You can save the output in a csv file

# %%
res.to_csv("test_AirSeaFluxCode.csv")

# %% [markdown]
# and generate some statistics which you can save in a txt file

# %%
print("Input summary", file=open('./stats.txt', 'a'))
print('input file name: {}, \n method: {}, \n gustiness: {}, \n cskin: {},'
      ' \n tolerance: {}, \n qmethod: {}, \n L: {}'.format("data_all.csv", "UA", "on",
                                                           0, "all", "Buck2", "Rb"),
      file=open('./stats.txt', 'a'))
ttl = np.asarray(["tau  ", "shf  ", "lhf  ", "u10n ", "t10n ", "q10n "])
header = ["var", "mean", "median", "min", "max", "5%", "95%"]
n = np.shape(res)
stats = np.copy(ttl)
a = res.iloc[:,:-1].to_numpy(dtype="float64").T
stats = np.c_[stats, np.nanmean(a, axis=1)]
stats = np.c_[stats, np.nanmedian(a, axis=1)]
stats = np.c_[stats, np.nanmin(a, axis=1)]
stats = np.c_[stats, np.nanmax(a, axis=1)]
stats = np.c_[stats, np.nanpercentile(a, 5, axis=1)]
stats = np.c_[stats, np.nanpercentile(a, 95, axis=1)]
print(tabulate(stats, headers=header, tablefmt="github", numalign="left",
               floatfmt=("s", "2.2e", "2.2e", "2.2e", "2.2e", "2.2e", "2.2e")),
      file=open('./stats.txt', 'a'))
print('-'*79+'\n', file=open('./stats.txt', 'a'))
del a

# %%
