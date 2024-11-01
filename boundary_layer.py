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

def get_dummy_global_dataset(resolution):
    
    return xr.Dataset(
        {
            'latitude': (['latitude'], np.arange(-90, 90 + resolution, resolution)),
            'longitude': (['longitude'], np.arange(0, 360, resolution)),
        }
    )
    
def potential_temperature(temp: np.array, pressure: np.array, 
                          pressure_ref: float= 1014.0, 
                          kappa: float=0.286):

    return np.multiply(temp, np.power(pressure_ref/pressure, kappa))

def potential_temperature_to_temperature(potential_temperature: np.array, pressure: np.array, 
                          pressure_ref: float= 1014.0, 
                          kappa: float=0.286):

    return np.multiply(potential_temperature, np.power(pressure/pressure_ref, kappa))

def calculate_density(pressure, temperature):
    # Ideal gas formula
    return pressure / (287*temperature)
    


# %% [markdown]
# ## Heuristic Boundary layer function

# %%
sst_da = xr.load_dataarray('/Users/bobbyantonio/repos/AirSeaFluxCode/sample_era5_data/era5_sea_surface_temperature_20070203.nc').isel(time=0)

# Regrid to 2 degrees
target_2_degree_ds = get_dummy_global_dataset(2.0)
sst_da = sst_da.regrid.linear(target_2_degree_ds).sel(latitude=np.arange(-90, 90,2)) # Subselecting latitude to be consistent with saved data

latitude_vals = sst_da['latitude'].values
longitude_vals = sst_da['longitude'].values

sst_array = sst_da.values
sst_shape = sst_array.shape

latitude_array = np.broadcast_to(np.array(latitude_vals).reshape(-1,1), sst_shape)
longitude_array = np.broadcast_to(np.array(longitude_vals).reshape(1,-1), sst_shape)

sea_mask = xr.where(~sst_da.isnull(), 1, 0)
sst_mask = ~np.isnan(sst_array)

lat_vals = latitude_array[sst_mask]
lon_vals = longitude_array[sst_mask]

sea_points = np.array(list(zip(lat_vals, lon_vals)))


# %%
def heuristic_boundary_layer_height(ds: xr.Dataset, 
                                    threshold: float=1.0):
    
    """Heuristic method to calculate boundary layer height, based on the location that 
    potential temperature changes significantly
    
    Args:
        ds (xr.Dataset): Dataset containing temperature, geopotential, 2m_temperature, on pressure levels containing
        1000hPa, 975hPa, 950hPa, +...

    Returns:
        _type_: _description_
    """
    
    ds = ds.sortby('level', ascending=False)
    pressure_levels = [int(item) for item in sorted(ds['level'].values, reverse=True)]
    
    da_pt = ds[f'temperature'].copy() 
    for p in pressure_levels:
        da_pt.loc[{'level': p}] = potential_temperature(temp=ds['temperature'].sel(level=p), pressure=p)

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
            large_diff_1000_975 = np.abs(da_pt.sel(level=1000) - da_pt.sel(level=975)) > threshold
            large_diff_2m_975 = np.abs(ds['2m_temperature'] - da_pt.sel(level=975)) > threshold
            large_diff_975_950 = np.abs(da_pt.sel(level=975) - da_pt.sel(level=950)) > threshold
            
            check_1000hpa_positive = ds[f'geopotential'].sel(level=1000) > 0
            
            logical_condition_1 = np.logical_and(check_1000hpa_positive, np.logical_and(large_diff_2m_1000, large_diff_1000_975))
            logical_condition_2 = np.logical_and(~check_1000hpa_positive, np.logical_and(large_diff_2m_975, large_diff_975_950))
            
            height_da = xr.where(np.logical_or(logical_condition_1, logical_condition_2), 10.0, np.nan)
            pressure_da = xr.where(np.logical_or(logical_condition_1, logical_condition_2), pressure_level_da, np.nan)
            
        elif n == len(pressure_levels):
            values_are_null = height_da.isnull()
            height_da = xr.where(values_are_null, ds['geopotential'].sel(level=p) / 9.81, height_da)
            pressure_da = xr.where(values_are_null, pressure_level_da, pressure_da)
        else:
            
            large_diff_here = np.abs(da_pt.sel(level=p) - da_pt.sel(level=pressure_levels[n])) > threshold
            check_positive = ds['geopotential'].sel(level=p) > 0

            values_are_null = height_da.isnull()
            
            logical_condition =  np.logical_and(check_positive, np.logical_and(large_diff_here, values_are_null))
            
            height_da = xr.where(logical_condition, ds['geopotential'].sel(level=p) / 9.81, height_da)
            pressure_da = xr.where(logical_condition, pressure_level_da, pressure_da)

    return height_da, pressure_da


# %%
sst_mask

# %%
smoothed_height_vals = ndimage.uniform_filter(height_da.fillna(height_da.mean()), 3)
smoothed_height_vals[sst_mask] = np.nan

# %%

# %%
ndimage.uniform_filter(height_da.values, 3)

# %%
time_ixs = range(5)
main_ds = xr.load_dataset('/Users/bobbyantonio/repos/AirSeaFluxCode/sample_boundary_height_data.nc')

fig, axs = plt.subplots(len(time_ixs), 4, figsize=(15,5*len(time_ixs)))

for time_ix in range(5):

    # Create heuristic function to calculate boundary layer height 
    bl_ds = main_ds.isel(time=time_ix).sortby('level', ascending=False)
    bl_ds = bl_ds.drop_vars(['number', 'expver'])
    pressure_levels = [int(item) for item in sorted(bl_ds['level'].values, reverse=True)]
    height_da, pressure_da = heuristic_boundary_layer_height(bl_ds, threshold=1.0)
    height_da = xr.where(sst_da.isnull(), np.nan, height_da)

    lat_range = [-90, 90]
    lon_range = [0,360]
    lat_vals = np.arange(*lat_range, 2.0)
    lon_vals = np.arange(*lon_range, 2.0)

    subset_ds = bl_ds.sel(longitude=lon_vals).sel(latitude=lat_vals)
    sst_mask = ~sst_da.sel(longitude=lon_vals).sel(latitude=lat_vals).isnull()


    height_da.sel(longitude=lon_vals).sel(latitude=lat_vals).plot.pcolormesh(ax=axs[time_ix,0], vmin=0, vmax=3000)
    axs[time_ix,0].set_title(pd.to_datetime(main_ds['time'].values[time_ix].item()).strftime('%Y-%m-%d'))
    
    xr.where(sst_mask, bl_ds['boundary_layer_height'], np.nan).plot.pcolormesh(ax=axs[time_ix,1], vmin=0, vmax=3000)
    axs[time_ix,1].set_title('ERA5')
    
    smoothed_height_vals = ndimage.uniform_filter(height_da.fillna(height_da.mean()).values, 3)
    smoothed_height_vals[~sst_mask] = np.nan
    
    axs[time_ix,2].pcolormesh(smoothed_height_vals, vmin=0, vmax=3000)
    axs[time_ix,2].set_title('smoothed')
    
    error = (height_da.sel(longitude=lon_vals).sel(latitude=lat_vals) - subset_ds['boundary_layer_height'])
    xr.where(sst_mask, error, np.nan).plot.pcolormesh( vmin=-1500, vmax=1500, ax=axs[time_ix,3], cmap='RdBu_r')
    axs[time_ix,3].set_title('Diff=')

# axs[1,0] = bl_ds['siconc'].sel(longitude=lon_vals).sel(latitude=lat_vals).plot()
# axs[1,1] = xr.where(sst_mask, large_diff_above.astype(np.int32), np.nan).plot.pcolormesh(ax=axs[1,1])

# %%
# Plot some examples
# lat = np.random.choice(np.arange(-50, -40, 2),1)
# lon = np.random.choice(np.arange(50,100, 2),1)

time_ix = 0

lat, lon = -65, 352
subset_ds = main_ds.isel(time=time_ix).sel(latitude=lat, method='nearest').sel(longitude=lon, method='nearest')


temperature_vals = np.concat([subset_ds['2m_temperature'].values.flatten(), subset_ds['temperature'].values.flatten()])
height_vals = [2] + [subset_ds['geopotential'].sel(level=p).item() / 9.81 for p in pressure_levels]
pot_temp_vals = potential_temperature(temp=temperature_vals, pressure=np.array([1014] + pressure_levels, dtype=np.float32))

fig, ax = plt.subplots(2,1, figsize=(10,10))

ax[0].scatter(pot_temp_vals, height_vals)
ax[0].hlines(y=subset_ds['boundary_layer_height'].item(), xmin=pot_temp_vals.min(), xmax=pot_temp_vals.max(), color='r')
ax[0].hlines(y=height_da.sel(latitude=lat, method='nearest').sel(longitude=lon, method='nearest').item(), xmin=pot_temp_vals.min(), xmax=pot_temp_vals.max(), color='b', linestyle='--')



# %%
# Make update to array, only to levels below the boundary height, and over sea
time_ix = 0
bl_ds = main_ds.isel(time=time_ix).sortby('pressure_level', ascending=False)
bl_ds = bl_ds.drop_vars(['number', 'expver'])
pressure_levels = [int(item) for item in sorted(ds['pressure_level'].values, reverse=True)]
height_da, pressure_da = heuristic_boundary_layer_height(bl_ds, threshold=1.0)

# %%
original_ds = bl_ds.copy()
updated_ds = bl_ds.copy()

# da_pt = original_ds[f'temperature'].copy() 
# for p in pressure_levels:
#     da_pt.loc[{'level': p}] = potential_temperature(temp=original_ds['temperature'].sel(level=p), pressure=p)

# %%
updated_ds = bl_ds.copy()

p = 1000
delta_pot_temp = 5.0
delta_temp = potential_temperature_to_temperature(delta_pot_temp, pressure=p) # Valid since pressure levels are staying the same. Do we need to account for varying sea level pressure though?

logical_condition = np.logical_and(sea_mask == 1, pressure_da == 1000)
tmp_da = xr.where(logical_condition, original_ds['temperature'].sel(level=p) + delta_temp, original_ds['temperature'].sel(level=p))
# original_ds['2m_temperature'] 

# %%
fig, axs = plt.subplots(3,1, figsize=(6,12))

xr.where(~sst_da.isnull(), bl_ds['temperature'].sel(level=p), np.nan).plot(ax=axs[0])

xr.where(~sst_da.isnull(), updated_ds['temperature'].sel(level=p), np.nan).plot(ax=axs[1])

xr.where(~sst_da.isnull(), logical_condition.astype(np.int8), np.nan).plot(ax=axs[2])

# %%
for p in pressure_levels:
    
    p_within_boundary_layer = pressure_da <= p
    
    new_ds
    

# %%
tmp_array = np.abs(bl_ds['2m_temperature'] - bl_ds['temperature'].sel(level=1000)).values

abs_diffs = [np.where(tmp_array <= 0.5, 1014, pressure_levels[-1])]

for pn in range(len(pressure_levels) - 1):
    tmp_array = np.abs(bl_ds['temperature'].sel(level=pressure_levels[pn]) - bl_ds['temperature'].sel(level=pressure_levels[pn+1])).values
    abs_diffs.append(np.where(tmp_array <= 0.5, pressure_levels[pn], pressure_levels[-1]))

abs_diffs.append(np.ones_like(tmp_array)* pressure_levels[-1])

abs_diffs = np.stack(abs_diffs)

# %%
num_points_per_date = 1
sea_point_ix = np.random.choice(range(len(sea_points)), num_points_per_date, replace=False)

lat, lon = sea_points[sea_point_ix.item(),:]

# %%


def calculate_things(ds, pressure_levels, kappa = 0.286):
    temp_vals = []
    q_vals = []
    height_vals = []
    for p in sorted(pressure_levels, reverse=True):
        temp_vals.append(ds['temperature'].sel(level=p).values.flatten())
        q_vals.append(ds['specific_humidity'].sel(level=p).values.flatten())
        height_vals.append(ds[f'height_{p}'].values.flatten())

    temp_vals = np.concatenate(temp_vals)
    q_vals = np.concatenate(q_vals)
    potential_temp_vals = potential_temperature(temp=temp_vals, pressure=np.array(pressure_levels))
    height_vals = np.concatenate(height_vals)

    temp_2m = ds['2m_temperature'].values.item()
    
    temp_vals_all = np.concatenate([[temp_2m], temp_vals])
    q_vals_all = np.concatenate([[None], q_vals])
    height_vals_all = np.concatenate([[ 2], height_vals])
    potential_temp_vals_all = np.concatenate([[temp_2m*(1014 / (ds['mean_sea_level_pressure'].item()*0.01))**kappa], potential_temp_vals])

    return temp_vals_all, potential_temp_vals_all, q_vals_all, height_vals_all


# %%
num_points_per_date = 1
sea_point_ix = np.random.choice(range(len(sea_points)), num_points_per_date, replace=False)

lat, lon = sea_points[sea_point_ix.item(),:]

ds_subset_t0 = bl_ds.sel(longitude=lon).sel(latitude=lat).sel(time=start)

temp_vals_all, potential_temp_vals_all, q_vals_all, height_vals_all = calculate_things(ds_subset_t0, pressure_levels=pressure_levels)

sst_avg = (ds_subset_t0['sea_surface_temperature'].values.item() + ds_subset_t0['sea_surface_temperature'].values.item())/2.0

potential_temps_diff = np.abs([potential_temp_vals_all[n+1] - potential_temp_vals_all[n] for n in range(len(potential_temp_vals_all)-1)])

inferred_boundary_layer_height = height_vals_all[np.where(potential_temps_diff >= 1.0)[0].min()]

figure, ax = plt.subplots(3, figsize=(5,15))
ax[0].scatter(potential_temp_vals_all, height_vals_all)

ax[0].set_ylabel('Height (m)')
ax[0].set_xlabel('Potential temperature (K)')
ax[0].hlines(ds_subset_t0['boundary_layer_height'], xmin=potential_temp_vals_all.min(), xmax=potential_temp_vals_all.max(), label='t0 boundary layer height')
ax[0].hlines(inferred_boundary_layer_height,  xmin=potential_temp_vals_all.min(), xmax=potential_temp_vals_all.max(), color='r', label='Heuristic BL')
ax[0].legend()

ax[1].scatter(temp_vals_all, height_vals_all)

ax[1].scatter([sst_avg], [0], color='g')
ax[1].hlines(ds_subset_t0['boundary_layer_height'], xmin=temp_vals_all.min(), xmax=temp_vals_all.max())
ax[1].hlines(ds_subset_t1['boundary_layer_height'],  xmin=temp_vals_all.min(), xmax=temp_vals_all.max(), color='r')
ax[1].set_ylabel('Height (m)')
ax[1].set_xlabel('Temperature (K)')

ax[2].scatter(q_vals_all, height_vals_all)
ax[2].scatter(q_vals_all_1, height_vals_all_1, marker='+', color='r')
ax[2].set_ylabel('Height (m)')
ax[2].set_xlabel('Specific humidity (kg/kg)')

