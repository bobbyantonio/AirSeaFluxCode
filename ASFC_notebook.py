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
ds['ewss'] = ds['ewss'] / (60*60) # Convert from integrated stress to mean stress
ds['ewss'] = ds['nsss'] / (60*60)
ds['surface_stress_magnitude'] = np.sqrt(ds['iews']**2 + ds['inss']**2)

ds = ds.rename({'u10': '10m_u_component_of_wind',
                'v10': '10m_v_component_of_wind',
                'z': 'geopotential',
                'q': 'specific_humidity',
                'msl': 'mean_sea_level_pressure',
                'sst': 'sea_surface_temperature',
                't2m': '2m_temperature',
                'skt': 'skin_temperature'}).drop_vars('expver').drop_vars('number')

def dataset_to_flux_dataframe(ds):
    plevels = ds['pressure_level'].values
    ds['specific_humidity'] = ds['specific_humidity'] * 1000 # Convert to g/kg

    # Expand pressure levels out
    for var in ['geopotential', 'specific_humidity']:
        for p in plevels:
            ds[f'{var}_{int(p)}'] = ds[var].sel(pressure_level=p)
            
    ds = ds.drop_vars('specific_humidity').drop_vars('geopotential').drop_vars('pressure_level')

    ds['wind_speed'] = np.sqrt(ds['10m_u_component_of_wind']**2 + ds['10m_v_component_of_wind']**2)

    ds['mean_sea_level_pressure'] = ds['mean_sea_level_pressure'] / 100 # Convert to hPA

    # Interpolate humidity to surface
    ds['specific_humidity_surface'] = ds['specific_humidity_1000'] + (ds['specific_humidity_1000'] - ds['specific_humidity_975']) * (ds['mean_sea_level_pressure'] - 1000) / 25

    # ds = ds.sel(longitude=[255.5, 255.75], method='nearest').sel(latitude=[9.5, 9.75], method='nearest')
    df = ds.to_dataframe().reset_index()

    df = df[~np.isnan(df['sea_surface_temperature'])].reset_index()
    return df



# %%
df = dataset_to_flux_dataframe(ds)
df.head()

# %% [markdown]
# ## Plot fluxes

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
# Pick a random point and plot it

row = df.sample(1)
fig, ax = plt.subplots(1,1)

q_vals = [row['q_surface'].item(), row['q_1000'].item(), row['q_975'].item()]
pressure_vals = [row['msl'].item(), 1000, 975]
height_vals = [0, row['z_1000'].item(), row['z_975'].item()]
plt.scatter(height_vals, q_vals)

# %%
inDt = pd.read_csv("Test_Data/data_all.csv")

# %%
inDt[inDt['Date'] == 20070203]

# %%
# Get the data into the right format

# %%
in_df = pd.DataFrame(dict(latitude=df['latitude'],
                        longitude=df['longitude'],
                        time = df['time'],
                    spd=df['wind_speed'].to_numpy(), 
                     T=df['t2m'].to_numpy(), 
                     SST_skin=df['skt'].to_numpy(), 
                     lat=df['latitude'].to_numpy(),
                     hum=df['q_surface'].to_numpy(), 
                     P=df['msl'].to_numpy()))

# %%
inDt[inDt['Date'] == 20070203]

# %%
in_df[(in_df['longitude']==255.75) & (df['latitude']==9.75)]

# %%
res_sst = AirSeaFluxCode.AirSeaFluxCode(spd=df['wind_speed'].to_numpy(), 
                     T=df['2m_temperature'].to_numpy(), 
                     SST=df['sea_surface_temperature'].to_numpy(), 
                     SST_fl="bulk", 
                     meth="ecmwf", 
                     lat=df['latitude'].to_numpy(),
                     hin=np.array([10, 2]), 
                     hum=('q', df['specific_humidity_surface'].to_numpy()), 
                     hout=10,
                     maxiter=10,
                     P=df['mean_sea_level_pressure'].to_numpy(), 
                     cskin=1, 
                     Rs=df['msdwswrf'].to_numpy(),
                     tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1], 
                     L="tsrv", 
                     wl=1,
                     out_var = ("tau", "sensible", "latent", "cd", "rho", "uref"))
res_sst = pd.concat([df[['latitude', 'longitude']], res_sst], axis=1)
res_sst_ds = res_sst.set_index(['latitude', 'longitude']).to_xarray()

# %%
res_ssst = AirSeaFluxCode.AirSeaFluxCode(spd=df['wind_speed'].to_numpy(), 
                     T=df['2m_temperature'].to_numpy(), 
                     SST=df['skin_temperature'].to_numpy(), 
                     SST_fl="skin", 
                     meth="ecmwf", 
                     lat=df['latitude'].to_numpy(),
                     hin=np.array([10, 2]), 
                     hum=('q', df['specific_humidity_surface'].to_numpy()), 
                     hout=10,
                     maxiter=10,
                     P=df['mean_sea_level_pressure'].to_numpy(), 
                     cskin=0, 
                     Rs=df['msdwswrf'].to_numpy(),
                     tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1], 
                     L="tsrv", 
                     wl=0,
                     out_var = ("tau", "sensible", "latent", "cd", "rho", "uref"))
res_ssst = pd.concat([df[['latitude', 'longitude']], res_ssst], axis=1)
res_ssst_ds = res_ssst.set_index(['latitude', 'longitude']).to_xarray()

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

print('Num nan tau: ', res_2deg['tau'].isna().sum(), ' / ', len(res_2deg))

# %%
# check tau is calculated as you expect
res['tau_calc'] = ((np.power(res['uref'],2))*res['cd']*res['rho'])

# %%
num_rows = 4
num_cols = 2

fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*5))
res_ssst_ds['sensible'].plot.imshow(cmap='RdBu_r', ax=ax[0,0], vmin=-400, vmax=400)
sshf_da = ds['msshf'].where(~ds['sea_surface_temperature'].isnull())
sshf_da.plot.imshow(ax=ax[0,1], vmin=-400, vmax=400, cmap='RdBu_r',)

res_ssst_ds['tau'].plot.imshow(ax=ax[1,0], vmin=0, vmax=4)
tau_da = ds['surface_stress_magnitude'].where(~ds['sea_surface_temperature'].isnull())
tau_da.plot.imshow(ax=ax[1,1], vmin=0, vmax=4)

res_ssst_ds['latent'].plot.imshow(ax=ax[2,0], vmin=-600, vmax=600, cmap='RdBu_r',)
tau_da = ds['mslhf'].where(~ds['sea_surface_temperature'].isnull())
tau_da.plot.imshow(ax=ax[2,1], vmin=-600, vmax=600, cmap='RdBu_r',)

ds['siconc'].plot.imshow(ax=ax[3,0])

print('Num nan tau: ', res_ssst['tau'].isna().sum(), ' / ', len(res_ssst))
print('Fraction nan tau: ', res_ssst['tau'].isna().sum()/len(res_ssst))

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
res_ds['tau_calc'].plot.hist()

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
