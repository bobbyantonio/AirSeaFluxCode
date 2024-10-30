# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: airseaflux
#     language: python
#     name: python3
# ---

# ## Introduction
#
# AirSeaFluxCode is developed to provide an easy and accessible way to calculate turbulent surface fluxes (TSFs) from a small number of bulk variables and for a viariety of bulk algorithms. 
#
# By running AirSeaFluxCode you can compare different bulk algorithms and to also investigate the effect choices within the implementation of each parameterisation have on the TSFs estimates. 
#

# ### Getting started
#
# Let's first import the basic python packages we will need for reading in our input data, to perform basic statistics  and plotting

# first import all packages you might need
# %matplotlib inline
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import numpy as np
import pandas as pd
import datetime
from glob import glob
from tabulate import tabulate
from AirSeaFluxCode import AirSeaFluxCode

# ## Air sea flux code on gridded data

# +
# TODO: precip needs to be aggregated

# +
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
ds['surface_stress_magnitude'] = np.sqrt(ds['iews']**2 + ds['inss']**2)
# ds['q'] = ds['q']*1000 

# Interpolate humidity to surface
ds['q_surface'] = ds['q_1000'] + (ds['q_1000'] - ds['q_975']) * (ds['msl'] - 1000) / 25

# ds = ds.sel(longitude=[255.5, 255.75], method='nearest').sel(latitude=[9.5, 9.75], method='nearest')
df = ds.to_dataframe().reset_index()

df = df[~np.isnan(df['sst'])].reset_index()
df.head()

# -

print(df.columns)

# +
# Pick a random point and plot it

row = df.sample(1)
fig, ax = plt.subplots(1,1)

q_vals = [row['q_surface'].item(), row['q_1000'].item(), row['q_975'].item()]
pressure_vals = [row['msl'].item(), 1000, 975]
height_vals = [0, row['z_1000'].item(), row['z_975'].item()]
plt.scatter(height_vals, q_vals)
# -

row

inDt = pd.read_csv("Test_Data/data_all.csv")

inDt[inDt['Date'] == 20070203]

# +
# Get the data into the right format
# -

in_df = pd.DataFrame(dict(latitude=df['latitude'],
                        longitude=df['longitude'],
                        time = df['time'],
                    spd=df['wind_speed'].to_numpy(), 
                     T=df['t2m'].to_numpy(), 
                     SST_skin=df['skt'].to_numpy(), 
                     lat=df['latitude'].to_numpy(),
                     hum=df['q_surface'].to_numpy(), 
                     P=df['msl'].to_numpy()))

inDt[inDt['Date'] == 20070203]

in_df[(in_df['longitude']==255.75) & (df['latitude']==9.75)]

res = AirSeaFluxCode(spd=df['wind_speed'].to_numpy(), 
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
                     out_var = ("tau", "sensible", "latent", "cd", "rho", "uref"))

res['rho']

# check tau is calculated as you expect
res['tau_calc'] = ((np.power(res['uref'],2))*res['cd']*res['rho'])

res = pd.concat([df[['latitude', 'longitude']], res], axis=1)

res_ds = res.set_index(['latitude', 'longitude']).to_xarray()

# +
num_rows = 2
num_cols = 2

fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*5))
res_ds['sensible'].plot.imshow(cmap='RdBu_r', ax=ax[0,0], vmin=-400, vmax=400)

sshf_da = ds['msshf'].where(~ds['sst'].isnull())
sshf_da.plot.imshow(ax=ax[0,1], vmin=-400, vmax=400, cmap='RdBu_r',)

res_ds['tau'].plot.imshow(ax=ax[1,0], vmin=0, vmax=4)
tau_da = ds['surface_stress_magnitude'].where(~ds['sst'].isnull())
tau_da.plot.imshow(ax=ax[1,1], vmin=0, vmax=4)
# -

res_ds['tau_calc'].plot.hist()

ds['surface_stress_magnitude'].where(~ds['sst'].isnull()).plot.hist()

res['tau'].max()

ds['surface_stress_magnitude'].where(~ds['sst'].isnull()).max()

# ### AirSeaFluxCode examples
#
# AirSeaFluxCode is set up to run in its default setting with a minimum number of input variables, namely wind speed; air temperature; and sea surface temperature. Let's load the code, import some real data composed for testing it (Research vessel data) and run AirSeaFluxCode with default settings (output height 10m, cool skin/warm layer corrections turned off, bulk algorithm Smith 1988, gustiness on, saturation vapour pressure following Buck (2012), tolerance limits set for both flux estimates and height adjusted variables (['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1]), number of iterations are ten, non converged points are set to missing and Monin-Obukhov stability length is calculated following the ECMWF implementation.

inDt = pd.read_csv("Test_Data/data_all.csv")

inDt = inDt[inDt['Date'] == 20070203]

# +
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
# -

t

np.abs(res['sensible']).max()

hu

ht

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











# res is the output of AirSeaFluxCode which is a dataFrame with keys: "tau", "sensible", "latent", "u10n", "t10n", "q10n". Let's plot the flux estimates.

fig, ax = plt.subplots(3, 1, sharex=True, sharey=False)
fig.tight_layout()
ax[0].plot(res["tau"], "-", color="grey", linewidth=1, alpha = 0.8)
ax[1].plot(res["sensible"], "-", color="grey", linewidth=1, alpha = 0.8)
ax[2].plot(res["latent"], "-", color="grey", linewidth=1, alpha = 0.8)
ax[0].set_ylabel('tau (Nm$^{-2}$)')
ax[1].set_ylabel('shf (Wm$^{-2}$)')
ax[2].set_ylabel('lhf (Wm$^{-2}$)')
plt.show()

# You can save the output in a csv file

res.to_csv("test_AirSeaFluxCode.csv")

# and generate some statistics which you can save in a txt file

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


