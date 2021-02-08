"""
example of running AirSeaFluxCode with
1. R/V data (data_all.nc) or
2. one day era5 hourly data (era5_r360x180.nc)
compute fluxes
output NetCDF4
and statistics in stats.txt

@author: sbiri
"""
#%% import packages
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from AirSeaFluxCode import AirSeaFluxCode
import time
from tabulate import tabulate
#%%
def reject_outliers(data, m=2):
    x = np.copy(data)
    x = np.where(np.abs(x - np.nanmean(x)) < m*np.nanstd(x),
                    x, np.nan)
    return x


def run_ASFC(inF, outF, gustIn, cskinIn, tolIn, meth):
    """


    Parameters
    ----------
    inF : str
        input filename either data_all.nc or era5_r360x180.nc
    outF : str
        output filename
    gustIn : float
        gustiness option e.g. [1, 1.2, 800]
    cskinIn : int
        cool skin option input 0 or 1
    tolIn : float
        tolerance input option e.g. ['all', 0.01, 0.01, 5e-05, 0.01, 1, 1]
    meth : str
        parametrisation method option

    Returns
    -------
    res : float
        AirSeaFluxCode output
    lon : float
        longitude from input netCDF file
    lat : float
        latitude from input netCDF file

    """
    if (inF == "data_all.nc"):
        #%% load data_all
        fid=nc.Dataset('data_all.nc')
        date = np.array(fid.variables["date"])
        lon = np.array(fid.variables["lon"])
        lat = np.array(fid.variables["lat"])
        spd = np.array(fid.variables["spd"])
        t = np.array(fid.variables["t"])
        sst = np.array(fid.variables["sst"])
        rh = np.array(fid.variables["rh"])
        p = np.array(fid.variables["p"])
        sw = np.array(fid.variables["sw"])
        hu = np.array(fid.variables["spd"].getncattr("height"))
        ht = np.array(fid.variables["t"].getncattr("height"))
        hin = np.array([hu, ht, ht])
        del hu, ht
        fid.close()
        del fid
        #%% run AirSeaFluxCode
        res = AirSeaFluxCode(spd, t, sst, lat=lat, hum=['rh', rh], P=p,
                             hin=hin, Rs=sw, tol=tolIn, gust=gustIn,
                             cskin=cskinIn, meth=meth, out=1, L="ecmwf", n=30)
        #%% save NetCDF4
        fid = nc.Dataset(outF,'w', format='NETCDF4')
        fid.createDimension('lon', len(lon))
        fid.createDimension('lat', len(lat))
        fid.createDimension('time', None)
        longitude = fid.createVariable('lon', 'f4', 'lon')
        latitude = fid.createVariable('lat', 'f4', 'lat')
        Date = fid.createVariable('Date', 'i4', 'time')
        tau = fid.createVariable('tau', 'f4', 'time')
        sensible = fid.createVariable('shf', 'f4', 'time')
        latent = fid.createVariable('lhf', 'f4', 'time')
        monob = fid.createVariable('MO', 'f4', 'time')
        cd = fid.createVariable('cd', 'f4', 'time')
        cdn = fid.createVariable('cdn', 'f4', 'time')
        ct = fid.createVariable('ct', 'f4', 'time')
        ctn = fid.createVariable('ctn', 'f4', 'time')
        cq = fid.createVariable('cq', 'f4', 'time')
        cqn = fid.createVariable('cqn', 'f4', 'time')
        tsrv = fid.createVariable('tsrv', 'f4', 'time')
        tsr = fid.createVariable('tsr', 'f4', 'time')
        qsr = fid.createVariable('qsr', 'f4', 'time')
        usr = fid.createVariable('usr', 'f4', 'time')
        psim = fid.createVariable('psim', 'f4', 'time')
        psit = fid.createVariable('psit', 'f4', 'time')
        psiq = fid.createVariable('psiq', 'f4', 'time')
        u10n = fid.createVariable('u10n', 'f4', 'time')
        t10n = fid.createVariable('t10n', 'f4', 'time')
        tv10n = fid.createVariable('tv10n', 'f4', 'time')
        q10n = fid.createVariable('q10n', 'f4', 'time')
        zo = fid.createVariable('zo', 'f4', 'time')
        zot = fid.createVariable('zot', 'f4', 'time')
        zoq = fid.createVariable('zoq', 'f4', 'time')
        urefs = fid.createVariable('uref', 'f4', 'time')
        trefs = fid.createVariable('tref', 'f4', 'time')
        qrefs = fid.createVariable('qref', 'f4', 'time')
        itera = fid.createVariable('iter', 'i4', 'time')
        dter = fid.createVariable('dter', 'f4', 'time')
        dqer = fid.createVariable('dqer', 'f4', 'time')
        dtwl = fid.createVariable('dter', 'f4', 'time')
        qair = fid.createVariable('qair', 'f4', 'time')
        qsea = fid.createVariable('qsea', 'f4', 'time')
        Rl = fid.createVariable('Rl', 'f4', 'time')
        Rs = fid.createVariable('Rs', 'f4', 'time')
        Rnl = fid.createVariable('Rnl', 'f4', 'time')

        longitude[:] = lon
        latitude[:] = lat
        Date[:] = date
        tau[:] = res[0]
        sensible[:] = res[1]
        latent[:] = res[2]
        monob[:] = res[3]
        cd[:] = res[4]
        cdn[:] = res[5]
        ct[:] = res[6]
        ctn[:] = res[7]
        cq[:] = res[8]
        cqn[:] = res[9]
        tsrv[:] = res[10]
        tsr[:] = res[11]
        qsr[:] = res[12]
        usr[:] = res[13]
        psim[:] = res[14]
        psit[:] = res[15]
        psiq[:] = res[16]
        u10n[:] = res[17]
        t10n[:] = res[18]
        tv10n[:] = res[19]
        q10n[:] = res[20]
        zo[:] = res[21]
        zot[:] = res[22]
        zoq[:] = res[23]
        urefs[:] = res[24]
        trefs[:] = res[25]
        qrefs[:] = res[26]
        itera[:] = res[27]
        dter[:] = res[28]
        dqer[:] = res[29]
        dtwl[:] = res[30]
        qair[:] = res[31]
        qsea[:] = res[32]
        Rl = res[33]
        Rs = res[34]
        Rnl = res[35]
        longitude.long_name = 'Longitude'
        longitude.units = 'degrees East'
        latitude.long_name = 'Latitude'
        latitude.units = 'degrees North'
        Date.long_name = "calendar date"
        Date.units = "YYYYMMDD UTC"
        tau.long_name = 'Wind stress'
        tau.units = 'N/m^2'
        sensible.long_name = 'Sensible heat fluxe'
        sensible.units = 'W/m^2'
        latent.long_name = 'Latent heat flux'
        latent.units = 'W/m^2'
        monob.long_name = 'Monin-Obukhov length'
        monob.units = 'm'
        cd.long_name = 'Drag coefficient'
        cd.units = ''
        cdn.long_name = 'Neutral Drag coefficient'
        cdn.units = ''
        ct.long_name = 'Heat exchange coefficient'
        ct.units = ''
        ctn.long_name = 'Neutral Heat exchange coefficient'
        ctn.units = ''
        cq.long_name = 'Moisture exchange coefficient'
        cq.units = ''
        cqn.long_name = 'Neutral Moisture exchange coefficient'
        cqn.units = ''
        tsrv.long_name = 'star virtual temperature'
        tsrv.units = 'degrees Celsius'
        tsr.long_name = 'star temperature'
        tsr.units = 'degrees Celsius'
        qsr.long_name = 'star specific humidity'
        qsr.units = 'gr/kgr'
        usr.long_name = 'friction velocity'
        usr.units = 'm/s'
        psim.long_name = 'Momentum stability function'
        psit.long_name = 'Heat stability function'
        u10n.long_name = '10m neutral wind speed'
        u10n.units = 'm/s'
        t10n.long_name = '10m neutral temperature'
        t10n.units = 'degrees Celsius'
        tv10n.long_name = '10m neutral virtual temperature'
        tv10n.units = 'degrees Celsius'
        q10n.long_name = '10m neutral specific humidity'
        q10n.units = 'gr/kgr'
        zo.long_name = 'momentum roughness length'
        zo.units = 'm'
        zot.long_name = 'temperature roughness length'
        zot.units = 'm'
        zoq.long_name = 'moisture roughness length'
        zoq.units = 'm'
        urefs.long_name = 'wind speed at ref height'
        urefs.units = 'm/s'
        trefs.long_name = 'temperature at ref height'
        trefs.units = 'degrees Celsius'
        qrefs.long_name = 'specific humidity at ref height'
        qrefs.units = 'gr/kgr'
        qair.long_name = 'specific humidity of air'
        qair.units = 'gr/kgr'
        qsea.long_name = 'specific humidity over water'
        qsea.units = 'gr/kgr'
        itera.long_name = 'number of iterations'
        fid.close()
        #%% delete variables
        del longitude, latitude, Date, tau, sensible, latent, monob, cd, cdn
        del ct, ctn, cq, cqn, tsrv, tsr, qsr, usr, psim, psit, psiq, u10n, t10n
        del tv10n, q10n, zo, zot, zoq, urefs, trefs, qrefs, itera, dter, dqer
        del qair, qsea, Rl, Rs, Rnl
        del t, rh, date, p, sw, spd, hin

    elif (inF == 'era5_r360x180.nc'):
        #%% load era5_r360x180.nc
        fid = nc.Dataset(inF)
        lon = np.array(fid.variables["lon"])
        lat = np.array(fid.variables["lat"])
        T = np.array(fid.variables["t2m"])
        tim = np.array(fid.variables["time"])
        Td = np.array(fid.variables["d2m"])
        sst = np.array(fid.variables["sst"])
        sst = np.where(sst < -100, np.nan, sst)
        p = np.array(fid.variables["msl"])/100 # to set hPa
        lw = np.array(fid.variables["strd"])/60/60
        sw = np.array(fid.variables["ssrd"])/60/60
        u = np.array(fid.variables["u10"])
        v = np.array(fid.variables["v10"])
        lsm = np.array(fid.variables["lsm"])
        fid.close()
        spd = np.sqrt(np.power(u, 2)+np.power(v, 2))
        del u, v, fid
        lsm = np.where(lsm > 0, np.nan, 1) # reverse 0 on land 1 over ocean
        hin = np.array([10, 2, 2])
        latIn = np.tile(lat, (len(lon), 1)).T.reshape(len(lon)*len(lat))
        #%% run AirSeaFluxCode
        res = np.zeros((len(tim),len(lon)*len(lat), 36))
        # reshape input and run code
        for x in range(len(tim)):
            a = AirSeaFluxCode(spd.reshape(len(tim), len(lon)*len(lat))[x, :],
                               T.reshape(len(tim), len(lon)*len(lat))[x, :],
                               sst.reshape(len(tim), len(lon)*len(lat))[x, :],
                               lat=latIn,
                               hum=['Td', Td.reshape(len(tim), len(lon)*len(lat))[x, :]],
                               P=p.reshape(len(tim), len(lon)*len(lat))[x, :],
                               hin=hin,
                               Rs=sw.reshape(len(tim), len(lon)*len(lat))[x, :],
                               Rl=lw.reshape(len(tim), len(lon)*len(lat))[x, :],
                               gust=gustIn, cskin=cskinIn, tol=tolIn, qmeth='WMO',
                               meth=meth, n=30, L="ecmwf")
            res[x, :, :] = a.T
            del a
        #%% save NetCDF4
        fid = nc.Dataset(outF,'w', format='NETCDF4')
        fid.createDimension('lon', len(lon))
        fid.createDimension('lat', len(lat))
        fid.createDimension('time', None)
        longitude = fid.createVariable('lon', 'f4', 'lon')
        latitude = fid.createVariable('lat', 'f4', 'lat')
        Date = fid.createVariable('Date', 'i4', 'time')
        tau = fid.createVariable('tau', 'f4', ('time','lat','lon'))
        sensible = fid.createVariable('shf', 'f4', ('time','lat','lon'))
        latent = fid.createVariable('lhf', 'f4', ('time','lat','lon'))
        monob = fid.createVariable('MO', 'f4', ('time','lat','lon'))
        cd = fid.createVariable('cd', 'f4', ('time','lat','lon'))
        cdn = fid.createVariable('cdn', 'f4', ('time','lat','lon'))
        ct = fid.createVariable('ct', 'f4', ('time','lat','lon'))
        ctn = fid.createVariable('ctn', 'f4', ('time','lat','lon'))
        cq = fid.createVariable('cq', 'f4', ('time','lat','lon'))
        cqn = fid.createVariable('cqn', 'f4', ('time','lat','lon'))
        tsrv = fid.createVariable('tsrv', 'f4', ('time','lat','lon'))
        tsr = fid.createVariable('tsr', 'f4', ('time','lat','lon'))
        qsr = fid.createVariable('qsr', 'f4', ('time','lat','lon'))
        usr = fid.createVariable('usr', 'f4', ('time','lat','lon'))
        psim = fid.createVariable('psim', 'f4', ('time','lat','lon'))
        psit = fid.createVariable('psit', 'f4', ('time','lat','lon'))
        psiq = fid.createVariable('psiq', 'f4', ('time','lat','lon'))
        u10n = fid.createVariable('u10n', 'f4', ('time','lat','lon'))
        t10n = fid.createVariable('t10n', 'f4', ('time','lat','lon'))
        tv10n = fid.createVariable('tv10n', 'f4', ('time','lat','lon'))
        q10n = fid.createVariable('q10n', 'f4', ('time','lat','lon'))
        zo = fid.createVariable('zo', 'f4', ('time','lat','lon'))
        zot = fid.createVariable('zot', 'f4', ('time','lat','lon'))
        zoq = fid.createVariable('zoq', 'f4', ('time','lat','lon'))
        urefs = fid.createVariable('uref', 'f4', ('time','lat','lon'))
        trefs = fid.createVariable('tref', 'f4', ('time','lat','lon'))
        qrefs = fid.createVariable('qref', 'f4', ('time','lat','lon'))
        itera = fid.createVariable('iter', 'i4', ('time','lat','lon'))
        dter = fid.createVariable('dter', 'f4', ('time','lat','lon'))
        dqer = fid.createVariable('dqer', 'f4', ('time','lat','lon'))
        dtwl = fid.createVariable('dtwl', 'f4', ('time','lat','lon'))
        qair = fid.createVariable('qair', 'f4', ('time','lat','lon'))
        qsea = fid.createVariable('qsea', 'f4', ('time','lat','lon'))
        Rl = fid.createVariable('Rl', 'f4', ('time','lat','lon'))
        Rs = fid.createVariable('Rs', 'f4', ('time','lat','lon'))
        Rnl = fid.createVariable('Rnl', 'f4', ('time','lat','lon'))

        longitude[:] = lon
        latitude[:] = lat
        Date[:] = tim
        tau[:] = res[:, :, 0].reshape((len(tim), len(lat), len(lon)))*lsm
        sensible[:] = res[:, :, 1].reshape((len(tim), len(lat), len(lon)))*lsm
        latent[:] = res[:, :, 2].reshape((len(tim), len(lat), len(lon)))*lsm
        monob[:] = res[:, :, 3].reshape((len(tim), len(lat), len(lon)))*lsm
        cd[:] = res[:, :, 4].reshape((len(tim), len(lat), len(lon)))*lsm
        cdn[:] = res[:, :, 5].reshape((len(tim), len(lat), len(lon)))*lsm
        ct[:] = res[:, :, 6].reshape((len(tim), len(lat), len(lon)))*lsm
        ctn[:] = res[:, :, 7].reshape((len(tim), len(lat), len(lon)))*lsm
        cq[:] = res[:, :, 8].reshape((len(tim), len(lat), len(lon)))*lsm
        cqn[:] = res[:, :, 9].reshape((len(tim), len(lat), len(lon)))*lsm
        tsrv[:] = res[:, :, 10].reshape((len(tim), len(lat), len(lon)))*lsm
        tsr[:] = res[:, :, 11].reshape((len(tim), len(lat), len(lon)))*lsm
        qsr[:] = res[:, :, 12].reshape((len(tim), len(lat), len(lon)))*lsm
        usr[:] = res[:, :, 13].reshape((len(tim), len(lat), len(lon)))*lsm
        psim[:] = res[:, :, 14].reshape((len(tim), len(lat), len(lon)))*lsm
        psit[:] = res[:, :, 15].reshape((len(tim), len(lat), len(lon)))*lsm
        psiq[:] = res[:, :, 16].reshape((len(tim), len(lat), len(lon)))*lsm
        u10n[:] = res[:, :, 17].reshape((len(tim), len(lat), len(lon)))*lsm
        t10n[:] = res[:, :, 18].reshape((len(tim), len(lat), len(lon)))*lsm
        tv10n[:] = res[:, :, 19].reshape((len(tim), len(lat), len(lon)))*lsm
        q10n[:] = res[:, :, 20].reshape((len(tim), len(lat), len(lon)))*lsm
        zo[:] = res[:, :, 21].reshape((len(tim), len(lat), len(lon)))*lsm
        zot[:] = res[:, :, 22].reshape((len(tim), len(lat), len(lon)))*lsm
        zoq[:] = res[:, :, 23].reshape((len(tim), len(lat), len(lon)))*lsm
        urefs[:] = res[:, :, 24].reshape((len(tim), len(lat), len(lon)))*lsm
        trefs[:] = res[:, :, 25].reshape((len(tim), len(lat), len(lon)))*lsm
        qrefs[:] = res[:, :, 26].reshape((len(tim), len(lat), len(lon)))*lsm
        itera[:] = res[:, :, 27].reshape((len(tim), len(lat), len(lon)))*lsm
        dter[:] = res[:, :, 28].reshape((len(tim), len(lat), len(lon)))*lsm
        dtwl[:] = res[:, :, 29].reshape((len(tim), len(lat), len(lon)))*lsm
        dqer[:] = res[:, :, 30].reshape((len(tim), len(lat), len(lon)))*lsm
        qair[:] = res[:, :, 31].reshape((len(tim), len(lat), len(lon)))*lsm
        qsea[:] = res[:, :, 32].reshape((len(tim), len(lat), len(lon)))*lsm
        Rl = res[:, :, 33].reshape((len(tim), len(lat), len(lon)))
        Rs = res[:, :, 34].reshape((len(tim), len(lat), len(lon)))
        Rnl = res[:, :, 35].reshape((len(tim), len(lat), len(lon)))
        longitude.long_name = 'Longitude'
        longitude.units = 'degrees East'
        latitude.long_name = 'Latitude'
        latitude.units = 'degrees North'
        Date.long_name = "calendar date"
        Date.units = "YYYYMMDD UTC"
        tau.long_name = 'Wind stress'
        tau.units = 'N/m^2'
        sensible.long_name = 'Sensible heat fluxe'
        sensible.units = 'W/m^2'
        latent.long_name = 'Latent heat flux'
        latent.units = 'W/m^2'
        monob.long_name = 'Monin-Obukhov length'
        monob.units = 'm'
        cd.long_name = 'Drag coefficient'
        cd.units = ''
        cdn.long_name = 'Neutral Drag coefficient'
        cdn.units = ''
        ct.long_name = 'Heat exchange coefficient'
        ct.units = ''
        ctn.long_name = 'Neutral Heat exchange coefficient'
        ctn.units = ''
        cq.long_name = 'Moisture exchange coefficient'
        cq.units = ''
        cqn.long_name = 'Neutral Moisture exchange coefficient'
        cqn.units = ''
        tsrv.long_name = 'star virtual temperature'
        tsrv.units = 'degrees Celsius'
        tsr.long_name = 'star temperature'
        tsr.units = 'degrees Celsius'
        qsr.long_name = 'star specific humidity'
        qsr.units = 'gr/kgr'
        usr.long_name = 'friction velocity'
        usr.units = 'm/s'
        psim.long_name = 'Momentum stability function'
        psit.long_name = 'Heat stability function'
        u10n.long_name = '10m neutral wind speed'
        u10n.units = 'm/s'
        t10n.long_name = '10m neutral temperature'
        t10n.units = 'degrees Celsius'
        tv10n.long_name = '10m neutral virtual temperature'
        tv10n.units = 'degrees Celsius'
        q10n.long_name = '10m neutral specific humidity'
        q10n.units = 'gr/kgr'
        zo.long_name = 'momentum roughness length'
        zo.units = 'm'
        zot.long_name = 'temperature roughness length'
        zot.units = 'm'
        zoq.long_name = 'moisture roughness length'
        zoq.units = 'm'
        urefs.long_name = 'wind speed at ref height'
        urefs.units = 'm/s'
        trefs.long_name = 'temperature at ref height'
        trefs.units = 'degrees Celsius'
        qrefs.long_name = 'specific humidity at ref height'
        qrefs.units = 'gr/kgr'
        qair.long_name = 'specific humidity of air'
        qair.units = 'gr/kgr'
        qsea.long_name = 'specific humidity over water'
        qsea.units = 'gr/kgr'
        itera.long_name = 'number of iterations'
        fid.close()
        #%% delete variables
        del longitude, latitude, Date, tau, sensible, latent, monob, cd, cdn
        del ct, ctn, cq, cqn, tsrv, tsr, qsr, usr, psim, psit, psiq, u10n, t10n
        del tv10n, q10n, zo, zot, zoq, urefs, trefs, qrefs, itera, dter, dqer
        del qair, qsea, Rl, Rs, Rnl, dtwl
        del T, Td, tim, p, lw, sw, lsm, spd, hin, latIn
    return res, lon, lat
#%% run function
start_time = time.perf_counter()
#------------------------------------------------------------------------------
inF = input("Give input file name (data_all.nc or era5_r360x180.nc): \n")
meth = input("Give prefered method: \n")
while meth not in ["S80", "S88", "LP82", "YT96", "UA", "LY04", "C30", "C35",
                   "C40", "ecmwf","Beljaars"]:
    print("method unknown")
    meth = input("Give prefered method: \n")
else:
    meth = meth #[meth]
ext = meth+"_"
#------------------------------------------------------------------------------
gustIn = input("Give gustiness option (to use default press enter): \n")
if (gustIn == ''):
    gustIn = None
    ext = ext+'gust_'
else:
    gustIn = np.asarray(eval(gustIn), dtype=float)
    if ((np.all(gustIn) == 0)):
        ext = ext+'nogust_'
    else:
        ext = ext+'gust_'
#------------------------------------------------------------------------------
cskinIn = input("Give cool skin option (to use default press enter): \n")
if (cskinIn == ''):
    cskinIn = None
    if ((cskinIn == None) and (meth == "S80" or meth == "S88" or meth == "LP82"
                             or meth == "YT96" or meth == "UA" or
                             meth == "LY04")):
        cskinIn = 0
        ext = ext+'noskin_'
    elif ((cskinIn == None) and (meth == "C30" or meth == "C35" or meth == "C40"
                               or meth == "ecmwf" or meth == "Beljaars")):
        cskinIn = 1
        ext = ext+'skin_'
else:
    cskinIn = int(cskinIn)
    if (cskinIn == 0):
        ext = ext+'noskin_'
    elif (cskinIn == 1):
        ext = ext+'skin_'
#------------------------------------------------------------------------------
tolIn = input("Give tolerance option (to use default press enter): \n")
if (tolIn == ''):
    tolIn = ['flux', 0.01, 1, 1]
else:
    tolIn = eval(tolIn)
ext = ext+'tol'+tolIn[0]
#------------------------------------------------------------------------------
outF = input("Give path and output file name: \n")
if (outF == ''):
    outF = "out_"+inF[:-3]+"_"+ext+".nc"
elif (outF[-3:] != '.nc'):
    outF = outF+".nc"
else:
    outF = outF
#------------------------------------------------------------------------------
print("\n run_ASFC.py, started for method "+meth)

res, lon, lat = run_ASFC(inF, outF, gustIn, cskinIn, tolIn, meth)
print("run_ASFC.py took ", np.round((time.perf_counter()-start_time)/60, 2),
      "minutes to run")

#%% generate flux plots
if (inF == 'era5_r360x180.nc'):
    cm = plt.cm.get_cmap('RdYlBu')
    ttl = ["tau (Nm$^{-2}$)", "shf (Wm$^{-2}$)", "lhf (Wm$^{-2}$)"]
    for i in range(3):
        plt.figure()
        plt.contourf(lon, lat,
                     np.nanmean(res[:, :, i], axis=0).reshape(len(lat),
                                                              len(lon)),
                     100, cmap=cm)
        plt.colorbar()
        plt.tight_layout()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(meth+', '+ttl[i])
        plt.savefig('./'+ttl[i][:3]+'_'+ext+'.png', dpi=300, bbox_inches='tight')
elif (inF == "data_all.nc"):
    ttl = ["tau (Nm$^{-2}$)", "shf (Wm$^{-2}$)", "lhf (Wm$^{-2}$)"]
    for i in range(3):
        plt.figure()
        plt.plot(res[i],'.c', markersize=1)
        plt.title(meth)
        plt.xlabel("points")
        plt.ylabel(ttl[i])
        plt.savefig('./'+ttl[i][:3]+'_'+ext+'.png', dpi=300, bbox_inches='tight')

#%% generate txt file with statistic
if ((cskinIn == None) and (meth == "S80" or meth == "S88" or meth == "LP82"
                           or meth == "YT96" or meth == "UA" or
                           meth == "LY04")):
   cskinIn = 0
elif ((cskinIn == None) and (meth == "C30" or meth == "C35" or meth == "C40"
                             or meth == "ecmwf" or meth == "Beljaars")):
   cskinIn = 1
if (np.all(gustIn == None) and (meth == "C30" or meth == "C35" or meth == "C40")):
    gustIn = [1, 1.2, 600]
elif (np.all(gustIn == None) and (meth == "UA" or meth == "ecmwf")):
    gustIn = [1, 1, 1000]
elif np.all(gustIn == None):
    gustIn = [1, 1.2, 800]
elif ((np.size(gustIn) < 3) and (gustIn == 0)):
    gust = [0, 0, 0]
if (tolIn == None):
    tolIn = ['flux', 0.01, 1, 1]

print("Input summary", file=open('./stats.txt', 'a'))
print('input file name: {}, \n method: {}, \n gustiness: {}, \n cskin: {},'
      ' \n tolerance: {}'.format(inF, meth, gustIn, cskinIn, tolIn),
      file=open('./stats.txt', 'a'))
ttl = np.asarray(["tau  ", "shf  ", "lhf  ", "L    ", "cd   ", "cdn  ",
                  "ct   ", "ctn  ", "cq   ", "cqn  ", "tsrv ", "tsr  ",
                  "qsr  ", "usr  ", "psim ", "psit ", "psiq ", "u10n ",
                  "t10n ", "tv10n", "q10n ", "zo   ", "zot  ", "zoq  ",
                  "urefs", "trefs", "qrefs", "itera", "dter ", "dqer ",
                  "dtwl ", "qair ", "qsea ", "Rl   ", "Rs   ", "Rnl  "])
header = ["var", "mean", "median", "min", "max", "5%", "95%"]
n = np.shape(res)
stats = np.copy(ttl)
if (inF == 'era5_r360x180.nc'):
    stats = np.c_[stats, np.nanmean(res.reshape(n[0]*n[1], n[2]), axis=0)]
    stats = np.c_[stats, np.nanmedian(res.reshape(n[0]*n[1], n[2]), axis=0)]
    stats = np.c_[stats, np.nanmin(res.reshape(n[0]*n[1], n[2]), axis=0)]
    stats = np.c_[stats, np.nanmax(res.reshape(n[0]*n[1], n[2]), axis=0)]
    stats = np.c_[stats, np.nanpercentile(res.reshape(n[0]*n[1], n[2]), 5,
                                          axis=0)]
    stats = np.c_[stats, np.nanpercentile(res.reshape(n[0]*n[1], n[2]), 95,
                                          axis=0)]
    print(tabulate(stats, headers=header, tablefmt="github", numalign="left",
                   floatfmt=("s", "2.2e", "2.2e", "2.2e", "2.2e", "2.2e",
                               "2.2e")), file=open('./stats.txt', 'a'))
    print('-'*79+'\n', file=open('./stats.txt', 'a'))
elif (inF == "data_all.nc"):
    stats = np.c_[stats, np.nanmean(res, axis=1)]
    stats = np.c_[stats, np.nanmedian(res, axis=1)]
    stats = np.c_[stats, np.nanmin(res, axis=1)]
    stats = np.c_[stats, np.nanmax(res, axis=1)]
    stats = np.c_[stats, np.nanpercentile(res, 5, axis=1)]
    stats = np.c_[stats, np.nanpercentile(res, 95, axis=1)]
    print(tabulate(stats, headers=header, tablefmt="github", numalign="left",
                   floatfmt=("s", "2.2e", "2.2e", "2.2e", "2.2e", "2.2e",
                               "2.2e")), file=open('./stats.txt', 'a'))
    print('-'*79+'\n', file=open('./stats.txt', 'a'))

print('input file name: {}, \n method: {}, \n gustiness: {}, \n cskin: {},'
      ' \n tolerance: {}, \n output is written in: {}'.format(inF, meth,
                                                              gustIn, cskinIn,
                                                              tolIn, outF),
      file=open('./readme.txt', 'w'))
