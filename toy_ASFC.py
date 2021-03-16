"""
example of running AirSeaFluxCode with
1. R/V data (data_all.csv) or
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
import pandas as pd
from AirSeaFluxCode import AirSeaFluxCode
import time
from tabulate import tabulate
#%%
def reject_outliers(data, m=2):
    x = np.copy(data)
    x = np.where(np.abs(x-np.nanmean(x)) < m*np.nanstd(x), x, np.nan)
    return x


def toy_ASFC(inF, outF, gustIn, cskinIn, tolIn, meth):
    """
    Example routine of how to run AirSeaFluxCode with the test data given
    and save output either as .csv or NetCDF

    Parameters
    ----------
    inF : str
        input filename either data_all.csv or era5_r360x180.nc
    outF : str
        output filename
    gustIn : float
        gustiness option e.g. [1, 1.2, 800]
    cskinIn : int
        cool skin option input 0 or 1
    tolIn : float
        tolerance input option e.g. ['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1]
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
    if (inF == "data_all.csv"):
        #%% load data_all
        inDt = pd.read_csv("data_all.csv")
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
        del hu, ht, inDt
        #%% run AirSeaFluxCode
        res = AirSeaFluxCode(spd, t, sst, lat=lat, hum=['rh', rh], P=p,
                             hin=hin, Rs=sw, tol=tolIn, gust=gustIn,
                             cskin=cskinIn, meth=meth, L="ecmwf", n=30)

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
        icon = np.array(fid.variables["siconc"])
        fid.close()
        spd = np.sqrt(np.power(u, 2)+np.power(v, 2))
        del u, v, fid
        lsm = np.where(lsm > 0, np.nan, 1) # reverse 0 on land 1 over ocean
        icon = np.where(icon < 0, np.nan, 1)
        msk = lsm*icon
        hin = np.array([10, 2, 2])
        latIn = np.tile(lat, (len(lon), 1)).T.reshape(len(lon)*len(lat))
        date = np.copy(tim)

        #%% run AirSeaFluxCode
        res = np.zeros((len(tim),len(lon)*len(lat), 39))
        flg = np.empty((len(tim),len(lon)*len(lat)), dtype="object")
        # reshape input and run code
        for x in range(len(tim)):
            temp = AirSeaFluxCode(spd.reshape(len(tim), len(lon)*len(lat))[x, :],
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
            a = temp.loc[:,"tau":"rh"]
            a = a.to_numpy()
            flg[x, :] = temp["flag"]
            res[x, :, :] = a
            del a, temp
            n = np.shape(res)
            res = np.asarray([res[:, :, i]*msk.reshape(n[0], n[1])
                              for i in range(39)])
            res = np.moveaxis(res, 0, -1)
        flg = np.where(np.isnan(msk.reshape(len(tim), len(lon)*len(lat))),
                        'm', flg)
    if (outF[-3:] == '.nc'):
        if (inF == 'era5_r360x180.nc'):
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
            ug = fid.createVariable('ug', 'f4', ('time','lat','lon'))
            Rib = fid.createVariable('Rib', 'f4', ('time','lat','lon'))
            rh = fid.createVariable('rh', 'f4', ('time','lat','lon'))
            flag = fid.createVariable('flag', 'U1', ('time','lat','lon'))

            longitude[:] = lon
            latitude[:] = lat
            Date[:] = tim
            tau[:] = res[:, :, 0].reshape((len(tim), len(lat), len(lon)))*msk
            sensible[:] = res[:, :, 1].reshape((len(tim), len(lat), len(lon)))*msk
            latent[:] = res[:, :, 2].reshape((len(tim), len(lat), len(lon)))*msk
            monob[:] = res[:, :, 3].reshape((len(tim), len(lat), len(lon)))*msk
            cd[:] = res[:, :, 4].reshape((len(tim), len(lat), len(lon)))*msk
            cdn[:] = res[:, :, 5].reshape((len(tim), len(lat), len(lon)))*msk
            ct[:] = res[:, :, 6].reshape((len(tim), len(lat), len(lon)))*msk
            ctn[:] = res[:, :, 7].reshape((len(tim), len(lat), len(lon)))*msk
            cq[:] = res[:, :, 8].reshape((len(tim), len(lat), len(lon)))*msk
            cqn[:] = res[:, :, 9].reshape((len(tim), len(lat), len(lon)))*msk
            tsrv[:] = res[:, :, 10].reshape((len(tim), len(lat), len(lon)))*msk
            tsr[:] = res[:, :, 11].reshape((len(tim), len(lat), len(lon)))*msk
            qsr[:] = res[:, :, 12].reshape((len(tim), len(lat), len(lon)))*msk
            usr[:] = res[:, :, 13].reshape((len(tim), len(lat), len(lon)))*msk
            psim[:] = res[:, :, 14].reshape((len(tim), len(lat), len(lon)))*msk
            psit[:] = res[:, :, 15].reshape((len(tim), len(lat), len(lon)))*msk
            psiq[:] = res[:, :, 16].reshape((len(tim), len(lat), len(lon)))*msk
            u10n[:] = res[:, :, 17].reshape((len(tim), len(lat), len(lon)))*msk
            t10n[:] = res[:, :, 18].reshape((len(tim), len(lat), len(lon)))*msk
            tv10n[:] = res[:, :, 19].reshape((len(tim), len(lat), len(lon)))*msk
            q10n[:] = res[:, :, 20].reshape((len(tim), len(lat), len(lon)))*msk
            zo[:] = res[:, :, 21].reshape((len(tim), len(lat), len(lon)))*msk
            zot[:] = res[:, :, 22].reshape((len(tim), len(lat), len(lon)))*msk
            zoq[:] = res[:, :, 23].reshape((len(tim), len(lat), len(lon)))*msk
            urefs[:] = res[:, :, 24].reshape((len(tim), len(lat), len(lon)))*msk
            trefs[:] = res[:, :, 25].reshape((len(tim), len(lat), len(lon)))*msk
            qrefs[:] = res[:, :, 26].reshape((len(tim), len(lat), len(lon)))*msk
            itera[:] = res[:, :, 27].reshape((len(tim), len(lat), len(lon)))*msk
            dter[:] = res[:, :, 28].reshape((len(tim), len(lat), len(lon)))*msk
            dqer[:] = res[:, :, 29].reshape((len(tim), len(lat), len(lon)))*msk
            dtwl[:] = res[:, :, 30].reshape((len(tim), len(lat), len(lon)))*msk
            qair[:] = res[:, :, 31].reshape((len(tim), len(lat), len(lon)))*msk
            qsea[:] = res[:, :, 32].reshape((len(tim), len(lat), len(lon)))*msk
            Rl[:] = res[:, :, 33].reshape((len(tim), len(lat), len(lon)))*msk
            Rs[:] = res[:, :, 34].reshape((len(tim), len(lat), len(lon)))*msk
            Rnl[:] = res[:, :, 35].reshape((len(tim), len(lat), len(lon)))*msk
            ug[:] = res[:, :, 36].reshape((len(tim), len(lat), len(lon)))*msk
            Rib[:] = res[:, :, 37].reshape((len(tim), len(lat), len(lon)))*msk
            rh[:] = res[:, :, 38].reshape((len(tim), len(lat), len(lon)))*msk
            flag[:] = flg.reshape((len(tim), len(lat), len(lon)))

            longitude.long_name = 'Longitude'
            longitude.units = 'degrees East'
            latitude.long_name = 'Latitude'
            latitude.units = 'degrees North'
            Date.long_name = "gregorian"
            Date.units = "hours since 1900-01-01 00:00:00.0"
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
            q10n.units = 'kgr/kgr'
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
            qrefs.units = 'kgr/kgr'
            qair.long_name = 'specific humidity of air'
            qair.units = 'kgr/kgr'
            qsea.long_name = 'specific humidity over water'
            qsea.units = 'kgr/kgr'
            itera.long_name = 'number of iterations'
            Rl.long_name = 'downward longwave radiation'
            Rl.units = 'W/m^2'
            Rs.long_name = 'downward shortwave radiation'
            Rs.units = 'W/m^2'
            Rnl.long_name = 'downward net longwave radiation'
            Rnl.units = 'W/m^2'
            ug.long_name = 'gust wind speed'
            ug.units = 'm/s'
            Rib.long_name = 'bulk Richardson number'
            rh.long_name = 'relative humidity'
            rh.units = '%'
            flag.long_name = ('flag "n" normal, "u": u10n < 0, "q": q10n < 0,'
                              '"l": zol<0.01, "m": missing, "i": points that'
                              'have not converged')
            fid.close()
            #%% delete variables
            del longitude, latitude, Date, tau, sensible, latent, monob, cd, cdn
            del ct, ctn, cq, cqn, tsrv, tsr, qsr, usr, psim, psit, psiq, u10n, t10n
            del tv10n, q10n, zo, zot, zoq, urefs, trefs, qrefs, itera, dter, dqer
            del qair, qsea, Rl, Rs, Rnl, dtwl
            del tim, T, Td, p, lw, sw, lsm, spd, hin, latIn, icon, msk
        else:
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
            dtwl = fid.createVariable('dtwl', 'f4', 'time')
            qair = fid.createVariable('qair', 'f4', 'time')
            qsea = fid.createVariable('qsea', 'f4', 'time')
            Rl = fid.createVariable('Rl', 'f4', 'time')
            Rs = fid.createVariable('Rs', 'f4', 'time')
            Rnl = fid.createVariable('Rnl', 'f4', 'time')
            ug = fid.createVariable('ug', 'f4', 'time')
            Rib = fid.createVariable('Rib', 'f4', 'time')
            rh = fid.createVariable('rh', 'f4', 'time')
            flag = fid.createVariable('flag', 'U1', 'time')

            longitude[:] = lon
            latitude[:] = lat
            Date[:] = date
            tau[:] = res["tau"]
            sensible[:] = res["shf"]
            latent[:] = res["lhf"]
            monob[:] = res["L"]
            cd[:] = res["cd"]
            cdn[:] = res["cdn"]
            ct[:] = res["ct"]
            ctn[:] = res["ctn"]
            cq[:] = res["cq"]
            cqn[:] = res["cqn"]
            tsrv[:] = res["tsrv"]
            tsr[:] = res["tsr"]
            qsr[:] = res["qsr"]
            usr[:] = res["usr"]
            psim[:] = res["psim"]
            psit[:] = res["psit"]
            psiq[:] = res["psiq"]
            u10n[:] = res["u10n"]
            t10n[:] = res["t10n"]
            tv10n[:] = res["tv10n"]
            q10n[:] = res["q10n"]
            zo[:] = res["zo"]
            zot[:] = res["zot"]
            zoq[:] = res["zoq"]
            urefs[:] = res["uref"]
            trefs[:] = res["tref"]
            qrefs[:] = res["qref"]
            itera[:] = res["iteration"]
            dter[:] = res["dter"]
            dqer[:] = res["dqer"]
            dtwl[:] = res["dtwl"]
            qair[:] = res["qair"]
            qsea[:] = res["qsea"]
            Rl[:] = res["Rl"]
            Rs[:] = res["Rs"]
            Rnl[:] = res["Rnl"]
            ug[:] = res["ug"]
            Rib[:] = res["Rib"]
            rh[:] = res["rh"]
            flag[:] = res["flag"]

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
            q10n.units = 'kgr/kgr'
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
            qrefs.units = 'kgr/kgr'
            qair.long_name = 'specific humidity of air'
            qair.units = 'kgr/kgr'
            qsea.long_name = 'specific humidity over water'
            qsea.units = 'kgr/kgr'
            itera.long_name = 'number of iterations'
            Rl.long_name = 'downward longwave radiation'
            Rl.units = 'W/m^2'
            Rs.long_name = 'downward shortwave radiation'
            Rs.units = 'W/m^2'
            Rnl.long_name = 'downward net longwave radiation'
            Rnl.units = 'W/m^2'
            ug.long_name = 'gust wind speed'
            ug.units = 'm/s'
            Rib.long_name = 'bulk Richardson number'
            rh.long_name = 'relative humidity'
            rh.units = '%'
            flag.long_name = ('flag "n" normal, "u": u10n < 0, "q": q10n < 0,'
                              '"l": zol<0.01, "m": missing, "i": points that'
                              'have not converged')
            fid.close()
            #%% delete variables
            del longitude, latitude, Date, tau, sensible, latent, monob, cd, cdn
            del ct, ctn, cq, cqn, tsrv, tsr, qsr, usr, psim, psit, psiq, u10n, t10n
            del tv10n, q10n, zo, zot, zoq, urefs, trefs, qrefs, itera, dter, dqer
            del qair, qsea, Rl, Rs, Rnl, ug, rh, Rib
            del t, date, p, sw, spd, hin, sst
    else:
        #%% save as .csv
        res.insert(loc=0, column='date', value=date)
        res.insert(loc=1, column='lon', value=lon)
        res.insert(loc=2, column='lat', value=lat)
        res.to_csv(outF)
    return res, lon, lat
#%% run function
start_time = time.perf_counter()
#------------------------------------------------------------------------------
inF = input("Give input file name (data_all.csv or era5_r360x180.nc): \n")
meth = input("Give prefered method: \n")
while meth not in ["S80", "S88", "LP82", "YT96", "UA", "LY04", "C30", "C35",
                   "ecmwf","Beljaars"]:
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
    elif ((cskinIn == None) and (meth == "C30" or meth == "C35"
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
    tolIn = ['flux', 1e-3, 0.1, 0.1]
else:
    tolIn = eval(tolIn)
ext = ext+'tol'+tolIn[0]
#------------------------------------------------------------------------------
outF = input("Give path and output file name: \n")
if ((outF == '') and (inF == "data_all.csv")):
    outF = "out_"+inF[:-4]+"_"+ext+".csv"
elif ((outF == '') and (inF == "era5_r360x180.nc")):
    outF = "out_"+inF[:-3]+"_"+ext+".nc"
elif ((outF[-4:] == '.csv') and (inF == 'era5_r360x180.nc')):
    outF = outF[:-4]+".nc"
elif ((outF[-3:] != '.nc') and (outF[-4:] != '.csv')):
    if (inF == "data_all.csv"):
        outF = outF+".csv"
    else:
        outF = outF+".nc"
else:
    outF = outF
#------------------------------------------------------------------------------
print("\n run_ASFC.py, started for method "+meth)

res, lon, lat = toy_ASFC(inF, outF, gustIn, cskinIn, tolIn, meth)
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
elif (inF == "data_all.csv"):
    ttl = ["tau (Nm$^{-2}$)", "shf (Wm$^{-2}$)", "lhf (Wm$^{-2}$)"]
    for i in range(3):
        plt.figure()
        plt.plot(res[ttl[i][:3]],'.c', markersize=1)
        plt.title(meth)
        plt.xlabel("points")
        plt.ylabel(ttl[i])
        plt.savefig('./'+ttl[i][:3]+'_'+ext+'.png', dpi=300, bbox_inches='tight')

#%% generate txt file with statistic
if ((cskinIn == None) and (meth == "S80" or meth == "S88" or meth == "LP82"
                           or meth == "YT96" or meth == "UA" or
                           meth == "LY04")):
   cskinIn = 0
elif ((cskinIn == None) and (meth == "C30" or meth == "C35"
                             or meth == "ecmwf" or meth == "Beljaars")):
   cskinIn = 1
if (np.all(gustIn == None) and (meth == "C30" or meth == "C35")):
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
                  "dtwl ", "qair ", "qsea ", "Rl   ", "Rs   ", "Rnl  ",
                  "ug   ", "Rib  ", "rh   "])
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
elif (inF == "data_all.csv"):
    a = res.loc[:,"tau":"rh"].to_numpy(dtype="float64").T
    stats = np.c_[stats, np.nanmean(a, axis=1)]
    stats = np.c_[stats, np.nanmedian(a, axis=1)]
    stats = np.c_[stats, np.nanmin(a, axis=1)]
    stats = np.c_[stats, np.nanmax(a, axis=1)]
    stats = np.c_[stats, np.nanpercentile(a, 5, axis=1)]
    stats = np.c_[stats, np.nanpercentile(a, 95, axis=1)]
    print(tabulate(stats, headers=header, tablefmt="github", numalign="left",
                   floatfmt=("s", "2.2e", "2.2e", "2.2e", "2.2e", "2.2e",
                               "2.2e")), file=open('./stats.txt', 'a'))
    print('-'*79+'\n', file=open('./stats.txt', 'a'))
    del a

print('input file name: {}, \n method: {}, \n gustiness: {}, \n cskin: {},'
      ' \n tolerance: {}, \n output is written in: {}'.format(inF, meth,
                                                              gustIn, cskinIn,
                                                              tolIn, outF),
      file=open('./readme.txt', 'w'))
