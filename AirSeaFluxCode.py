import numpy as np
import pandas as pd
import logging
from get_init import get_init
from hum_subs import (get_hum, gamma_moist)
from util_subs import (kappa, CtoK, get_heights)
from flux_subs import (cs_C35, cs_Beljaars, cs_ecmwf, wl_ecmwf,
                       get_gust, get_L, get_strs, psim_calc,
                       psit_calc, cdn_calc, cd_calc, ctcq_calc, ctcqn_calc)



def AirSeaFluxCode(spd, T, SST, lat=None, hum=None, P=None, hin=18, hout=10,
                   Rl=None, Rs=None, cskin=None, skin="C35", wl=0, gust=None,
                   meth="S80", qmeth="Buck2", tol=None, n=10, out=0, L=None):
    """
    Calculates turbulent surface fluxes using different parameterizations
    Calculates height adjusted values for spd, T, q

    Parameters
    ----------
        spd : float
            relative wind speed in m/s (is assumed as magnitude difference
            between wind and surface current vectors)
        T : float
            air temperature in K (will convert if < 200)
        SST : float
            sea surface temperature in K (will convert if < 200)
        lat : float
            latitude (deg), default 45deg
        hum : float
            humidity input switch 2x1 [x, values] default is relative humidity
            x='rh' : relative humidity in %
            x='q' : specific humidity (g/kg)
            x='Td' : dew point temperature (K)
        P : float
            air pressure (hPa), default 1013hPa
        hin : float
            sensor heights in m (array 3x1 or 3xn), default 18m
        hout : float
            output height, default is 10m
        Rl : float
            downward longwave radiation (W/m^2)
        Rs : float
            downward shortwave radiation (W/m^2)
        cskin : int
            0 switch cool skin adjustment off, else 1
            default is 1
        skin : str
            cool skin method option "C35", "ecmwf" or "Beljaars"
        wl : int
            warm layer correction default is 0, to switch on set to 1
        gust : int
            3x1 [x, beta, zi] x=1 to include the effect of gustiness, else 0
            beta gustiness parameter, beta=1 for UA, beta=1.2 for COARE
            zi PBL height (m) 600 for COARE, 1000 for UA and ecmwf, 800 default
            default for COARE [1, 1.2, 600]
            default for UA, ecmwf [1, 1, 1000]
            default else [1, 1.2, 800]
        meth : str
            "S80", "S88", "LP82", "YT96", "UA", "LY04", "C30", "C35",
            "ecmwf", "Beljaars"
        qmeth : str
            is the saturation evaporation method to use amongst
            "HylandWexler","Hardy","Preining","Wexler","GoffGratch","WMO",
            "MagnusTetens","Buck","Buck2","WMO2018","Sonntag","Bolton",
            "IAPWS","MurphyKoop"]
            default is Buck2
        tol : float
           4x1 or 7x1 [option, lim1-3 or lim1-6]
           option : 'flux' to set tolerance limits for fluxes only lim1-3
           option : 'ref' to set tolerance limits for height adjustment lim-1-3
           option : 'all' to set tolerance limits for both fluxes and height
                    adjustment lim1-6 ['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1]
           default is tol=['flux', 1e-3, 0.1, 0.1]
        n : int
            number of iterations (defautl = 10)
        out : int
            set 0 to set points that have not converged to missing (default)
            set 1 to keep points
        L : str
           Monin-Obukhov length definition options
           "S80"  : default for S80, S88, LP82, YT96 and LY04
           "ecmwf" : following ecmwf (IFS Documentation cy46r1), default for
           ecmwf
    Returns
    -------
        res : array that contains
                       1. momentum flux       (N/m^2)
                       2. sensible heat       (W/m^2)
                       3. latent heat         (W/m^2)
                       4. Monin-Obhukov length (m)
                       5. drag coefficient (cd)
                       6. neutral drag coefficient (cdn)
                       7. heat exchange coefficient (ct)
                       8. neutral heat exchange coefficient (ctn)
                       9. moisture exhange coefficient (cq)
                       10. neutral moisture exchange coefficient (cqn)
                       11. star virtual temperatcure (tsrv)
                       12. star temperature (tsr)
                       13. star specific humidity (qsr)
                       14. star wind speed (usr)
                       15. momentum stability function (psim)
                       16. heat stability function (psit)
                       17. moisture stability function (psiq)
                       18. 10m neutral wind speed (u10n)
                       19. 10m neutral temperature (t10n)
                       20. 10m neutral virtual temperature (tv10n)
                       21. 10m neutral specific humidity (q10n)
                       22. surface roughness length (zo)
                       23. heat roughness length (zot)
                       24. moisture roughness length (zoq)
                       25. wind speed at reference height (uref)
                       26. temperature at reference height (tref)
                       27. specific humidity at reference height (qref)
                       28. number of iterations until convergence
                       29. cool-skin temperature depression (dter)
                       30. cool-skin humidity depression (dqer)
                       31. warm layer correction (dtwl)
                       32. specific humidity of air (qair)
                       33. specific humidity at sea surface (qsea)
                       34. downward longwave radiation (Rl)
                       35. downward shortwave radiation (Rs)
                       36. downward net longwave radiation (Rnl)
                       37. gust wind speed (ug)
                       38. Bulk Richardson number (Rib)
                       39. relative humidity (rh)
                       40. flag ("n": normal, "o": out of nominal range,
                                 "u": u10n<0, "q":q10n<0
                                 "m": missing, "l": Rib<-0.5 or Rib>0.2,
                                 "i": convergence fail at n)

    2021 / Author S. Biri
    """
    logging.basicConfig(filename='flux_calc.log',
                        format='%(asctime)s %(message)s',level=logging.INFO)
    logging.captureWarnings(True)
    #  check input values and set defaults where appropriate
    lat, hum, P, Rl, Rs, cskin, skin, wl, gust, tol, L = get_init(spd, T, SST,
                                                                  lat, hum, P,
                                                                  Rl, Rs,
                                                                  cskin, skin,
                                                                  wl, gust, L,
                                                                  tol, meth,
                                                                  qmeth)
    flag = np.ones(spd.shape, dtype="object")*"n"
    flag = np.where(np.isnan(spd+T+SST+lat+hum[1]+P+Rs), "m", flag)
    ref_ht = 10        # reference height
    h_in = get_heights(hin, len(spd))  # heights of input measurements/fields
    h_out = get_heights(hout, 1)       # desired height of output variables
    logging.info('method %s, inputs: lat: %s | P: %s | Rl: %s |'
                 ' Rs: %s | gust: %s | cskin: %s | L : %s', meth,
                 np.nanmedian(lat), np.nanmedian(P), np.nanmedian(Rl),
                 np.nanmedian(Rs), gust, cskin, L)
    #  set up/calculate temperatures and specific humidities
    th = np.where(T < 200, (np.copy(T)+CtoK) *
                  np.power(1000/P,287.1/1004.67),
                  np.copy(T)*np.power(1000/P,287.1/1004.67))  # potential T
    sst = np.where(SST < 200, np.copy(SST)+CtoK, np.copy(SST))
    qair, qsea = get_hum(hum, T, sst, P, qmeth)
    Rb = np.empty(sst.shape)
    #lapse rate
    tlapse = gamma_moist(SST, T, qair/1000)
    Ta = np.where(T < 200, np.copy(T)+CtoK+tlapse*h_in[1],
                  np.copy(T)+tlapse*h_in[1])  # convert to Kelvin if needed
    logging.info('method %s and q method %s | qsea:%s, qair:%s', meth, qmeth,
                  np.nanmedian(qsea), np.nanmedian(qair))
    if (np.all(np.isnan(qsea)) or np.all(np.isnan(qair))):
        print("qsea and qair cannot be nan")

    dt = Ta - sst
    dq = qair - qsea

    #  first guesses
    t10n, q10n = np.copy(Ta), np.copy(qair)
    tv10n = t10n*(1+0.6077*q10n)
    #  Zeng et al. 1998
    tv=th*(1+0.6077*qair)   # virtual potential T
    dtv=dt*(1+0.6077*qair)+0.6077*th*dq
    # ------------
    rho = P*100/(287.1*tv10n)
    lv = (2.501-0.00237*(sst-CtoK))*1e6
    cp = 1004.67*(1 + 0.00084*qsea)
    u10n = np.copy(spd)
    cd10n = cdn_calc(u10n, Ta, None, lat, meth)
    ct10n, ct, cq10n, cq = (np.zeros(spd.shape)*np.nan, np.zeros(spd.shape)*np.nan,
                        np.zeros(spd.shape)*np.nan, np.zeros(spd.shape)*np.nan)
    psim, psit, psiq = (np.zeros(spd.shape), np.zeros(spd.shape),
                        np.zeros(spd.shape))
    cd = cd_calc(cd10n, h_in[0], ref_ht, psim)
    tsr, tsrv = np.zeros(spd.shape), np.zeros(spd.shape)
    qsr = np.zeros(spd.shape)
    # cskin parameters
    tkt = 0.001*np.ones(T.shape)
    dter = np.ones(T.shape)*0.3
    dqer = dter*0.622*lv*qsea/(287.1*np.power(sst, 2))
    Rnl = 0.97*(5.67e-8*np.power(sst-0.3*cskin, 4)-Rl)
    Qs = 0.945*Rs
    dtwl = np.ones(T.shape)*0.3
    skt = np.copy(sst)
    # gustiness adjustment
    if (gust[0] == 1 and meth == "UA"):
        wind = np.where(dtv >= 0, np.where(spd > 0.1, spd, 0.1),
                        np.sqrt(np.power(np.copy(spd), 2)+np.power(0.5, 2)))
    elif (gust[0] == 1):
        wind = np.sqrt(np.power(np.copy(spd), 2)+np.power(0.5, 2))
    elif (gust[0] == 0):
        wind = np.copy(spd)
    # stars and roughness lengths
    usr = np.sqrt(cd*np.power(wind, 2))
    zo = 0.0001*np.ones(spd.shape)
    zot, zoq = 0.0001*np.ones(spd.shape), 0.0001*np.ones(spd.shape)
    monob = -100*np.ones(spd.shape)  # Monin-Obukhov length
    tsr = (dt+dter*cskin-dtwl*wl)*kappa/(np.log(h_in[1]/zot) -
                                         psit_calc(h_in[1]/monob, meth))
    qsr = (dq+dqer*cskin)*kappa/(np.log(h_in[2]/zoq) -
                                 psit_calc(h_in[2]/monob, meth))
    # set-up to feed into iteration loop
    it, ind = 0, np.where(spd > 0)
    ii, itera = True, np.zeros(spd.shape)*np.nan
    tau = 0.05*np.ones(spd.shape)
    sensible = 5*np.ones(spd.shape)
    latent = 65*np.ones(spd.shape)
    #  iteration loop
    while np.any(ii):
        it += 1
        if it > n:
            break
        if (tol[0] == 'flux'):
            old = np.array([np.copy(tau), np.copy(sensible), np.copy(latent)])
        elif (tol[0] == 'ref'):
            old = np.array([np.copy(u10n), np.copy(t10n), np.copy(q10n)])
        elif (tol[0] == 'all'):
            old = np.array([np.copy(u10n), np.copy(t10n), np.copy(q10n),
                            np.copy(tau), np.copy(sensible), np.copy(latent)])
        cd10n[ind] = cdn_calc(u10n[ind], Ta[ind], None, lat[ind], meth)
        if (np.all(np.isnan(cd10n))):
            break
            logging.info('break %s at iteration %s cd10n<0', meth, it)
        zo[ind] = ref_ht/np.exp(kappa/np.sqrt(cd10n[ind]))
        psim[ind] = psim_calc(h_in[0, ind]/monob[ind], meth)
        cd[ind] = cd_calc(cd10n[ind], h_in[0, ind], ref_ht, psim[ind])
        ct10n[ind], cq10n[ind] = ctcqn_calc(h_in[1, ind]/monob[ind],
                                            cd10n[ind], u10n[ind], zo[ind],
                                            Ta[ind], meth)
        zot[ind] = ref_ht/(np.exp(np.power(kappa, 2) /
                           (ct10n[ind]*np.log(ref_ht/zo[ind]))))
        zoq[ind] = ref_ht/(np.exp(np.power(kappa, 2) /
                           (cq10n[ind]*np.log(ref_ht/zo[ind]))))
        psit[ind] = psit_calc(h_in[1, ind]/monob[ind], meth)
        psiq[ind] = psit_calc(h_in[2, ind]/monob[ind], meth)
        ct[ind], cq[ind] = ctcq_calc(cd10n[ind], cd[ind], ct10n[ind], cq10n[ind],
                                      h_in[1, ind], h_in[2, ind], ref_ht,
                                      psit[ind], psiq[ind])
        usr[ind], tsr[ind], qsr[ind] = get_strs(h_in[:, ind], monob[ind],
                                                wind[ind], zo[ind], zot[ind],
                                                zoq[ind], dt[ind], dq[ind],
                                                dter[ind], dqer[ind], dtwl[ind],
                                                ct[ind], cq[ind], cskin, wl,
                                                meth)
        if ((cskin == 1) and (wl == 0)):
            if (skin == "C35"):
                dter[ind], dqer[ind], tkt[ind] = cs_C35(sst[ind], qsea[ind],
                                                        rho[ind], Rs[ind],
                                                        Rnl[ind],
                                                        cp[ind], lv[ind],
                                                        np.copy(tkt[ind]),
                                                        usr[ind], tsr[ind],
                                                        qsr[ind], lat[ind])
            elif (skin == "ecmwf"):
                dter[ind] = cs_ecmwf(rho[ind], Rs[ind], Rnl[ind], cp[ind],
                                     lv[ind], usr[ind], tsr[ind], qsr[ind],
                                     sst[ind], lat[ind])
                dqer[ind] = (dter[ind]*0.622*lv[ind]*qsea[ind] /
                             (287.1*np.power(sst[ind], 2)))
            elif (skin == "Beljaars"):
                Qs[ind], dter[ind] = cs_Beljaars(rho[ind], Rs[ind], Rnl[ind],
                                                 cp[ind], lv[ind], usr[ind],
                                                 tsr[ind], qsr[ind], lat[ind],
                                                 np.copy(Qs[ind]))
                dqer = dter*0.622*lv*qsea/(287.1*np.power(sst, 2))
        elif ((cskin == 1) and (wl == 1)):
            if (skin == "C35"):
                dter[ind], dqer[ind], tkt[ind] = cs_C35(sst[ind], qsea[ind],
                                                        rho[ind], Rs[ind],
                                                        Rnl[ind],
                                                        cp[ind], lv[ind],
                                                        np.copy(tkt[ind]),
                                                        usr[ind], tsr[ind],
                                                        qsr[ind], lat[ind])
                dtwl[ind] = wl_ecmwf(rho[ind], Rs[ind], Rnl[ind], cp[ind],
                                     lv[ind], usr[ind], tsr[ind], qsr[ind],
                                     np.copy(sst[ind]), np.copy(skt[ind]),
                                     np.copy(dter[ind]), lat[ind])
                skt = np.copy(sst)-dter+dtwl
            elif (skin == "ecmwf"):
                dter[ind] = cs_ecmwf(rho[ind], Rs[ind], Rnl[ind], cp[ind],
                                     lv[ind], usr[ind], tsr[ind], qsr[ind],
                                     sst[ind], lat[ind])
                dtwl[ind] = wl_ecmwf(rho[ind], Rs[ind], Rnl[ind], cp[ind],
                                     lv[ind], usr[ind], tsr[ind], qsr[ind],
                                     np.copy(sst[ind]), np.copy(skt[ind]),
                                     np.copy(dter[ind]), lat[ind])
                skt = np.copy(sst)-dter+dtwl
                dqer[ind] = (dter[ind]*0.622*lv[ind]*qsea[ind] /
                             (287.1*np.power(skt[ind], 2)))
            elif (skin == "Beljaars"):
                Qs[ind], dter[ind] = cs_Beljaars(rho[ind], Rs[ind], Rnl[ind],
                                                 cp[ind], lv[ind], usr[ind],
                                                 tsr[ind], qsr[ind], lat[ind],
                                                 np.copy(Qs[ind]))
                dtwl[ind] = wl_ecmwf(rho[ind], Rs[ind], Rnl[ind], cp[ind],
                                     lv[ind], usr[ind], tsr[ind], qsr[ind],
                                     np.copy(sst[ind]), np.copy(skt[ind]),
                                     np.copy(dter[ind]), lat[ind])
                skt = np.copy(sst)-dter+dtwl
                dqer = dter*0.622*lv*qsea/(287.1*np.power(skt, 2))
        else:
           dter[ind] = np.zeros(sst[ind].shape)
           dqer[ind] = np.zeros(sst[ind].shape)
           tkt[ind] = 0.001*np.ones(T[ind].shape)
        logging.info('method %s | dter = %s | dqer = %s | tkt = %s | Rnl = %s '
                     '| usr = %s | tsr = %s | qsr = %s', meth,
                     np.nanmedian(dter), np.nanmedian(dqer),
                     np.nanmedian(tkt), np.nanmedian(Rnl),
                     np.nanmedian(usr), np.nanmedian(tsr),
                     np.nanmedian(qsr))
        Rnl[ind] = 0.97*(5.67e-8*np.power(sst[ind] -
                          dter[ind]*cskin, 4)-Rl[ind])
        t10n[ind] = (Ta[ind] -
                     tsr[ind]/kappa*(np.log(h_in[1, ind]/ref_ht)-psit[ind]))
        q10n[ind] = (qair[ind] -
                     qsr[ind]/kappa*(np.log(h_in[2, ind]/ref_ht)-psiq[ind]))
        tv10n[ind] = t10n[ind]*(1+0.6077*q10n[ind])
        tsrv[ind], monob[ind], Rb[ind] = get_L(L, lat[ind], usr[ind], tsr[ind],
                                               qsr[ind], h_in[:, ind], Ta[ind],
                                               sst[ind]-dter[ind]*cskin+dtwl[ind]*wl,
                                               qair[ind], qsea[ind], wind[ind],
                                               np.copy(monob[ind]), psim[ind],
                                               meth)
        # sst[ind]-dter[ind]*cskin+dtwl[ind]*wl
        psim[ind] = psim_calc(h_in[0, ind]/monob[ind], meth)
        psit[ind] = psit_calc(h_in[1, ind]/monob[ind], meth)
        psiq[ind] = psit_calc(h_in[2, ind]/monob[ind], meth)
        if (gust[0] == 1 and meth == "UA"):
            wind[ind] = np.where(dtv[ind] >= 0, np.where(spd[ind] > 0.1,
                                  spd[ind], 0.1),
                                  np.sqrt(np.power(np.copy(spd[ind]), 2) +
                                  np.power(get_gust(gust[1], tv[ind], usr[ind],
                                  tsrv[ind], gust[2], lat[ind]), 2)))
                                  # Zeng et al. 1998 (20)
        elif (gust[0] == 1 and (meth == "C30" or meth == "C35")): # or meth == "C40"
            wind[ind] = np.sqrt(np.power(np.copy(spd[ind]), 2) +
                                np.power(get_gust(gust[1], Ta[ind], usr[ind],
                                tsrv[ind], gust[2], lat[ind]), 2))
        elif (gust[0] == 1):
            wind[ind] = np.sqrt(np.power(np.copy(spd[ind]), 2) +
                                np.power(get_gust(gust[1], Ta[ind], usr[ind],
                                tsrv[ind], gust[2], lat[ind]), 2))
        elif (gust[0] == 0):
            wind[ind] = np.copy(spd[ind])
        u10n[ind] = wind[ind]-usr[ind]/kappa*(np.log(h_in[0, ind]/10) -
                                              psim[ind])
        flag = np.where((u10n < 0) & (flag == "n"), "u",
                        np.where((u10n < 0) & (flag != "u"),
                                 flag+[","]+["u"], flag))
        u10n = np.where(u10n < 0, np.nan, u10n)
        itera[ind] = np.ones(1)*it
        sensible = -rho*cp*usr*tsr
        latent = -rho*lv*usr*qsr
        if (gust[0] == 1):
            tau = rho*np.power(usr, 2)*(spd/wind)
        elif (gust[0] == 0):
            tau = rho*np.power(usr, 2)
        if (tol[0] == 'flux'):
            new = np.array([np.copy(tau), np.copy(sensible), np.copy(latent)])
        elif (tol[0] == 'ref'):
            new = np.array([np.copy(u10n), np.copy(t10n), np.copy(q10n)])
        elif (tol[0] == 'all'):
            new = np.array([np.copy(u10n), np.copy(t10n), np.copy(q10n),
                            np.copy(tau), np.copy(sensible), np.copy(latent)])
        d = np.abs(new-old)
        if (tol[0] == 'flux'):
            ind = np.where((d[0, :] > tol[1])+(d[1, :] > tol[2]) +
                            (d[2, :] > tol[3]))
        elif (tol[0] == 'ref'):
            ind = np.where((d[0, :] > tol[1])+(d[1, :] > tol[2]) +
                            (d[2, :] > tol[3]))
        elif (tol[0] == 'all'):
            ind = np.where((d[0, :] > tol[1])+(d[1, :] > tol[2]) +
                            (d[2, :] > tol[3])+(d[3, :] > tol[4]) +
                            (d[4, :] > tol[5])+(d[5, :] > tol[6]))
        if (ind[0].size == 0):
            ii = False
        else:
            ii = True
    itera[ind] = -1
    # itera = np.where(itera > n, -1, itera)
    logging.info('method %s | # of iterations:%s', meth, it)
    logging.info('method %s | # of points that did not converge :%s', meth,
                  ind[0].size)
    # calculate output parameters
    rho = (0.34838*P)/(tv10n)
    t10n = t10n-(273.16+tlapse*ref_ht)
    # solve for zo from cd10n
    zo = ref_ht/np.exp(kappa/np.sqrt(cd10n))
    # adjust neutral cdn at any output height
    cdn = np.power(kappa/np.log(hout/zo), 2)
    cd = cd_calc(cdn, h_out[0], h_out[0], psim)
    # solve for zot, zoq from ct10n, cq10n
    zot = ref_ht/(np.exp(kappa**2/(ct10n*np.log(ref_ht/zo))))
    zoq = ref_ht/(np.exp(kappa**2/(cq10n*np.log(ref_ht/zo))))
    # adjust neutral ctn, cqn at any output height
    ctn =np.power(kappa, 2)/(np.log(h_out[0]/zo)*np.log(h_out[1]/zot))
    cqn =np.power(kappa, 2)/(np.log(h_out[0]/zo)*np.log(h_out[2]/zoq))
    ct, cq = ctcq_calc(cdn, cd, ctn, cqn, h_out[1], h_out[2], h_out[1],
                       psit, psiq)
    uref = (spd-usr/kappa*(np.log(h_in[0]/h_out[0])-psim +
            psim_calc(h_out[0]/monob, meth)))
    tref = (Ta-tsr/kappa*(np.log(h_in[1]/h_out[1])-psit +
            psit_calc(h_out[0]/monob, meth)))
    tref = tref-(CtoK+tlapse*h_out[1])
    qref = (qair-qsr/kappa*(np.log(h_in[2]/h_out[2]) -
            psit+psit_calc(h_out[2]/monob, meth)))
    if (wl == 0):
        dtwl = np.zeros(T.shape) # reset to zero if not used
    flag = np.where((q10n < 0) & (flag == "n"), "q",
                    np.where((q10n < 0) & (flag != "n"), flag+[","]+["q"],
                             flag))
    flag = np.where(((Rb < -0.5) | (Rb > 0.2)) & (flag == "n"), "l",
                    np.where(((Rb < -0.5) | (Rb > 0.2)) & (flag != "n"),
                             flag+[","]+["l"], flag))
    flag = np.where((itera == -1) & (flag == "n"), "i",
                    np.where((itera == -1) & (flag != "n"), flag+[","]+["i"],
                             flag))
    if (meth == "S80"):
        flag = np.where(((spd < 6) | (spd > 22)) & (flag == "n"), "o",
                        np.where(((spd < 6) | (spd > 22)) & (flag != "n"),
                                 flag+[","]+["o"], flag))
    elif (meth == "LP82"):
        flag = np.where(((spd < 3) | (spd > 25)) & (flag == "n"), "o",
                        np.where(((spd < 3) | (spd > 25)) & (flag != "n"),
                                 flag+[","]+["o"], flag))
    elif (meth == "YT96"):
        flag = np.where(((spd < 3) | (spd > 26)) & (flag == "n"), "o",
                        np.where(((spd < 3) | (spd > 26)) & (flag != "n"),
                                 flag+[","]+["o"], flag))
    elif (meth == "UA"):
        flag = np.where((spd > 18) & (flag == "n"), "o",
                        np.where((spd > 18) & (flag != "n"),
                                 flag+[","]+["o"], flag))
    elif (meth == "LY04"):
        flag = np.where((spd < 0.5) & (flag == "n"), "o",
                        np.where((spd < 0.5) & (flag != "n"),
                                 flag+[","]+["o"], flag))
    if (hum == None):
        rh = np.ones(sst.shape)*80
    elif (hum[0] == 'rh'):
        rh = hum[1]
        rh = np.where(rh > 100, np.nan, rh)
    elif (hum[0] == 'Td'):
        Td = hum[1] # dew point temperature (K)
        Td = np.where(Td < 200, np.copy(Td)+CtoK, np.copy(Td))
        T = np.where(T < 200, np.copy(T)+CtoK, np.copy(T))
        esd = 611.21*np.exp(17.502*((Td-273.16)/(Td-32.19)))
        es = 611.21*np.exp(17.502*((T-273.16)/(T-32.19)))
        rh = 100*esd/es
        rh = np.where(rh > 100, np.nan, rh)

    res = np.zeros((39, len(spd)))
    res[0][:] = tau
    res[1][:] = sensible
    res[2][:] = latent
    res[3][:] = monob
    res[4][:] = cd
    res[5][:] = cdn
    res[6][:] = ct
    res[7][:] = ctn
    res[8][:] = cq
    res[9][:] = cqn
    res[10][:] = tsrv
    res[11][:] = tsr
    res[12][:] = qsr
    res[13][:] = usr
    res[14][:] = psim
    res[15][:] = psit
    res[16][:] = psiq
    res[17][:] = u10n
    res[18][:] = t10n
    res[19][:] = tv10n
    res[20][:] = q10n
    res[21][:] = zo
    res[22][:] = zot
    res[23][:] = zoq
    res[24][:] = uref
    res[25][:] = tref
    res[26][:] = qref
    res[27][:] = itera
    res[28][:] = dter
    res[29][:] = dqer
    res[30][:] = dtwl
    res[31][:] = qair
    res[32][:] = qsea
    res[33][:] = Rl
    res[34][:] = Rs
    res[35][:] = Rnl
    res[36][:] = np.sqrt(np.power(wind, 2)-np.power(spd, 2))
    res[37][:] = Rb
    res[38][:] = rh

    if (out == 0):
        res[:, ind] = np.nan
    # set missing values where data have non acceptable values
    res = np.asarray([np.where((spd < 0) | (q10n < 0), np.nan,
                               res[i][:]) for i in range(39)])
    # output with pandas
    resAll = pd.DataFrame(data=res.T, index=range(len(spd)),
                        columns=["tau", "shf", "lhf", "L", "cd", "cdn", "ct",
                                 "ctn", "cq", "cqn", "tsrv", "tsr", "qsr",
                                 "usr", "psim", "psit","psiq", "u10n", "t10n",
                                 "tv10n", "q10n", "zo", "zot", "zoq", "uref",
                                 "tref", "qref", "iteration", "dter", "dqer",
                                 "dtwl", "qair", "qsea", "Rl", "Rs", "Rnl",
                                 "ug", "Rib", "rh"])
    resAll["flag"] = flag
    return resAll

