import numpy as np
import sys
import logging
from flux_subs import (kappa, CtoK, get_heights, cdn_calc, cd_calc, get_skin,
                       psim_calc, psit_calc, ctcq_calc, ctcqn_calc, get_gust,
                       gc, q_calc, qsea_calc, qsat26sea, qsat26air,
                       visc_air, psit_26, psiu_26)


def AirSeaFluxCode(spd, T, SST, lat, RH, P, hin, hout, zi=600,
                   Rl=None, Rs=None, jcool=1, meth="S88", n=10):
    """ Calculates momentum and heat fluxes using different parameterizations

    Parameters
    ----------
        meth : str
            "S80","S88","LP82","YT96","UA","LY04","C30","C35","C40","ERA5"
        spd : float
            relative wind speed in m/s (is assumed as magnitude difference
            between wind and surface current vectors)
        T : float
            air temperature in K (will convert if < 200)
        SST : float
            sea surface temperature in K (will convert if < 200)
        lat : float
            latitude (deg)
        RH : float
            relative humidity in %
        P : float
            air pressure
        hin : float
            sensor heights in m (array of 1->3 values: 1 -> u=t=q; /
            2 -> u,t=q; 3 -> u,t,q ) default 10m
        hout : float
            output height, default is 10m
        zi : int
            PBL height (m)
        Rl : float
            downward longwave radiation (W/m^2)
        Rs : float
            downward shortwave radiation (W/m^2)
        jcool : int
            0 if sst is true ocean skin temperature, else 1
        n : int
            number of iterations

    Returns
    -------
        res : array that contains
                       1. momentum flux (W/m^2)
                       2. sensible heat (W/m^2)
                       3. latent heat (W/m^2)
                       4. Monin-Obhukov length (mb)
                       5. drag coefficient (cd)
                       6. neutral drag coefficient (cdn)
                       7. heat exhange coefficient (ct)
                       8. neutral heat exhange coefficient (ctn)
                       9. moisture exhange coefficient (cq)
                       10. neutral moisture exhange coefficient (cqn)
                       11. star virtual temperature (tsrv)
                       12. star temperature (tsr)
                       13. star humidity (qsr)
                       14. star velocity (usr)
                       15. momentum stability function (psim)
                       16. heat stability funciton (psit)
                       17. 10m neutral velocity (u10n)
                       18. 10m neutral temperature (t10n)
                       19. 10m neutral virtual temperature (tv10n)
                       20. 10m neutral specific humidity (q10n)
                       21. surface roughness length (zo)
                       22. heat roughness length (zot)
                       23. moisture roughness length (zoq)
                       24. velocity at reference height (urefs)
                       25. temperature at reference height (trefs)
                       26. specific humidity at reference height (qrefs)
                       27. number of iterations until convergence
        ind : int
            the indices in the matrix for the points that did not converge
            after the maximum number of iterations
    The code is based on bform.f and flux_calc.R modified by S. Biri
    """
    logging.basicConfig(filename='flux_calc.log',
                        format='%(asctime)s %(message)s',level=logging.INFO)
    ref_ht, tlapse = 10, 0.0098   # reference height, lapse rate
    hh_in = get_heights(hin)      # heights of input measurements/fields
    hh_out = get_heights(hout)    # desired height of output variables
    if np.all(np.isnan(lat)):     # set latitude to 45deg if empty
        lat=45*np.ones(spd.shape)
    g = gc(lat, None)             # acceleration due to gravity
    ctn, ct, cqn, cq = (np.zeros(spd.shape)*np.nan, np.zeros(spd.shape)*np.nan,
                        np.zeros(spd.shape)*np.nan, np.zeros(spd.shape)*np.nan)
    cdn = np.zeros(spd.shape)*np.nan
    # if input values are nan break
    if (np.all(np.isnan(spd)) or np.all(np.isnan(T)) or np.all(np.isnan(SST))):
        sys.exit("input wind, T or SST is empty")
        logging.debug('all input is nan')
    if (np.all(np.isnan(RH)) and meth == "C35"):
        RH = np.ones(spd.shape)*80  # if empty set to default for COARE3.5
    elif (np.all(np.isnan(RH))):
        sys.exit("input RH is empty")
        logging.debug('input RH is empty')
    if (np.all(np.isnan(P)) and (meth == "C30" or meth == "C40")):
        P = np.ones(spd.shape)*1015  # if empty set to default for COARE3.0
    elif ((np.all(np.isnan(P))) and meth == "C35"):
        P = np.ones(spd.shape)*1013  # if empty set to default for COARE3.5
    elif (np.all(np.isnan(P))):
        sys.exit("input P is empty")
        logging.debug('input P is empty')
    if (np.all(np.isnan(Rl)) and meth == "C30"):
        Rl = np.ones(spd.shape)*150    # set to default for COARE3.0
    elif ((np.all(np.isnan(Rl)) and meth == "C35") or
          (np.all(np.isnan(Rl)) and meth == "C40")):
        Rl = np.ones(spd.shape)*370    # set to default for COARE3.5
    if (np.all(np.isnan(Rs)) and meth == "C30"):
        Rs = np.ones(spd.shape)*370  # set to default for COARE3.0
    elif ((np.all(np.isnan(Rs))) and (meth == "C35" or meth == "C40")):
        Rs = np.ones(spd.shape)*150  # set to default for COARE3.5
    if ((np.all(np.isnan(zi))) and (meth == "C30" or meth == "C35" or
        meth == "C40")):
        zi = 600  # set to default for COARE3.5
    elif ((np.all(np.isnan(zi))) and (meth == "ERA5" or meth == "UA")):
        zi = 1000
    ####
    th = np.where(np.nanmax(T) < 200, (np.copy(T)+CtoK) *
                  np.power(1000/P,287.1/1004.67),
                  np.copy(T)*np.power(1000/P,287.1/1004.67))  # potential T
    Ta = np.where(np.nanmax(T) < 200, np.copy(T)+CtoK+tlapse*hh_in[1],
                  np.copy(T)+tlapse*hh_in[1])  # convert to Kelvin if needed
    sst = np.where(np.nanmax(SST) < 200, np.copy(SST)+CtoK, np.copy(SST))
    if (meth == "C30" or meth == "C35" or meth == "C40"  or meth == "UA" or
        meth == "ERA5"):
        qsea = qsat26sea(sst, P)/1000  # surface water specific humidity (g/kg)
        Q, _ = qsat26air(T, P, RH)     # specific humidity of air (g/kg)
        qair = Q/1000
        del Q
        logging.info('method %s | qsea:%s, qair:%s', meth,
                     np.ma.median(qsea[~np.isnan(qsea)]),
                     np.ma.median(qair[~np.isnan(qair)]))
    else:
        qsea = qsea_calc(sst, P)
        qair = q_calc(Ta, RH, P)
        logging.info('method %s | qsea:%s, qair:%s', meth,
                     np.ma.median(qsea[~np.isnan(qsea)]),
                     np.ma.median(qair[~np.isnan(qair)]))
    if (np.all(np.isnan(qsea)) or np.all(np.isnan(qair))):
        print("qsea and qair cannot be nan")
        logging.info('method %s qsea and qair cannot be nan | sst:%s, Ta:%s,'
                      'P:%s, RH:%s', meth, np.ma.median(sst[~np.isnan(sst)]),
                      np.ma.median(Ta[~np.isnan(Ta)]),
                      np.ma.median(P[~np.isnan(P)]),
                      np.ma.median(RH[~np.isnan(RH)]))
    # first guesses
    dt = Ta - sst
    dq = qair - qsea
    t10n, q10n = np.copy(Ta), np.copy(qair)
    tv10n = t10n*(1 + 0.61*q10n)
    #  Zeng et al. 1998
    tv=th*(1.+0.61*qair)   # virtual potential T
    dtv=dt*(1.+0.61*qair)+0.61*th*dq
    # ------------
    rho = P*100/(287.1*tv10n)
#    rho=P*100/(287.1*sst*(1+0.61*qsea))  # Zeng et al. 1998
    lv = (2.501-0.00237*SST)*1e6
    cp = 1004.67*(1 + 0.00084*qsea)
#    cp = 1004.67*np.ones(Ta.shape)  # Zeng et al. 1998, C3.0, C3.5
    u10n = np.copy(spd)
    monob = -100*np.ones(u10n.shape)  # Monin-Obukhov length
    cdn = cdn_calc(u10n, Ta, None, lat, meth)
    psim, psit, psiq = (np.zeros(u10n.shape), np.zeros(u10n.shape),
                        np.zeros(u10n.shape))
    cd = cd_calc(cdn, hh_in[0], ref_ht, psim)
    tsr, tsrv = np.zeros(u10n.shape), np.zeros(u10n.shape)
    qsr = np.zeros(u10n.shape)
    # if jcool == 1:
    tkt = 0.001*np.ones(T.shape)
    Rnl = 0.97*(5.67e-8*np.power(sst-0.3*jcool+CtoK, 4)-Rl)
    dter = np.ones(T.shape)*0.3
    dqer = dter*0.622*lv*qsea/(287.1*np.power(sst, 2))
    if (meth == "UA"):
        wind = np.where(dtv >= 0, np.where(spd > 0.1, spd, 0.1),
                        np.sqrt(np.power(np.copy(spd), 2)+np.power(0.5, 2)))
        usr = 0.06
        for i in range(5):
            zo = 0.013*np.power(usr,2)/g+0.11*visc_air(Ta)/usr
            usr=kappa*wind/np.log(hh_in[0]/zo)
        Rb = g*hh_in[0]*dtv/(tv*wind**2)
        zol = np.where(Rb >= 0, Rb*np.log(hh_in[0]/zo) /
                       (1-5*np.where(Rb < 0.19, Rb, 0.19)),
                       Rb*np.log(hh_in[0]/zo))
        monob = hh_in[0]/zol
        zo = 0.013*np.power(usr, 2)/g + 0.11*visc_air(Ta)/usr
        zot = zo/np.exp(2.67*np.power(usr*zo/visc_air(Ta), 0.25)-2.57)
        zoq = zot
        logging.info('method %s | wind:%s, usr:%s, '
                     'zo:%s, zot:%s, zoq:%s, Rb:%s, monob:%s', meth,
                     np.ma.median(wind[~np.isnan(wind)]),
                     np.ma.median(usr[~np.isnan(usr)]),
                     np.ma.median(zo[~np.isnan(zo)]),
                     np.ma.median(zot[~np.isnan(zot)]),
                     np.ma.median(zoq[~np.isnan(zoq)]),
                     np.ma.median(Rb[~np.isnan(Rb)]),
                     np.ma.median(monob[~np.isnan(monob)]))
    elif (meth == "ERA5"):
        wind = np.sqrt(np.power(np.copy(spd), 2)+np.power(0.5, 2))
        usr = np.sqrt(cd*np.power(wind, 2))
        Rb = ((g*hh_in[0]*((2*dt)/(Ta+sst-g*hh_in[0]) +
                0.61*dq))/np.power(wind, 2))
        zo = 0.11*visc_air(Ta)/usr+0.018*np.power(usr, 2)/g
        zot = 0.40*visc_air(Ta)/usr
        zol = (Rb*((np.log((hh_in[0]+zo)/zo)-psim_calc((hh_in[0]+zo) /
               monob, meth)+psim_calc(zo/monob, meth)) /
               (np.log((hh_in[0]+zo)/zot) -
               psit_calc((hh_in[0]+zo)/monob, meth) +
               psit_calc(zot/monob, meth))))
        monob = hh_in[0]/zol
        logging.info('method %s | wind:%s, usr:%s, '
                     'zo:%s, zot:%s, Rb:%s, monob:%s', meth,
                     np.ma.median(wind[~np.isnan(wind)]),
                     np.ma.median(usr[~np.isnan(usr)]),
                     np.ma.median(zo[~np.isnan(zo)]),
                     np.ma.median(zot[~np.isnan(zot)]),
                     np.ma.median(Rb[~np.isnan(Rb)]),
                     np.ma.median(monob[~np.isnan(monob)]))
    elif (meth == "C30" or meth == "C35" or meth == "C40"):
        wind = np.sqrt(np.power(np.copy(spd), 2)+np.power(0.5, 2))
        usr = np.sqrt(cd*np.power(wind, 2))
        a = 0.011*np.ones(T.shape)
        a = np.where(wind > 10, 0.011+(wind-10)/(18-10)*(0.018-0.011),
                         np.where(wind > 18, 0.018, a))
        zo = a*np.power(usr, 2)/g+0.11*visc_air(T)/usr
        rr = zo*usr/visc_air(T)
        zoq = np.minimum(5e-5/np.power(rr, 0.6), 1.15e-4)
        zot = zoq
        Rb = g*hh_in[0]*dtv/((T+CtoK)*np.power(wind, 2))
        zol =  (Rb*((np.log((hh_in[0]+zo)/zo)-psim_calc((hh_in[0]+zo) /
               monob, meth)+psim_calc(zo/monob, meth)) /
               (np.log((hh_in[0]+zo)/zot) -
               psit_calc((hh_in[0]+zo)/monob, meth) +
               psit_calc(zot/monob, meth))))
        monob = hh_in[0]/zol
        logging.info('method %s | wind:%s, usr:%s, '
                     'zo:%s, zot:%s, Rb:%s, monob:%s', meth,
                     np.ma.median(wind[~np.isnan(wind)]),
                     np.ma.median(usr[~np.isnan(usr)]),
                     np.ma.median(zo[~np.isnan(zo)]),
                     np.ma.median(zot[~np.isnan(zot)]),
                     np.ma.median(Rb[~np.isnan(Rb)]),
                     np.ma.median(monob[~np.isnan(monob)]))
    else:
        wind = np.copy(spd)
        zo, zot = 0.0001*np.ones(u10n.shape), 0.0001*np.ones(u10n.shape)
        usr = np.sqrt(cd*np.power(wind, 2))
        logging.info('method %s | wind:%s, usr:%s, '
                     'zo:%s, zot:%s, Rb:%s, monob:%s', meth,
                     np.ma.median(wind[~np.isnan(wind)]),
                     np.ma.median(usr[~np.isnan(usr)]),
                     np.ma.median(zo[~np.isnan(zo)]),
                     np.ma.median(zot[~np.isnan(zot)]),
                     np.ma.median(monob[~np.isnan(monob)]))
    tsr = (dt+dter*jcool)*kappa/(np.log(hin[1]/zot) -
                                 psit_calc(hh_in[1]/monob, meth))
    qsr = (dq+dqer*jcool)*kappa/(np.log(hin[2]/zot) -
                                 psit_calc(hh_in[2]/monob, meth))
    # tolerance for u,t,q,usr,tsr,qsr
    tol = np.array([0.01, 0.01, 5e-05, 0.005, 0.001, 5e-07])  #[1e-06, 0.01, 5e-07, 1e-06, 0.001, 5e-07]
    it, ind = 0, np.where(spd > 0)
    ii, itera = True, np.zeros(spd.shape)*np.nan
    while np.any(ii):
        it += 1
        if it > n:
            break
        old = np.array([np.copy(u10n), np.copy(t10n), np.copy(q10n),
                       np.copy(usr), np.copy(tsr), np.copy(qsr)])
        cdn[ind] = cdn_calc(u10n[ind], Ta[ind], None, lat[ind], meth)
        if (np.all(np.isnan(cdn))):
            break
            logging.info('break %s at iteration %s cdn<0', meth, it)
        zo[ind] = ref_ht/np.exp(kappa/np.sqrt(cdn[ind]))
        psim[ind] = psim_calc(hh_in[0]/monob[ind], meth)
        cd[ind] = cd_calc(cdn[ind], hh_in[0], ref_ht, psim[ind])
        ctn[ind], cqn[ind] = ctcqn_calc(hh_in[1]/monob[ind], cdn[ind],
                                        u10n[ind], zo[ind], Ta[ind], meth)
        psit[ind] = psit_calc(hh_in[1]/monob[ind], meth)
        psiq[ind] = psit_calc(hh_in[2]/monob[ind], meth)
        ct[ind], cq[ind] = ctcq_calc(cdn[ind], cd[ind], ctn[ind], cqn[ind],
                                     hh_in[1], hh_in[2], ref_ht,
                                     psit[ind], psiq[ind])
        if (meth == "UA"):
            usr[ind] = np.where(hh_in[0]/monob[ind] < -1.574, kappa*wind[ind] /
                                (np.log(-1.574*monob[ind]/zo[ind]) -
                                psim_calc(-1.574, meth) +
                                psim_calc(zo[ind]/monob[ind], meth) +
                                1.14*(np.power(-hh_in[0]/monob[ind], 1/3) -
                                np.power(1.574, 1/3))),
                                np.where((hh_in[0]/monob[ind] > -1.574) &
                                (hh_in[0]/monob[ind] < 0),
                                kappa*wind[ind]/(np.log(hh_in[0]/zo[ind]) -
                                psim_calc(hh_in[0]/monob[ind], meth) +
                                psim_calc(zo[ind]/monob[ind], meth)),
                                np.where((hh_in[0]/monob[ind] > 0) &
                                (hh_in[0]/monob[ind] < 1),
                                kappa*wind[ind]/(np.log(hh_in[0]/zo[ind]) +
                                5*hh_in[0]/monob[ind]-5*zo[ind]/monob[ind]),
                                kappa*wind[ind]/(np.log(monob[ind]/zo[ind]) +
                                5-5*zo[ind]/monob[ind] +
                                5*np.log(hh_in[0]/monob[ind]) +
                                hh_in[0]/monob[ind]-1))))
                                # Zeng et al. 1998 (7-10)
            tsr[ind] = np.where(hh_in[1]/monob[ind] < -0.465, kappa*(dt[ind] +
                                dter[ind]*jcool) /
                                (np.log((-0.465*monob[ind])/zot[ind]) -
                                psit_calc(-0.465, meth)+0.8 *
                                (np.power(0.465, -1/3) -
                                np.power(-hh_in[1]/monob[ind], -1/3))),
                                np.where((hh_in[1]/monob[ind] > -0.465) &
                                (hh_in[1]/monob[ind] < 0),
                                kappa*(dt[ind]+dter[ind]*jcool) /
                                (np.log(hh_in[1]/zot[ind]) -
                                psit_calc(hh_in[1]/monob[ind], meth) +
                                psit_calc(zot[ind]/monob[ind], meth)),
                                np.where((hh_in[1]/monob[ind] > 0) &
                                (hh_in[1]/monob[ind] < 1),
                                kappa*(dt[ind]+dter[ind]*jcool) /
                                (np.log(hh_in[1]/zot[ind]) +
                                5*hh_in[1]/monob[ind]-5*zot[ind]/monob[ind]),
                                kappa*(dt[ind]+dter[ind]*jcool) /
                                (np.log(monob[ind]/zot[ind])+5 -
                                5*zot[ind]/monob[ind] +
                                5*np.log(hh_in[1]/monob[ind]) +
                                hh_in[1]/monob[ind]-1))))
                                # Zeng et al. 1998 (11-14)
            qsr[ind] = np.where(hh_in[2]/monob[ind] < -0.465, kappa*(dq[ind] +
                                dqer[ind]*jcool) /
                                (np.log((-0.465*monob[ind])/zoq[ind]) -
                                psit_calc(-0.465, meth) +
                                psit_calc(zoq[ind]/monob[ind], meth) +
                                0.8*(np.power(0.465, -1/3) -
                                np.power(-hh_in[2]/monob[ind], -1/3))),
                                np.where((hh_in[2]/monob[ind] > -0.465) &
                                (hh_in[2]/monob[ind] < 0),
                                kappa*(dq[ind]+dqer[ind]*jcool) /
                                (np.log(hh_in[1]/zot[ind]) -
                                psit_calc(hh_in[2]/monob[ind], meth) +
                                psit_calc(zoq[ind]/monob[ind], meth)),
                                np.where((hh_in[2]/monob[ind] > 0) &
                                (hh_in[2]/monob[ind]<1), kappa*(dq[ind] +
                                dqer[ind]*jcool) /
                                (np.log(hh_in[1]/zoq[ind]) +
                                5*hh_in[2]/monob[ind]-5*zoq[ind]/monob[ind]),
                                kappa*(dq[ind]+dqer[ind]*jcool) /
                                (np.log(monob[ind]/zoq[ind])+5 -
                                5*zoq[ind]/monob[ind] +
                                5*np.log(hh_in[2]/monob[ind]) +
                                hh_in[2]/monob[ind]-1))))
        elif (meth == "C30" or meth == "C35" or meth == "C40"):
            usr[ind] = (wind[ind]*kappa/(np.log(hh_in[0]/zo[ind]) -
                        psiu_26(hh_in[0]/monob[ind], meth)))
            logging.info('method %s | dter = %s | Rnl = %s '
                         '| usr = %s | tsr = %s | qsr = %s', meth,
                         np.ma.median(dter[~np.isnan(dter)]),
                         np.ma.median(Rnl[~np.isnan(Rnl)]),
                         np.ma.median(usr[~np.isnan(usr)]),
                         np.ma.median(tsr[~np.isnan(tsr)]),
                         np.ma.median(qsr[~np.isnan(qsr)]))
            qsr[ind] = ((dq[ind]+dqer[ind]*jcool)*(kappa /
                        (np.log(hin[2]/zoq[ind])-psit_26(hin[2]/monob[ind]))))
            tsr[ind] = ((dt[ind]+dter[ind]*jcool)*(kappa /
                        (np.log(hin[1]/zot[ind])-psit_26(hin[1]/monob[ind]))))
        else:
            usr[ind] = np.sqrt(cd[ind]*np.power(wind[ind], 2))
            #        = (wind[ind]*kappa/(np.log(hh_in[0]/zo[ind]) -
            #           psim_calc(hh_in[0]/monob[ind], meth)))
            tsr[ind] = ct[ind]*wind[ind]*(dt[ind]+dter[ind]*jcool)/usr[ind]
            qsr[ind] = cq[ind]*wind[ind]*(dq[ind]+dqer[ind]*jcool)/usr[ind]
        if (jcool == 1):
            dter[ind], dqer[ind], tkt[ind] = get_skin(sst[ind], qsea[ind],
                                                      rho[ind], Rl[ind],
                                                      Rs[ind], Rnl[ind],
                                                      cp[ind], lv[ind],
                                                      np.copy(tkt[ind]),
                                                      usr[ind], tsr[ind],
                                                      qsr[ind], lat[ind])
        else:
            dter[ind] = np.zeros(spd[ind].shape)
            dqer[ind] = np.zeros(spd[ind].shape)
            tkt[ind] = np.zeros(spd[ind].shape)
        Rnl[ind] = 0.97*(5.67e-8*np.power(SST[ind] -
                         dter[ind]*jcool+CtoK, 4)-Rl[ind])
        t10n[ind] = (Ta[ind] -
                     (tsr[ind]*(np.log(hh_in[1]/ref_ht)-psit[ind])/kappa))
        q10n[ind] = (qair[ind] -
                     (qsr[ind]*(np.log(hh_in[2]/ref_ht)-psiq[ind])/kappa))
        tv10n[ind] = t10n[ind]*(1+0.61*q10n[ind])
        if (meth == "UA"):
            tsrv[ind] = tsr[ind]*(1.+0.61*qair[ind])+0.61*th[ind]*qsr[ind]
            monob[ind] = (tv[ind]*np.power(usr[ind], 2))/(kappa*g[ind]*tsrv[ind])
        elif (meth == "C30" or meth == "C35" or meth == "C40"):
            tsrv[ind] = tsr[ind]+0.61*(T[ind]+CtoK)*qsr[ind]
            zol[ind] = (kappa*g[ind]*hh_in[0]/(T[ind]+CtoK)*(tsr[ind] +
                        0.61*(T[ind]+CtoK)*qsr[ind])/np.power(usr[ind], 2))
            monob[ind] = hh_in[0]/zol[ind]
        elif (meth == "ERA5"):
            tsrv[ind] = tsr[ind]+0.61*t10n[ind]*qsr[ind]
            Rb[ind] = ((g[ind]*hh_in[0]*((2*dt[ind])/(Ta[ind]+sst[ind]-g[ind] *
                       hh_in[0])+0.61*dq[ind]))/np.power(wind[ind], 2))
            zo[ind] = (0.11*visc_air(Ta[ind])/usr[ind]+0.018 *
                       np.power(usr[ind], 2)/g[ind])
            zot[ind] = 0.40*visc_air(Ta[ind])/usr[ind]
            zol[ind] = (Rb[ind]*((np.log((hh_in[0]+zo[ind])/zo[ind]) -
                        psim_calc((hh_in[0]+zo[ind])/monob[ind], meth) +
                        psim_calc(zo[ind]/monob[ind], meth)) /
                        (np.log((hh_in[0]+zo[ind])/zot[ind]) -
                        psit_calc((hh_in[0]+zo[ind])/monob[ind], meth) +
                        psit_calc(zot[ind]/monob[ind], meth))))
            monob[ind] = hh_in[0]/zol[ind]
        else:
            tsrv[ind] = tsr[ind]+0.61*t10n[ind]*qsr[ind]
            monob[ind] = (tv10n[ind]*usr[ind]**2)/(g[ind]*kappa*tsrv[ind])
        psim[ind] = psim_calc(hh_in[0]/monob[ind], meth)
        psit[ind] = psit_calc(hh_in[1]/monob[ind], meth)
        psiq[ind] = psit_calc(hh_in[2]/monob[ind], meth)
        if (meth == "UA"):
            wind[ind] = np.where(dtv[ind] >= 0, np.where(spd[ind] > 0.1,
                                 spd[ind], 0.1),
                                 np.sqrt(np.power(np.copy(spd[ind]), 2) +
                                 np.power(get_gust(1, tv[ind], usr[ind],
                                 tsrv[ind], zi, lat[ind]), 2)))
                                 # Zeng et al. 1998 (20)
            u10n[ind] = (wind[ind]+(usr[ind]/kappa)*(np.log(hh_in[0]/10) -
                         psim[ind]))
            u10n[u10n < 0] = np.nan
        elif (meth == "C30" or meth == "C35" or meth == "C40"):
            wind[ind] = np.sqrt(np.power(np.copy(spd[ind]), 2) +
                                np.power(get_gust(1.2, Ta[ind], usr[ind],
                                tsrv[ind], zi, lat[ind]), 2))
            u10n[ind] = ((wind[ind] + usr[ind]/kappa*(np.log(10/hh_in[0])-
                         psiu_26(10/monob[ind], meth) +
                         psiu_26(hh_in[0]/monob[ind], meth)) +
                         psiu_26(10/monob[ind], meth)*usr[ind]/kappa /
                         (wind[ind]/spd[ind])))
            u10n[u10n < 0] = np.nan
        elif (meth == "ERA5"):
            wind[ind] = np.sqrt(np.power(np.copy(spd[ind]), 2) +
                                np.power(get_gust(1, Ta[ind], usr[ind],
                                tsrv[ind], zi, lat[ind]), 2))
            u10n[ind] = (spd[ind]+(usr[ind]/kappa)*(np.log(hh_in[0] /
                         ref_ht)-psim[ind]))
            u10n[u10n < 0] = np.nan
        else:
            u10n[ind] = (wind[ind]+(usr[ind]/kappa)*(np.log(hh_in[0]/10) -
                         psim[ind]))
            u10n[u10n < 0] = np.nan
        itera[ind] = np.ones(1)*it
        new = np.array([np.copy(u10n), np.copy(t10n), np.copy(q10n),
                       np.copy(usr), np.copy(tsr), np.copy(qsr)])
        d = np.abs(new-old)
        ind = np.where((d[0, :] > tol[0])+(d[1, :] > tol[1]) +
                      (d[2, :] > tol[2])+(d[3, :] > tol[3]) +
                      (d[4, :] > tol[4])+(d[5, :] > tol[5]))
        if (ind[0].size == 0):
            ii = False
        else:
            ii = True
        # if ((it == 3) or (it == 6) or (it == 10)):
        #     print('Method {}, # {} tol u10n: {} | t10n: {} | q10n: {} | '
        #           'u*: {} | t*: {} | q*: {}'.format(meth, it,
        #           np.ma.max(d[0, ~np.isnan(d[0,:])]),
        #           np.ma.max(d[1, ~np.isnan(d[1,:])]),
        #           np.ma.max(d[2, ~np.isnan(d[2,:])]),
        #           np.ma.max(d[3, ~np.isnan(d[3,:])]),
        #           np.ma.max(d[4, ~np.isnan(d[4,:])]),
        #           np.ma.max(d[5, ~np.isnan(d[5,:])])),
        #           file=open('tol_mid.txt','a'))
    logging.info('method %s | # of iterations:%s', meth, it)
    logging.info('method %s | # of points that did not converge :%s', meth,
                  ind[0].size)
    # calculate output parameters
    rho = (0.34838*P)/(tv10n)
#    rho = P*100./(287.1*(T+CtoK)*(1+0.61*qair))  # C35
    t10n = t10n-(273.16+tlapse*ref_ht)
    sensible = -rho*cp*usr*tsr
    latent = -rho*lv*usr*qsr
    if (meth == "C30" or meth == "C35" or meth == "C40" or meth == "UA" or
        meth == "ERA5"):
        tau = rho*np.power(usr, 2)*(spd/wind)
    else:
        tau = rho*np.power(usr, 2)
    zo = ref_ht/np.exp(kappa/cdn**0.5)
    zot = ref_ht/(np.exp(kappa**2/(ctn*np.log(ref_ht/zo))))
    zoq = ref_ht/(np.exp(kappa**2/(cqn*np.log(ref_ht/zo))))
    urefs = (spd-(usr/kappa)*(np.log(hh_in[0]/hh_out[0])-psim +
             psim_calc(hh_out[0]/monob, meth)))
    trefs = (Ta-(tsr/kappa)*(np.log(hh_in[1]/hh_out[1])-psit +
             psit_calc(hh_out[0]/monob, meth)))
    trefs = trefs-(273.16+tlapse*hh_out[1])
    qrefs = (qair-(qsr/kappa)*(np.log(hh_in[2]/hh_out[2]) -
             psit+psit_calc(hh_out[2]/monob, meth)))
    res = np.zeros((27, len(spd)))
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
    res[16][:] = u10n
    res[17][:] = t10n
    res[18][:] = tv10n
    res[19][:] = q10n
    res[20][:] = zo
    res[21][:] = zot
    res[22][:] = zoq
    res[23][:] = urefs
    res[24][:] = trefs
    res[25][:] = qrefs
    res[26][:] = itera
    return res, ind
