import numpy as np
import sys
import logging
from flux_subs import (kappa, CtoK, get_heights, get_skin, get_gust, get_L,
                       get_hum,
                       psim_calc, psit_calc, psit_26, psiu_26,
                       cdn_calc, cd_calc, ctcq_calc, ctcqn_calc)


def AirSeaFluxCode_L(spd, T, SST, lat=None, hum=None, P=None,
                       hin=18, hout=10, Rl=None, Rs=None, cskin=None,
                       gust=None, meth="S80", qmeth="Buck2", tol=None, n=10,
                       out=0, L=None):
    """ Calculates momentum and heat fluxes using different parameterizations

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
        gust : int
            3x1 [x, beta, zi] x=1 to include the effect of gustiness, else 0
            beta gustiness parameter, beta=1 for UA, beta=1.2 for COARE
            zi PBL height (m) 600 for COARE, 1000 for UA and ERA5, 800 default
            default for COARE [1, 1.2, 600]
            default for UA, ERA5 [1, 1, 1000]
            default else [1, 1.2, 800]
        meth : str
            "S80","S88","LP82","YT96","UA","LY04","C30","C35","C40","ERA5"
        qmeth : str
            is the saturation evaporation method to use amongst
            "HylandWexler","Hardy","Preining","Wexler","GoffGratch","CIMO",
            "MagnusTetens","Buck","Buck2","WMO","WMO2000","Sonntag","Bolton",
            "IAPWS","MurphyKoop"]
            default is Buck2
        tol : float
           4x1 or 7x1 [option, lim1-3 or lim1-6]
           option : 'flux' to set tolerance limits for fluxes only lim1-3
           option : 'ref' to set tolerance limits for height adjustment lim-1-3
           option : 'all' to set tolerance limits for both fluxes and height
                    adjustment lim1-6 ['all', 0.01, 0.01, 5e-05, 0.01, 1, 1]
           default is tol=['flux', 0.01, 1, 1]
        n : int
            number of iterations (defautl = 10)
        out : int
            set 0 to set points that have not converged to missing (default)
            set 1 to keep points
        L : int
           Monin-Obukhov length definition options
           0 : default for S80, S88, LP82, YT96 and LY04
           1 : following UA (Zeng et al., 1998), default for UA
           2 : following ERA5 (IFS Documentation cy46r1), default for ERA5
           3 : COARE3.5 (Edson et al., 2013), default for C30, C35 and C40
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
                       16. heat stability function (psit)
                       17. moisture stability function (psiq)
                       18. 10m neutral velocity (u10n)
                       19. 10m neutral temperature (t10n)
                       20. 10m neutral virtual temperature (tv10n)
                       21. 10m neutral specific humidity (q10n)
                       22. surface roughness length (zo)
                       23. heat roughness length (zot)
                       24. moisture roughness length (zoq)
                       25. velocity at reference height (uref)
                       26. temperature at reference height (tref)
                       27. specific humidity at reference height (qref)
                       28. number of iterations until convergence
        ind : int
            the indices in the matrix for the points that did not converge
            after the maximum number of iterations
    The code is based on bform.f and flux_calc.R modified by S. Biri
    """
    logging.basicConfig(filename='flux_calc.log',
                        format='%(asctime)s %(message)s',level=logging.INFO)
    if ((type(spd) != np.ndarray) or (type(T) != np.ndarray) or
        (type(SST) != np.ndarray)):
        sys.exit("input type of spd, T and SST should be numpy.ndarray")
    ref_ht, tlapse = 10, 0.0098        # reference height, lapse rate
    h_in = get_heights(hin, len(spd))  # heights of input measurements/fields
    h_out = get_heights(hout, 1)       # desired height of output variables
    # if input values are nan break
    if meth not in ["S80", "S88", "LP82", "YT96", "UA", "LY04", "C30", "C35",
                    "C40","ERA5"]:
        sys.exit("unknown method")
    if qmeth not in ["HylandWexler", "Hardy", "Preining", "Wexler", "CIMO",
                     "GoffGratch", "MagnusTetens", "Buck", "Buck2", "WMO",
                     "WMO2000", "Sonntag", "Bolton", "IAPWS", "MurphyKoop"]:
        sys.exit("unknown q-method")
    if (np.all(np.isnan(spd)) or np.all(np.isnan(T)) or np.all(np.isnan(SST))):
        sys.exit("input wind, T or SST is empty")
        logging.debug('all spd or T or SST input is nan')
    if (np.all(lat == None)):  # set latitude to 45deg if empty
        lat = 45*np.ones(spd.shape)
    elif ((np.all(lat != None)) and (np.size(lat) == 1)):
        lat = np.ones(spd.shape)*np.copy(lat)
    if ((np.all(P == None)) and (meth == "C30" or meth == "C40")):
        P = np.ones(spd.shape)*1015  # if empty set to default for COARE3.0
    elif ((np.all(P == None)) or np.all(np.isnan(P))):
        P = np.ones(spd.shape)*1013
        logging.debug('input P is empty and set to 1013hPa')
    elif (((np.all(P != None)) or np.all(~np.isnan(P))) and np.size(P) == 1):
        P = np.ones(spd.shape)*np.copy(P)
    if ((np.all(Rl == None) or np.all(np.isnan(Rl))) and meth == "C30"):
        Rl = np.ones(spd.shape)*150    # set to default for COARE3.0
    elif (((np.all(Rl == None) or np.all(np.isnan(Rl))) and meth == "C35") or
          ((np.all(Rl == None) or np.all(np.isnan(Rl))) and meth == "C40")):
        Rl = np.ones(spd.shape)*370    # set to default for COARE3.5
    elif (np.all(Rl == None) or np.all(np.isnan(Rl))):
        Rl = np.ones(spd.shape)*370    # set to default for COARE3.5
    if ((np.all(Rs == None) or np.all(np.isnan(Rs))) and meth == "C30"):
        Rs = np.ones(spd.shape)*370  # set to default for COARE3.0
    elif (np.all(Rs == None) or np.all(np.isnan(Rs))):
        Rs = np.ones(spd.shape)*150  # set to default for COARE3.5
    if ((gust == None) and (meth == "C30" or meth == "C35" or meth == "C40")):
        gust = [1, 1.2, 600]
    elif ((gust == None) and (meth == "UA" or meth == "ERA5")):
        gust = [1, 1, 1000]
    elif (gust == None):
        gust = [1, 1.2, 800]
    elif (np.size(gust) < 3):
        sys.exit("gust input must be a 3x1 array")
    if (tol == None):
        tol = ['flux', 0.01, 1, 1]
    elif (tol[0] not in ['flux', 'ref', 'all']):
        sys.exit("unknown tolerance input")
    if ((cskin == None) and (meth == "S80" or meth == "S88" or meth == "LP82"
                             or meth == "YT96")):
       cskin = 0
    elif ((cskin == None) and (meth == "UA" or meth == "LY04" or meth == "C30"
                               or meth == "C35" or meth == "C40"
                               or meth == "ERA5")):
       cskin = 1
    logging.info('method %s, inputs: lat: %s | P: %s | Rl: %s |'
                 ' Rs: %s | gust: %s | cskin: %s', meth,
                 np.nanmedian(lat), np.nanmedian(P), np.nanmedian(Rl),
                 np.nanmedian(Rs), gust, cskin)
    if (L not in [None, 0, 1, 2, 3]):
        sys.exit("L input must be either None, 0, 1, 2 or 3")
    if ((L == None) and (meth == "S80" or meth == "S88" or meth == "LP82"
                             or meth == "YT96" or meth == "LY04")):
       L = 0
    elif ((L == None) and (meth == "UA")):
       L = 1
    elif ((L == None) and (meth == "ERA5")):
       L = 2
    elif ((L == None) and (meth == "C30" or meth == "C35" or meth == "C40")):
       L = 3
    ####
    th = np.where(T < 200, (np.copy(T)+CtoK) *
                  np.power(1000/P,287.1/1004.67),
                  np.copy(T)*np.power(1000/P,287.1/1004.67))  # potential T
    Ta = np.where(T < 200, np.copy(T)+CtoK+tlapse*h_in[1],
                  np.copy(T)+tlapse*h_in[1])  # convert to Kelvin if needed
    sst = np.where(SST < 200, np.copy(SST)+CtoK, np.copy(SST))
    qair, qsea = get_hum(hum, T, sst, P, qmeth)
    logging.info('method %s and q method %s | qsea:%s, qair:%s', meth, qmeth,
                  np.nanmedian(qsea), np.nanmedian(qair))
    if (np.all(np.isnan(qsea)) or np.all(np.isnan(qair))):
        print("qsea and qair cannot be nan")
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
    lv = (2.501-0.00237*(sst-CtoK))*1e6
    cp = 1004.67*(1 + 0.00084*qsea)
    u10n = np.copy(spd)
    monob = -100*np.ones(spd.shape)
    cdn = cdn_calc(u10n, Ta, None, lat, meth)
    ctn, ct, cqn, cq = (np.zeros(spd.shape)*np.nan, np.zeros(spd.shape)*np.nan,
                        np.zeros(spd.shape)*np.nan, np.zeros(spd.shape)*np.nan)
    psim, psit, psiq = (np.zeros(spd.shape), np.zeros(spd.shape),
                        np.zeros(spd.shape))
    cd = cd_calc(cdn, h_in[0], ref_ht, psim)
    tsr, tsrv = np.zeros(spd.shape), np.zeros(spd.shape)
    qsr = np.zeros(spd.shape)
    # cskin parameters
    tkt = 0.001*np.ones(T.shape)
    Rnl = 0.97*(5.67e-8*np.power(sst-0.3*cskin+CtoK, 4)-Rl)
    dter = np.ones(T.shape)*0.3
    dqer = dter*0.622*lv*qsea/(287.1*np.power(sst, 2))
    if (gust[0] == 1 and meth == "UA"):
        wind = np.where(dtv >= 0, np.where(spd > 0.1, spd, 0.1),
                        np.sqrt(np.power(np.copy(spd), 2)+np.power(0.5, 2)))
    elif (gust[0] == 1):
        wind = np.sqrt(np.power(np.copy(spd), 2)+np.power(0.5, 2))
    elif (gust[0] == 0):
        wind = np.copy(spd)
    usr = np.sqrt(cd*np.power(wind, 2))
    zo = 0.0001*np.ones(spd.shape)
    zot, zoq = 0.0001*np.ones(spd.shape), 0.0001*np.ones(spd.shape)
    monob = -100*np.ones(spd.shape)  # Monin-Obukhov length
    tsr = (dt+dter*cskin)*kappa/(np.log(h_in[1]/zot) -
                                 psit_calc(h_in[1]/monob, meth))
    qsr = (dq+dqer*cskin)*kappa/(np.log(h_in[2]/zoq) -
                                 psit_calc(h_in[2]/monob, meth))
    it, ind = 0, np.where(spd > 0)
    ii, itera = True, np.zeros(spd.shape)*np.nan
    tau = 0.05*np.ones(spd.shape)
    sensible = 5*np.ones(spd.shape)
    latent = 65*np.ones(spd.shape)
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
        cdn[ind] = cdn_calc(u10n[ind], Ta[ind], None, lat[ind], meth)
        if (np.all(np.isnan(cdn))):
            break
            logging.info('break %s at iteration %s cdn<0', meth, it)
        zo[ind] = ref_ht/np.exp(kappa/np.sqrt(cdn[ind]))
        psim[ind] = psim_calc(h_in[0, ind]/monob[ind], meth)
        cd[ind] = cd_calc(cdn[ind], h_in[0, ind], ref_ht, psim[ind])
        ctn[ind], cqn[ind] = ctcqn_calc(h_in[1, ind]/monob[ind], cdn[ind],
                                        u10n[ind], zo[ind], Ta[ind], meth)
        zot[ind] = ref_ht/(np.exp(np.power(kappa, 2) /
                           (ctn[ind]*np.log(ref_ht/zo[ind]))))
        zoq[ind] = ref_ht/(np.exp(np.power(kappa, 2) /
                           (cqn[ind]*np.log(ref_ht/zo[ind]))))
        psit[ind] = psit_calc(h_in[1, ind]/monob[ind], meth)
        psiq[ind] = psit_calc(h_in[2, ind]/monob[ind], meth)
        ct[ind], cq[ind] = ctcq_calc(cdn[ind], cd[ind], ctn[ind], cqn[ind],
                                      h_in[1, ind], h_in[2, ind], ref_ht,
                                      psit[ind], psiq[ind])
        if (meth == "UA"):
            usr[ind] = np.where(h_in[0, ind]/monob[ind] < -1.574,
                                kappa*wind[ind] /
                                (np.log(-1.574*monob[ind]/zo[ind]) -
                                psim_calc(-1.574, meth) +
                                psim_calc(zo[ind]/monob[ind], meth) +
                                1.14*(np.power(-h_in[0, ind]/monob[ind], 1/3) -
                                np.power(1.574, 1/3))),
                                np.where((h_in[0, ind]/monob[ind] > -1.574) &
                                (h_in[0, ind]/monob[ind] < 0),
                                kappa*wind[ind]/(np.log(h_in[0, ind]/zo[ind]) -
                                psim_calc(h_in[0, ind]/monob[ind], meth) +
                                psim_calc(zo[ind]/monob[ind], meth)),
                                np.where((h_in[0, ind]/monob[ind] > 0) &
                                (h_in[0, ind]/monob[ind] < 1),
                                kappa*wind[ind]/(np.log(h_in[0, ind]/zo[ind]) +
                                5*h_in[0, ind]/monob[ind]-5*zo[ind] /
                                monob[ind]), kappa*wind[ind] /
                                (np.log(monob[ind]/zo[ind]) +
                                5-5*zo[ind]/monob[ind] +
                                5*np.log(h_in[0, ind]/monob[ind]) +
                                h_in[0, ind]/monob[ind]-1))))
                                # Zeng et al. 1998 (7-10)
            tsr[ind] = np.where(h_in[1, ind]/monob[ind] < -0.465,
                                kappa*(dt[ind] +
                                dter[ind]*cskin) /
                                (np.log((-0.465*monob[ind])/zot[ind]) -
                                psit_calc(-0.465, meth)+0.8 *
                                (np.power(0.465, -1/3) -
                                np.power(-h_in[1, ind]/monob[ind], -1/3))),
                                np.where((h_in[1, ind]/monob[ind] > -0.465) &
                                (h_in[1, ind]/monob[ind] < 0),
                                kappa*(dt[ind]+dter[ind]*cskin) /
                                (np.log(h_in[1, ind]/zot[ind]) -
                                psit_calc(h_in[1, ind]/monob[ind], meth) +
                                psit_calc(zot[ind]/monob[ind], meth)),
                                np.where((h_in[1, ind]/monob[ind] > 0) &
                                (h_in[1, ind]/monob[ind] < 1),
                                kappa*(dt[ind]+dter[ind]*cskin) /
                                (np.log(h_in[1, ind]/zot[ind]) +
                                5*h_in[1, ind]/monob[ind]-5*zot[ind] /
                                monob[ind]),
                                kappa*(dt[ind]+dter[ind]*cskin) /
                                (np.log(monob[ind]/zot[ind])+5 -
                                5*zot[ind]/monob[ind] +
                                5*np.log(h_in[1, ind]/monob[ind]) +
                                h_in[1, ind]/monob[ind]-1))))
                                # Zeng et al. 1998 (11-14)
            qsr[ind] = np.where(h_in[2, ind]/monob[ind] < -0.465,
                                kappa*(dq[ind] +
                                dqer[ind]*cskin) /
                                (np.log((-0.465*monob[ind])/zoq[ind]) -
                                psit_calc(-0.465, meth) +
                                psit_calc(zoq[ind]/monob[ind], meth) +
                                0.8*(np.power(0.465, -1/3) -
                                np.power(-h_in[2, ind]/monob[ind], -1/3))),
                                np.where((h_in[2, ind]/monob[ind] > -0.465) &
                                (h_in[2, ind]/monob[ind] < 0),
                                kappa*(dq[ind]+dqer[ind]*cskin) /
                                (np.log(h_in[1, ind]/zot[ind]) -
                                psit_calc(h_in[2, ind]/monob[ind], meth) +
                                psit_calc(zoq[ind]/monob[ind], meth)),
                                np.where((h_in[2, ind]/monob[ind] > 0) &
                                (h_in[2, ind]/monob[ind]<1), kappa*(dq[ind] +
                                dqer[ind]*cskin) /
                                (np.log(h_in[1, ind]/zoq[ind]) +
                                5*h_in[2, ind]/monob[ind]-5*zoq[ind]/monob[ind]),
                                kappa*(dq[ind]+dqer[ind]*cskin) /
                                (np.log(monob[ind]/zoq[ind])+5 -
                                5*zoq[ind]/monob[ind] +
                                5*np.log(h_in[2, ind]/monob[ind]) +
                                h_in[2, ind]/monob[ind]-1))))
        elif (meth == "C30" or meth == "C35" or meth == "C40"):
            usr[ind] = (wind[ind]*kappa/(np.log(h_in[0, ind]/zo[ind]) -
                        psiu_26(h_in[0, ind]/monob[ind], meth)))
            qsr[ind] = ((dq[ind]+dqer[ind]*cskin)*(kappa/(np.log(h_in[2, ind] /
                        zoq[ind])-psit_26(h_in[2, ind]/monob[ind]))))
            tsr[ind] = ((dt[ind]+dter[ind]*cskin)*(kappa/(np.log(h_in[1, ind] /
                        zot[ind])-psit_26(h_in[1, ind]/monob[ind]))))
        else:
            usr[ind] = (wind[ind]*kappa/(np.log(h_in[0, ind]/zo[ind]) -
                        psim_calc(h_in[0, ind]/monob[ind], meth)))
            qsr[ind] = cq[ind]*wind[ind]*(dq[ind]+dqer[ind]*cskin)/usr[ind]
            tsr[ind] = ct[ind]*wind[ind]*(dt[ind]+dter[ind]*cskin)/usr[ind]
        if (cskin == 1):
            dter[ind], dqer[ind], tkt[ind] = get_skin(sst[ind], qsea[ind],
                                                      rho[ind], Rl[ind],
                                                      Rs[ind], Rnl[ind],
                                                      cp[ind], lv[ind],
                                                      np.copy(tkt[ind]),
                                                      usr[ind], tsr[ind],
                                                      qsr[ind], lat[ind])
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
        Rnl[ind] = 0.97*(5.67e-8*np.power(sst[ind]-CtoK -
                          dter[ind]*cskin+CtoK, 4)-Rl[ind])
        t10n[ind] = (Ta[ind] -
                     tsr[ind]/kappa*(np.log(h_in[1, ind]/ref_ht)-psit[ind]))
        q10n[ind] = (qair[ind] -
                     qsr[ind]/kappa*(np.log(h_in[2, ind]/ref_ht)-psiq[ind]))
        tv10n[ind] = t10n[ind]*(1+0.61*q10n[ind])
        tsrv[ind], monob[ind] = get_L(L, lat[ind], usr[ind], tsr[ind],
                                      qsr[ind], t10n[ind], tv10n[ind],
                                      qair[ind], h_in[:, ind], T[ind], Ta[ind],
                                      th[ind], tv[ind], sst[ind], dt[ind],
                                      dq[ind], wind[ind], np.copy(monob[ind]),
                                      meth)
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
        elif (gust[0] == 1 and (meth == "C30" or meth == "C35" or
                                meth == "C40")):
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
        d = np.fabs(new-old)
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
    logging.info('method %s | # of iterations:%s', meth, it)
    logging.info('method %s | # of points that did not converge :%s', meth,
                  ind[0].size)
    # calculate output parameters
    rho = (0.34838*P)/(tv10n)
    t10n = t10n-(273.16+tlapse*ref_ht)
    zo = ref_ht/np.exp(kappa/cdn**0.5)
    zot = ref_ht/(np.exp(kappa**2/(ctn*np.log(ref_ht/zo))))
    zoq = ref_ht/(np.exp(kappa**2/(cqn*np.log(ref_ht/zo))))
    uref = (spd-usr/kappa*(np.log(h_in[0]/h_out[0])-psim +
            psim_calc(h_out[0]/monob, meth)))
    tref = (Ta-tsr/kappa*(np.log(h_in[1]/h_out[1])-psit +
            psit_calc(h_out[0]/monob, meth)))
    tref = tref-(273.16+tlapse*h_out[1])
    qref = (qair-qsr/kappa*(np.log(h_in[2]/h_out[2]) -
            psit+psit_calc(h_out[2]/monob, meth)))
    res = np.zeros((28, len(spd)))
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
    if (out == 0):
        res[:, ind] = np.nan
    # set missing values where data have non acceptable values
    res = np.where(spd < 0, np.nan, res)

    return res, ind
