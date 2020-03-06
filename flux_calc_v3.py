import numpy as np
import sys
import logging
from flux_subs import (kappa, CtoK, get_heights, cdn_calc, cd_calc, get_skin,
                       psim_calc, psit_calc, ctcq_calc, ctcqn_calc, get_gust,
                       gc, q_calc, qsea_calc, qsat26sea, qsat26air,
                       visc_air, psit_26, psiu_26, psiu_40, cd_C35,
                       charnock_C35)


def flux_calc_v3(spd, T, SST, lat, RH, P, hin, hout, wcp, sigH, zi=600,
                 Rl=None, Rs=None, jcool=1, meth="S88", n=10):
    """ Calculates momentum and heat fluxes using different parameterizations

    Parameters
    ----------
        meth : str
            "S80","S88","LP82","YT96","UA","HEXOS","HEXOSwave",
            "LY04","C30","C35","ERA5"
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
        wcp : float
            phase speed of dominant waves (m/s) called in C35
        sigH : float
            significant wave height (m) called in C35
        zi : int
            PBL height (m) called in C35
        Rl : float
            downward longwave radiation (W/m^2)
        Rs : float
            downward shortwave radiation (W/m^2)
        jcool : bool
            0 if sst is true ocean skin temperature called in C35
        n : int
            number of iterations, for C35 set to 10
        out : str
            format in which to save output ('output.nc' or None)

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
        lat=45*np.ones(np.shape(spd))
    g = gc(lat, None)             # acceleration due to gravity
    ctn, ct, cqn, cq = (np.zeros(spd.shape)*np.nan, np.zeros(spd.shape)*np.nan,
                        np.zeros(spd.shape)*np.nan, np.zeros(spd.shape)*np.nan)
    # if input values are nan break
    if (np.all(np.isnan(spd)) or np.all(np.isnan(T)) or np.all(np.isnan(SST))):
        sys.exit("input wind, T or SST is empty")
        logging.debug('all input is nan')
    if (np.all(np.isnan(RH)) and meth == "C35"):
        RH = np.ones(np.shape(spd))*80  # if empty set to default for COARE3.5
    else:
        sys.exit("input RH is empty")
        logging.debug('input RH is empty')
    if (np.all(np.isnan(P)) and meth == "C35"):
        P = np.ones(np.shape(spd))*1013  # if empty set to default for COARE3.5
    else:
        sys.exit("input P is empty")
        logging.debug('input P is empty')
    if (np.all(np.isnan(Rl)) and meth == "C35"):
        Rl = np.ones(np.shape(spd))*370    # set to default for COARE3.5
    if (np.all(np.isnan(Rs)) and meth == "C35"):
        Rs = np.ones(np.shape(spd))*150  # set to default for COARE3.5
    if (np.all(np.isnan(zi)) and meth == "C35"):
        zi = 600  # set to default for COARE3.5
    elif ((np.all(np.isnan(zi)) and meth == "ERA5") or
          (np.all(np.isnan(zi)) and meth == "UA")):
        zi = 1000
    # additional parameters for C35
    waveage, seastate = 1, 1
    if np.isnan(wcp[0]):
        wcp = np.ones(np.shape(spd))*np.nan
        waveage = 0
    if np.isnan(sigH[0]):
        sigH = np.ones(np.shape(spd))*np.nan
        seastate = 0
    if waveage and seastate:
        print('Using seastate dependent parameterization')
        logging.info('Using seastate dependent parameterization')
    if waveage and ~seastate:
        print('Using waveage dependent parameterization')
        logging.info('Using waveage dependent parameterization')
    ####
    Ta = np.where(np.nanmax(T) < 200, np.copy(T)+CtoK+tlapse*hh_in[1],
                  np.copy(T)+tlapse*hh_in[1])  # convert to Kelvin if needed
    sst = np.where(np.nanmax(SST) < 200, np.copy(SST)+CtoK, np.copy(SST))
    if (meth == "C35"):
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
        logging.debug('method %s qsea and qair cannot be nan | sst:%s, Ta:%s,'
                      'P:%s, RH:%s', meth, np.ma.median(sst[~np.isnan(sst)]),
                      np.ma.median(Ta[~np.isnan(Ta)]),
                      np.ma.median(P[~np.isnan(P)]),
                      np.ma.median(RH[~np.isnan(RH)]))
    # first guesses
    inan = np.where(np.isnan(spd+T+SST+lat+RH+P))
    t10n, q10n = np.copy(Ta), np.copy(qair)
    tv10n = t10n*(1 + 0.61*q10n)
    rho = (0.34838*P)/(tv10n)
    rhoa = P*100/(287.1*(T+CtoK)*(1+0.61*qair))
    lv = (2.501-0.00237*SST)*1e6
    dt = sst - Ta
    dq = qsea - qair
    if (meth == "C35"):
        cp = 1004.67  # cpa  = 1004.67, cpv  = cpa*(1+0.84*Q)
        wetc = 0.622*lv*qsea/(287.1*sst**2)
        ug = 0.5*np.ones(np.shape(spd))
        wind = np.sqrt(np.copy(spd)**2+ug**2)
        dter = 0.3*np.ones(np.shape(spd))  # cool skin temperature depression
        dqer = wetc*dter
        Rnl = 0.97*(5.67e-8*(SST-dter*jcool+CtoK)**4-Rl)
        u10 = wind*np.log(10/1e-4)/np.log(hh_in[0]/1e-4)
        usr = 0.035*u10
        u10n = np.copy(u10)
        zo = 0.011*usr**2/g + 0.11*visc_air(T)/usr
        cdn = (kappa/np.log(10/zo))**2
        cqn = 0.00115*np.ones(np.shape(spd))
        ctn = cqn/np.sqrt(cdn)
        cd = (kappa/np.log(hh_in[0]/zo))**2
        ct = kappa/np.log(hh_in[1]/10/np.exp(kappa/ctn))
        CC = kappa*ct/cd
        Ribcu = -hh_in[0]/zi/0.004/1.2**3
        Ribu = (-g*hh_in[0]/(T+CtoK)*((dt-dter*jcool)+0.61*(T+CtoK)*(dq)) /
                wind**2)
        zetu = np.where(Ribu < 0, CC*Ribu/(1+Ribu/Ribcu),
                        CC*Ribu*(1+27/9*Ribu/CC))
        k50 = np.where(zetu > 50)  # stable, very thin M-O relative to zu
        monob = hh_in[0]/zetu
        gf = wind/spd
        usr = wind*kappa/(np.log(hh_in[0]/zo)-psiu_40(hh_in[0]/monob))
        tsr = (-(dt-dter*jcool)*kappa/(np.log(hh_in[1]/10/np.exp(kappa/ctn)) -
               psit_26(hh_in[1]/monob)))
        qsr = (-(dq-wetc*dter*jcool)*kappa/(np.log(hh_in[2]/10 /
               np.exp(kappa/ctn))-psit_26(hh_in[2]/monob)))
        tsrv = tsr+0.61*(T+CtoK)*qsr
        ac = np.zeros((len(spd), 3))
        charn = 0.011*np.ones(np.shape(spd))
        charn = np.where(wind > 10, 0.011+(wind-10)/(18-10)*(0.018-0.011),
                         np.where(wind > 18, 0.018, 0.011))
        ac[:, 0] = charn
    else:
        cp = 1004.67*(1 + 0.00084*qsea)
        u10n = np.copy(spd)
        monob = -100*np.ones(u10n.shape)  # Monin-Obukhov length
        monob[inan] = np.nan
        cdn = cdn_calc(u10n, Ta, None, meth)
        psim, psit, psiq = (np.zeros(u10n.shape), np.zeros(u10n.shape),
                            np.zeros(u10n.shape))
        psim[inan], psit[inan], psiq[inan] = np.nan, np.nan, np.nan
        cd = cd_calc(cdn, hh_in[0], ref_ht, psim)
        tsr, tsrv = np.zeros(u10n.shape), np.zeros(u10n.shape)
        qsr = np.zeros(u10n.shape)
        tsr[inan], tsrv[inan], qsr[inan] = np.nan, np.nan, np.nan
        if (meth == "UA"):
            wind = np.sqrt(np.power(np.copy(spd),2)+np.power(0.5,2))
            zo, zot = 0.0001*np.ones(u10n.shape), 0.0001*np.ones(u10n.shape)
            zol = hh_in[0]/monob
        elif (meth == "ERA5"):
            wind = np.sqrt(np.power(np.copy(spd),2)+np.power(0.5,2))
            usr = np.sqrt(cd*wind**2)
            Ribu = ((g*hh_in[0]*((2*(Ta-sst))/(Ta+sst-g*hh_in[0]) +
                    0.61*(qair-qsea)))/np.power(wind, 2))
            zo = 0.11*visc_air(Ta)/usr+0.018*np.power(usr, 2)/g
            zot = 0.40*visc_air(Ta)/usr
            zol = (Ribu*((np.log((hh_in[0]+zo)/zo)-psim_calc((hh_in[0]+zo) /
                   monob, meth)+psim_calc(zo/monob, meth)) /
                   (np.log((hh_in[0]+zo)/zot) -
                   psit_calc((hh_in[0]+zo)/monob, meth) +
                   psit_calc(zot/monob, meth))))
        else:
            wind = np.copy(spd)
            zo = 0.0001*np.ones(u10n.shape)
            zo[inan] = np.nan
        usr = np.sqrt(cd * wind**2)
    # tolerance for u,t,q,usr,tsr,qsr
    tol = np.array([0.01, 0.01, 5e-05, 0.005, 0.001, 5e-07])
    it = 0
    ind = np.where(spd > 0)
    ii = True
    itera = np.zeros(spd.shape)*np.nan
    while np.any(ii):
        it += 1
#        print('iter: '+str(it)+', p: '+str(np.shape(ind)[1]))
        if it > n:
            break
        if (meth == "C35"):
            old = np.array([np.copy(u10n[ind]), np.copy(usr[ind]),
                           np.copy(tsr[ind]), np.copy(qsr[ind])])
            zet = (kappa*g[ind]*hh_in[0]/(T[ind]+CtoK)*(tsr[ind]+0.61 *
                   (T[ind]+CtoK)*qsr[ind])/(usr[ind]**2))
            monob[ind] = hh_in[0]/zet
            zo[ind], cd[ind], ct[ind], cq[ind] = cd_C35(u10n[ind], wind[ind],
                                                        usr[ind], ac[ind,0],
                                                        monob[ind],
                                                        (T[ind]+CtoK), hh_in,
                                                        lat[ind])
            usr[ind] = wind[ind]*cd[ind]
            qsr[ind] = -(dq[ind]-wetc[ind]*dter[ind]*jcool)*cq[ind]
            tsr[ind] = -(dt[ind]-dter[ind]*jcool)*ct[ind]
            tsrv[ind] = tsr[ind]+0.61*(T[ind]+CtoK)*qsr[ind]
            ug[ind] = get_gust(1.2, (T[ind]+CtoK), usr[ind], tsrv[ind],
                               zi, lat[ind])  # gustiness
            wind[ind] = np.sqrt(np.copy(spd[ind])**2 + ug[ind]**2)
            gf = wind/spd
            dter[ind], dqer[ind] = get_skin(sst[ind], qsea[ind], rhoa[ind],
                                            Rl[ind], Rs[ind], Rnl[ind], cp,
                                            lv[ind], usr[ind], tsr[ind],
                                            qsr[ind], lat[ind])
            Rnl = 0.97*(5.67e-8*(SST-dter*jcool+CtoK)**4-Rl)
            if it == 1:  # save first iteration solution for case of zetu>50
                usr50, tsr50, qsr50, L50 = (usr[k50], tsr[k50], qsr[k50],
                                            monob[k50])
                dter50, dqer50 = dter[k50], dqer[k50]
            u10n[ind] = usr[ind]/kappa/gf[ind]*np.log(10/zo[ind])
            ac[ind][:] = charnock_C35(wind[ind], u10n[ind], usr[ind], seastate,
                                      waveage, wcp[ind], sigH[ind], lat[ind])
            new = np.array([np.copy(u10n[ind]), np.copy(usr[ind]),
                           np.copy(tsr[ind]), np.copy(qsr[ind])])
            d = np.abs(new-old)
            ind = np.where((d[0,:] > tol[0])+(d[1,:] > tol[3]) +
                           (d[2,:] > tol[4])+(d[3,:] > tol[5]))
            itera[ind] = np.ones(1)*it
            if np.shape(ind)[0] == 0:
               break
            else:
               ii = ((d[0,:] > tol[0])+(d[1,:] > tol[3])+(d[2,:] > tol[4]) +
                     (d[3,:] > tol[5]))
        else:
            old = np.array([np.copy(u10n[ind]), np.copy(t10n[ind]),
                           np.copy(q10n[ind]), np.copy(usr[ind]),
                           np.copy(tsr[ind]), np.copy(qsr[ind])])
            cdn[ind] = cdn_calc(u10n[ind], Ta[ind], None, meth)
            if (np.all(np.isnan(cdn))):
                break  # sys.exit("cdn cannot be nan")
                logging.debug('break %s at iteration %s cdn<0', meth, it)
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
            usr[ind] = np.sqrt(cd[ind]*wind[ind]**2)
            tsr[ind] = ct[ind]*wind[ind]*(-dt[ind])/usr[ind]
            qsr[ind] = cq[ind]*wind[ind]*(-dq[ind])/usr[ind]
            if (meth == "UA"):
                zot[ind] = 10/(np.exp(kappa**2/(ctn[ind]*np.log(10/zo[ind]))))
                zol[ind] = hh_in[0]/monob[ind]
                t10n[ind] = np.where(zol[ind]<-0.465,Ta[ind]-(tsr[ind]/kappa) *
                                     (np.log((-0.465*monob[ind])/zot[ind]) -
                                     psit_calc(-0.465,meth)+0.8 *
                                     (np.power(0.465,-1/3) -
                                     np.power(-zol[ind],-1/3))), \
                                     np.where((zol[ind]>-0.465) & (zol[ind]<0),
                                     Ta[ind]-(tsr[ind]/kappa) *
                                     (np.log(hh_in[1]/zot[ind]) -
                                     psit_calc(zol[ind],meth)), \
                                     np.where((zol[ind]>0) & (zol[ind]<1),
                                     Ta[ind]-(tsr[ind]/kappa) *
                                     (np.log(hh_in[1]/zot[ind])+5*monob[ind]),
                                     Ta[ind]-(tsr[ind]/kappa) *
                                     (np.log(monob[ind]/zot[ind])+5 +
                                     5*np.log(monob[ind])+monob[ind]-1))))
                                     # Zeng et al. 1998 (11-14)
                q10n[ind] = np.where(zol[ind]<-0.465,qair[ind] -
                                     (qsr[ind]/kappa) *
                                     (np.log((-0.465*monob[ind])/zot[ind]) -
                                     psit_calc(-0.465,meth)+0.8 *
                                     (np.power(0.465,-1/3) -
                                     np.power(-zol[ind],-1/3))), \
                                     np.where((zol[ind]>-0.465) & (zol[ind]<0),
                                     qair[ind]-(qsr[ind]/kappa) *
                                     (np.log(hh_in[1]/zot[ind]) -
                                     psit_calc(zol[ind],meth)), \
                                     np.where((zol[ind]>0) & (zol[ind]<1), \
                                     qair[ind]-(qsr[ind]/kappa) *
                                     (np.log(hh_in[1]/zot[ind])+5*monob[ind]),
                                     qair[ind]-(qsr[ind]/kappa) *
                                     (np.log(monob[ind]/zot[ind])+5 +
                                     5*np.log(monob[ind])+monob[ind]-1))))
            else:
                fact = (np.log(hh_in[1]/ref_ht)-psit[ind])/kappa
                t10n[ind] = Ta[ind] - (tsr[ind]*fact)
                fact = (np.log(hh_in[2]/ref_ht)-psiq[ind])/kappa
                q10n[ind] = qair[ind] - (qsr[ind]*fact)
            tv10n[ind] = t10n[ind]*(1+0.61*q10n[ind])
            tsrv[ind] = tsr[ind]+0.61*t10n[ind]*qsr[ind]
            if (meth == "ERA5"):
                zol[ind] = (kappa*g[ind]*hh_in[0]/Ta[ind]*(tsr[ind] +
                        0.61*Ta[ind]*qsr[ind])/(usr[ind]**2))
                monob[ind] = hh_in[0]/zol[ind]
            else:
                monob[ind] = (tv10n[ind]*usr[ind]**2)/(g[ind]*kappa*tsrv[ind])
            psim[ind] = psim_calc(hh_in[0]/monob[ind], meth)
            psit[ind] = psit_calc(hh_in[1]/monob[ind], meth)
            psiq[ind] = psit_calc(hh_in[2]/monob[ind], meth)
            if (meth == "UA"):
                u10n[ind] = np.where(zol[ind]<-1.574,(usr[ind]/kappa) *
                                     ((np.log(-1.574*monob[ind]/zo[ind]) -
                                     psim_calc(-1.574,meth))+1.14 *
                                     (np.power(-monob[ind],1/3) -
                                     np.power(1.574,1/3))), \
                                     np.where((zol[ind]>-1.574) & (zol[ind]<0),
                                     (usr[ind]/kappa)*(np.log(ref_ht/zo[ind]) -
                                     psim_calc(zol[ind])), \
                                     np.where((zol[ind]>0) & (zol[ind]<1),
                                     (usr[ind]/kappa)*(np.log(ref_ht/zo[ind]) +
                                     5*zol[ind]),(usr[ind]/kappa) *
                                     (np.log(monob[ind]/zo[ind])+5 +
                                     5*np.log(zol[ind])+zol[ind]-1))))
                                     # Zeng et al. 1998 (7-10)
                u10n[u10n<0] = np.nan
                wind[ind] = np.sqrt(np.power(np.copy(spd[ind]),2) +
                                    np.power(get_gust(1,Ta[ind], usr[ind],
                                    tsrv[ind],zi,lat[ind]),2))
                                    # Zeng et al. 1998 (20)
            elif (meth == "ERA5"):
                wind[ind] = np.sqrt(np.copy(spd[ind])**2 +
                                    (get_gust(1, Ta[ind], usr[ind], tsrv[ind],
                                    zi, lat[ind]))**2)
                u10n[ind] = (wind[ind]-(usr[ind]/kappa)*(np.log(hh_in[0] /
                             ref_ht)-psim[ind]))
                u10n[u10n < 0] = np.nan
            else:
                u10n[ind] = (wind[ind]-(usr[ind]/kappa)*(np.log(hh_in[0] /
                             ref_ht)-psim[ind]))
                u10n[u10n < 0] = np.nan  # 0.5*old[0,np.where(u10n<0)]
            new = np.array([np.copy(u10n[ind]), np.copy(t10n[ind]),
                           np.copy(q10n[ind]), np.copy(usr[ind]),
                           np.copy(tsr[ind]), np.copy(qsr[ind])])
            d = np.abs(new-old)
            ind = np.where((d[0, :] > tol[0])+(d[1, :] > tol[1]) +
                           (d[2, :] > tol[2])+(d[3, :] > tol[3]) +
                           (d[4, :] > tol[4])+(d[5, :] > tol[5]))
            itera[ind] = np.ones(1)*it
            if np.shape(ind)[0] == 0:
                break
            else:
                ii = ((d[0, ind] > tol[0])+(d[1, ind] > tol[1]) +
                      (d[2, ind] > tol[2])+(d[3, ind] > tol[3]) +
                      (d[4, ind] > tol[4])+(d[5, ind] > tol[5]))

    # calculate output parameters
    if (meth == "C35"):
        usr[k50], tsr[k50], qsr[k50], monob[k50] = usr50, tsr50, qsr50, L50
        dter[k50], dqer[k50] = dter50, dqer50
        rhoa = P*100/(287.1*(T+CtoK)*(1+0.61*qair))
        rr = zo*usr/visc_air(T+CtoK)
        zoq = np.where(5.8e-5/rr**0.72 > 1.6e-4, 1.6e-4, 5.8e-5/rr**0.72)
        zot = zoq
        psiT = psit_26(hh_in[1]/monob)
        psi10T = psit_26(10/monob)
        psi = psiu_26(hh_in[0]/monob)
        psirf = psiu_26(hh_out[0]/monob)
        q10 = qair + qsr/kappa*(np.log(10/hh_in[2])-psi10T+psiT)
        tau = rhoa*usr*usr/gf
        sensible = -rhoa*cp*usr*tsr
        latent = -rhoa*lv*usr*qsr
        cd = tau/rhoa/wind/np.where(spd < 0.1, 0.1, spd)
        cdn = 1000*kappa**2/np.log(10/zo)**2
        ct = -usr*tsr/wind/(dt-dter*jcool)
        ctn = 1000*kappa**2/np.log(10/zo)/np.log(10/zot)
        cq = -usr*qsr/(dq-dqer*jcool)/wind
        cqn = 1000*kappa**2/np.log(10/zo)/np.log(10/zoq)
        psim = psiu_26(hh_in[0]/monob)
        psit = psit_26(hh_in[1]/monob)
        u10n = u10+psiu_26(10/monob)*usr/kappa/gf
        t10n = (T+tsr/kappa*(np.log(10/hh_in[1])-psi10T+psiT) +
                tlapse*(hh_in[1]-10)+psi10T*tsr/kappa)
        q10n = q10 + psi10T*qsr/kappa
        tv10n = t10n*(1+0.61*q10n)
        urefs = spd + usr/kappa/gf*(np.log(hh_out[0]/hh_in[0])-psirf+psi)
        trefs = (T+tsr/kappa*(np.log(hh_out[1]/hh_in[1]) -
                 psit_26(hh_out[1]/monob)+psiT)+tlapse*(hh_in[1]-hh_out[1]))
        qrefs = (qair+qsr/kappa*(np.log(hh_out[2]/hh_in[2]) -
                 psit_26(hh_out[2]/monob)+psiT))
    else:
        rho = (0.34838*P)/(tv10n)
        t10n = t10n-(273.16+tlapse*ref_ht)
        sensible = -1*tsr*usr*cp*rho
        latent = -1*qsr*usr*lv*rho
        tau = 1*rho*usr**2
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
