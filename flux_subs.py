import numpy as np
from util_subs import (CtoK, kappa, gc, visc_air)

# ---------------------------------------------------------------------


def cdn_calc(u10n, Ta, Tp, lat, meth="S80"):
    """ Calculates neutral drag coefficient

    Parameters
    ----------
    u10n : float
        neutral 10m wind speed (m/s)
    Ta   : float
        air temperature (K)
    Tp   : float
        wave period
    lat : float
        latitude
    meth : str

    Returns
    -------
    cdn : float
    """
    cdn = np.zeros(Ta.shape)*np.nan
    if (meth == "S80"):
        cdn = np.where(u10n <= 3, (0.61+0.567/u10n)*0.001,
                       (0.61+0.063*u10n)*0.001)
    elif (meth == "LP82"):
        cdn = np.where((u10n < 11) & (u10n >= 4), 1.2*0.001,
                       np.where((u10n <= 25) & (u10n >= 11),
                       (0.49+0.065*u10n)*0.001, 1.14*0.001))
    elif (meth == "S88" or meth == "UA" or meth == "ERA5" or meth == "C30" or
          meth == "C35" or meth == "C40" or meth == "Beljaars"):
        cdn = cdn_from_roughness(u10n, Ta, None, lat, meth)
    elif (meth == "YT96"):
        # for u<3 same as S80
        cdn = np.where((u10n < 6) & (u10n >= 3),
                       (0.29+3.1/u10n+7.7/u10n**2)*0.001,
                       np.where((u10n <= 26) & (u10n >= 6),
                       (0.60 + 0.070*u10n)*0.001, (0.61+0.567/u10n)*0.001))
    elif (meth == "LY04"):
        cdn = np.where(u10n >= 0.5,
                       (0.142+(2.7/u10n)+(u10n/13.09))*0.001,
                       (0.142+(2.7/0.5)+(0.5/13.09))*0.001)
    else:
        print("unknown method cdn: "+meth)
    return cdn
# ---------------------------------------------------------------------


def cdn_from_roughness(u10n, Ta, Tp, lat, meth="S88"):
    """ Calculates neutral drag coefficient from roughness length

    Parameters
    ----------
    u10n : float
        neutral 10m wind speed (m/s)
    Ta   : float
        air temperature (K)
    Tp   : float
        wave period
    lat : float
        latitude
    meth : str

    Returns
    -------
    cdn : float
    """
    g, tol = gc(lat, None), 0.000001
    cdn, usr = np.zeros(Ta.shape), np.zeros(Ta.shape)
    cdnn = (0.61+0.063*u10n)*0.001
    zo, zc, zs = np.zeros(Ta.shape), np.zeros(Ta.shape), np.zeros(Ta.shape)
    for it in range(5):
        cdn = np.copy(cdnn)
        usr = np.sqrt(cdn*u10n**2)
        if (meth == "S88"):
            # Charnock roughness length (eq. 4 in Smith 88)
            zc = 0.011*np.power(usr, 2)/g
            #  smooth surface roughness length (eq. 6 in Smith 88)
            zs = 0.11*visc_air(Ta)/usr
            zo = zc + zs  #  eq. 7 & 8 in Smith 88
        elif (meth == "UA"):
            # valid for 0<u<18m/s # Zeng et al. 1998 (24)
            zo = 0.013*np.power(usr, 2)/g+0.11*visc_air(Ta)/usr
        elif (meth == "C30"):
            a = 0.011*np.ones(Ta.shape)
            a = np.where(u10n > 10, 0.011+(u10n-10)*(0.018-0.011)/(18-10),
                         np.where(u10n > 18, 0.018, a))
            zo = a*np.power(usr, 2)/g+0.11*visc_air(Ta)/usr
        elif (meth == "C35"):
            a = 0.011*np.ones(Ta.shape)
            # a = np.where(u10n > 19, 0.0017*19-0.0050,
            #             np.where((u10n > 7) & (u10n <= 18),
            #                       0.0017*u10n-0.0050, a))
            a = np.where(u10n > 19, 0.0017*19-0.0050, 0.0017*u10n-0.0050)
            zo = 0.11*visc_air(Ta)/usr+a*np.power(usr, 2)/g
        elif (meth == "C40"):
            a = 0.011*np.ones(Ta.shape)
            a = np.where(u10n > 22, 0.0016*22-0.0035, 0.0016*u10n-0.0035)
            zo = a*np.power(usr, 2)/g+0.11*visc_air(Ta)/usr # surface roughness
        elif ((meth == "ERA5" or meth == "Beljaars")):
            # eq. (3.26) p.38 over sea IFS Documentation cy46r1
            zo = 0.018*np.power(usr, 2)/g+0.11*visc_air(Ta)/usr
        else:
            print("unknown method for cdn_from_roughness "+meth)
        cdnn = (kappa/np.log(10/zo))**2
    cdn = np.where(np.abs(cdnn-cdn) < tol, cdnn, np.nan)
    return cdn
# ---------------------------------------------------------------------


def cd_calc(cdn, height, ref_ht, psim):
    """ Calculates drag coefficient at reference height

    Parameters
    ----------
    cdn : float
        neutral drag coefficient
    height : float
        original sensor height (m)
    ref_ht : float
        reference height (m)
    psim : float
        momentum stability function

    Returns
    -------
    cd : float
    """
    cd = (cdn/np.power(1+(np.sqrt(cdn)*(np.log(height/ref_ht)-psim))/kappa, 2))
    return cd
# ---------------------------------------------------------------------


def ctcqn_calc(zol, cdn, u10n, zo, Ta, meth="S80"):
    """ Calculates neutral heat and moisture exchange coefficients

    Parameters
    ----------
    zol  : float
        height over MO length
    cdn  : float
        neutral drag coefficient
    u10n : float
        neutral 10m wind speed (m/s)
    zo   : float
        surface roughness (m)
    Ta   : float
        air temperature (K)
    meth : str

    Returns
    -------
    ctn : float
        neutral heat exchange coefficient
    cqn : float
        neutral moisture exchange coefficient
    """
    if (meth == "S80" or meth == "S88" or meth == "YT96"):
        cqn = np.ones(Ta.shape)*1.20*0.001  # from S88
        ctn = np.ones(Ta.shape)*1.00*0.001
    elif (meth == "LP82"):
        cqn = np.where((zol <= 0) & (u10n > 4) & (u10n < 14), 1.15*0.001,
                       1*0.001)
        ctn = np.where((zol <= 0) & (u10n > 4) & (u10n < 25), 1.13*0.001,
                       0.66*0.001)
    elif (meth == "LY04"):
        cqn = 34.6*0.001*np.sqrt(cdn)
        ctn = np.where(zol <= 0, 32.7*0.001*np.sqrt(cdn), 18*0.001*np.sqrt(cdn))
    elif (meth == "UA"):
        usr = np.sqrt(cdn*np.power(u10n, 2))
        # Zeng et al. 1998 (25)
        re=usr*zo/visc_air(Ta)
        zoq = zo/np.exp(2.67*np.power(re, 1/4)-2.57)
        zot = zoq
        cqn = np.where((u10n > 0.5) & (u10n < 18), np.power(kappa, 2) /
                       (np.log(10/zo)*np.log(10/zoq)), np.nan)
        ctn = np.where((u10n > 0.5) & (u10n < 18), np.power(kappa, 2) /
                       (np.log(10/zo)*np.log(10/zoq)), np.nan)
    elif (meth == "C30"):
        usr = np.sqrt(cdn*np.power(u10n, 2))
        rr = zo*usr/visc_air(Ta)
        zoq = np.where(5e-5/np.power(rr, 0.6) > 1.15e-4, 1.15e-4,
                       5e-5/np.power(rr, 0.6))  # moisture roughness
        zot=zoq  # temperature roughness
        cqn = kappa**2/np.log(10/zo)/np.log(10/zoq)
        ctn = kappa**2/np.log(10/zo)/np.log(10/zot)
    elif (meth == "C35"):
        usr = np.sqrt(cdn*np.power(u10n, 2))
        rr = zo*usr/visc_air(Ta)
        zoq = np.where(5.8e-5/np.power(rr, 0.72) > 1.6e-4, 1.6e-4,
                       5.8e-5/np.power(rr, 0.72))  # moisture roughness
        zot=zoq  # temperature roughness
        cqn = kappa**2/np.log(10/zo)/np.log(10/zoq)
        ctn = kappa**2/np.log(10/zo)/np.log(10/zot)
    elif (meth == "C40"):
        usr = np.sqrt(cdn*np.power(u10n, 2))
        rr = zo*usr/visc_air(Ta)
        zot = np.where(1.0e-4/np.power(rr, 0.55) > 2.4e-4/np.power(rr, 1.2),
                       2.4e-4/np.power(rr, 1.2),
                       1.0e-4/np.power(rr, 0.55)) # temperature roughness
        zoq = np.where(2.0e-5/np.power(rr,0.22) > 1.1e-4/np.power(rr,0.9),
                       1.1e-4/np.power(rr,0.9), 2.0e-5/np.power(rr,0.22))
        # moisture roughness determined by the CLIMODE, GASEX and CBLAST data
#        zoq = np.where(5e-5/np.power(rr, 0.6) > 1.15e-4, 1.15e-4,
#                       5e-5/np.power(rr, 0.6))  # moisture roughness as in C30
        cqn = kappa**2/np.log(10/zo)/np.log(10/zoq)
        ctn = kappa**2/np.log(10/zo)/np.log(10/zot)
    elif (meth == "ERA5" or meth == "Beljaars"):
        # eq. (3.26) p.38 over sea IFS Documentation cy46r1
        usr = np.sqrt(cdn*np.power(u10n, 2))
        zot = 0.40*visc_air(Ta)/usr
        zoq = 0.62*visc_air(Ta)/usr
        cqn = kappa**2/np.log(10/zo)/np.log(10/zoq)
        ctn = kappa**2/np.log(10/zo)/np.log(10/zot)
    else:
        print("unknown method ctcqn: "+meth)
    return ctn, cqn
# ---------------------------------------------------------------------


def ctcq_calc(cdn, cd, ctn, cqn, ht, hq, ref_ht, psit, psiq):
    """ Calculates heat and moisture exchange coefficients at reference height

    Parameters
    ----------
    cdn : float
        neutral drag coefficient
    cd  : float
        drag coefficient at reference height
    ctn : float
        neutral heat exchange coefficient
    cqn : float
        neutral moisture exchange coefficient
    h_t : float
        original temperature sensor height (m)
    h_q : float
        original moisture sensor height (m)
    ref_ht : float
        reference height (m)
    psit : float
        heat stability function
    psiq : float
        moisture stability function

    Returns
    -------
    ct : float
       heat exchange coefficient
    cq : float
       moisture exchange coefficient
    """
    ct = (ctn*np.sqrt(cd/cdn) /
          (1+ctn*((np.log(ht/ref_ht)-psit)/(kappa*np.sqrt(cdn)))))
    cq = (cqn*np.sqrt(cd/cdn) /
          (1+cqn*((np.log(hq/ref_ht)-psiq)/(kappa*np.sqrt(cdn)))))
    return ct, cq
# ---------------------------------------------------------------------


def get_stabco(meth="S80"):
    """ Gives the coefficients \\alpha, \\beta, \\gamma for stability functions

    Parameters
    ----------
    meth : str

    Returns
    -------
    coeffs : float
    """
    alpha, beta, gamma = 0, 0, 0
    if (meth == "S80" or meth == "S88" or meth == "LY04" or
        meth == "UA" or meth == "ERA5" or meth == "C30" or
        meth == "C35" or meth == "C40" or meth == "Beljaars"):
        alpha, beta, gamma = 16, 0.25, 5  # Smith 1980, from Dyer (1974)
    elif (meth == "LP82"):
        alpha, beta, gamma = 16, 0.25, 7
    elif (meth == "YT96"):
        alpha, beta, gamma = 20, 0.25, 5
    else:
        print("unknown method stabco: "+meth)
    coeffs = np.zeros(3)
    coeffs[0] = alpha
    coeffs[1] = beta
    coeffs[2] = gamma
    return coeffs
# ---------------------------------------------------------------------


def psim_calc(zol, meth="S80"):
    """ Calculates momentum stability function

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str

    Returns
    -------
    psim : float
    """
    if (meth == "ERA5"):
        psim = psim_era5(zol)
    elif (meth == "C30" or meth == "C35" or meth == "C40"):
        psim = psiu_26(zol, meth)
    elif (meth == "Beljaars"): # Beljaars (1997) eq. 16, 17
        psim = np.where(zol < 0, psim_conv(zol, meth), psi_Bel(zol))
    else:
        psim = np.where(zol < 0, psim_conv(zol, meth),
                        psim_stab(zol, meth))
    return psim
# ---------------------------------------------------------------------


def psit_calc(zol, meth="S80"):
    """ Calculates heat stability function

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str
        parameterisation method

    Returns
    -------
    psit : float
    """
    if (meth == "ERA5"):
        psit = np.where(zol < 0, psi_conv(zol, meth),
                        psi_era5(zol))
    elif (meth == "C30" or meth == "C35" or meth == "C40"):
        psit = psit_26(zol)
    elif (meth == "Beljaars"): # Beljaars (1997) eq. 16, 17
        psit = np.where(zol < 0, psi_conv(zol, meth), psi_Bel(zol))
    else:
        psit = np.where(zol < 0, psi_conv(zol, meth),
                        psi_stab(zol, meth))
    return psit
# ---------------------------------------------------------------------


def psi_Bel(zol):
    """ Calculates momentum/heat stability function

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str
        parameterisation method

    Returns
    -------
    psit : float
    """
    a, b, c, d = 0.7, 0.75, 5, 0.35
    psi = -(a*zol+b*(zol-c/d)*np.exp(-d*zol)+b*c/d)
    return psi
# ---------------------------------------------------------------------


def psi_era5(zol):
    """ Calculates heat stability function for stable conditions
        for method ERA5

    Parameters
    ----------
    zol : float
        height over MO length

    Returns
    -------
    psit : float
    """
    # eq (3.22) p. 37 IFS Documentation cy46r1
    a, b, c, d = 1, 2/3, 5, 0.35
    psit = -b*(zol-c/d)*np.exp(-d*zol)-np.power(1+(2/3)*a*zol, 1.5)-(b*c)/d+1
    return psit
# ---------------------------------------------------------------------


def psit_26(zol):
    """ Computes temperature structure function as in C35

    Parameters
    ----------
    zol : float
        height over MO length

    Returns
    -------
    psi : float
    """
    b, d = 2/3, 0.35
    dzol = np.where(d*zol > 50, 50, d*zol)
    psi = np.where(zol > 0,-(np.power(1+b*zol, 1.5)+b*(zol-14.28) *
                             np.exp(-dzol)+8.525), np.nan)
    psik = np.where(zol < 0, 2*np.log((1+np.sqrt(1-15*zol))/2), np.nan)
    psic = np.where(zol < 0, 1.5*np.log((1+np.power(1-34.15*zol, 1/3) +
                    np.power(1-34.15*zol, 2/3))/3)-np.sqrt(3) *
                    np.arctan(1+2*np.power(1-34.15*zol, 1/3))/np.sqrt(3) +
                    4*np.arctan(1)/np.sqrt(3), np.nan)
    f = np.power(zol, 2)/(1+np.power(zol, 2))
    psi = np.where(zol < 0, (1-f)*psik+f*psic, psi)
    return psi
# ---------------------------------------------------------------------


def psi_conv(zol, meth):
    """ Calculates heat stability function for unstable conditions

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str
        parameterisation method

    Returns
    -------
    psit : float
    """
    coeffs = get_stabco(meth)
    alpha, beta = coeffs[0], coeffs[1]
    xtmp = np.power(1-alpha*zol, beta)
    psit = 2*np.log((1+np.power(xtmp, 2))*0.5)
    return psit
# ---------------------------------------------------------------------


def psi_stab(zol, meth):
    """ Calculates heat stability function for stable conditions

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str
        parameterisation method

    Returns
    -------
    psit : float
    """
    coeffs = get_stabco(meth)
    gamma = coeffs[2]
    psit = -gamma*zol
    return psit
# ---------------------------------------------------------------------


def psim_era5(zol):
    """ Calculates momentum stability function for method ERA5

    Parameters
    ----------
    zol : float
        height over MO length

    Returns
    -------
    psim : float
    """
    # eq (3.20, 3.22) p. 37 IFS Documentation cy46r1
    coeffs = get_stabco("ERA5")
    alpha, beta = coeffs[0], coeffs[1]
    xtmp = np.power(1-alpha*zol, beta)
    a, b, c, d = 1, 2/3, 5, 0.35
    psim = np.where(zol < 0, np.pi/2-2*np.arctan(xtmp) +
                    np.log((np.power(1+xtmp, 2)*(1+np.power(xtmp, 2)))/8),
                    -b*(zol-c/d)*np.exp(-d*zol)-a*zol-(b*c)/d)
    return psim
# ---------------------------------------------------------------------


def psiu_26(zol, meth):
    """ Computes velocity structure function C35

    Parameters
    ----------
    zol : float
        height over MO length

    Returns
    -------
    psi : float
    """
    if (meth == "C30"):
        dzol = np.where(0.35*zol > 50, 50, 0.35*zol) # stable
        psi = np.where(zol > 0, -((1+zol)+0.6667*(zol-14.28)*np.exp(-dzol) +
                                  8.525), np.nan)
        x = np.where(zol < 0, np.power(1-15*zol, 0.25), np.nan)
        psik = np.where(zol < 0, 2*np.log((1+x)/2)+np.log((1+np.power(x, 2)) /
                        2)-2*np.arctan(x)+2*np.arctan(1), np.nan)
        x = np.where(zol < 0, np.power(1-10.15*zol, 0.3333), np.nan)
        psic = np.where(zol < 0, 1.5*np.log((1+x+np.power(x, 2))/3) -
                        np.sqrt(3)*np.arctan((1+2*x)/np.sqrt(3)) +
                        4*np.arctan(1)/np.sqrt(3), np.nan)
        f = np.power(zol, 2)/(1+np.power(zol, 2))
        psi = np.where(zol < 0, (1-f)*psik+f*psic, psi)
    elif (meth == "C35" or meth == "C40"):
        dzol = np.where(0.35*zol > 50, 50, 0.35*zol)  # stable
        a, b, c, d = 0.7, 3/4, 5, 0.35
        psi = np.where(zol > 0, -(a*zol+b*(zol-c/d)*np.exp(-dzol)+b*c/d),
                       np.nan)
        x = np.where(zol < 0, np.power(1-15*zol, 0.25), np.nan)
        psik = np.where(zol < 0, 2*np.log((1+x)/2)+np.log((1+x**2)/2) -
                        2*np.arctan(x)+2*np.arctan(1), np.nan)
        x = np.where(zol < 0, np.power(1-10.15*zol, 0.3333), np.nan)
        psic = np.where(zol < 0, 1.5*np.log((1+x+np.power(x, 2))/3) -
                        np.sqrt(3)*np.arctan((1+2*x)/np.sqrt(3)) +
                        4*np.arctan(1)/np.sqrt(3), np.nan)
        f = np.power(zol, 2)/(1+np.power(zol, 2))
        psi = np.where(zol < 0, (1-f)*psik+f*psic, psi)
    return psi
#------------------------------------------------------------------------------



def psim_conv(zol, meth):
    """ Calculates momentum stability function for unstable conditions

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str
        parameterisation method

    Returns
    -------
    psim : float
    """
    coeffs = get_stabco(meth)
    alpha, beta = coeffs[0], coeffs[1]
    xtmp = np.power(1-alpha*zol, beta)
    psim = (2*np.log((1+xtmp)*0.5)+np.log((1+np.power(xtmp, 2))*0.5) -
            2*np.arctan(xtmp)+np.pi/2)
    return psim
# ---------------------------------------------------------------------


def psim_stab(zol, meth):
    """ Calculates momentum stability function for stable conditions

    Parameters
    ----------
    zol : float
        height over MO length
    meth : str
        parameterisation method

    Returns
    -------
    psim : float
    """
    coeffs = get_stabco(meth)
    gamma = coeffs[2]
    psim = -gamma*zol
    return psim
# ---------------------------------------------------------------------


def get_skin(sst, qsea, rho, Rl, Rs, Rnl, cp, lv, tkt, usr, tsr, qsr, lat):
    """ Computes cool skin

    Parameters
    ----------
    sst : float
        sea surface temperature ($^\circ$\,C)
    qsea : float
        specific humidity over sea (g/kg)
    rho : float
        density of air (kg/m^3)
    Rl : float
        downward longwave radiation (W/m^2)
    Rs : float
        downward shortwave radiation (W/m^2)
    Rnl : float
        upwelling IR radiation (W/m^2)
    cp : float
       specific heat of air at constant pressure
    lv : float
       latent heat of vaporization
    tkt : float
       cool skin thickness
    usr : float
       friction velocity
    tsr : float
       star temperature
    qsr : float
       star humidity
    lat : float
       latitude

    Returns
    -------
    dter : float
       cool-skin temperature depression
    dqer : float
       cool-skin humidity depression
    tkt  : float
       cool skin thickness
    """
    # coded following Saunders (1967) with lambda = 6
    g = gc(lat, None)
    if (np.nanmin(sst) > 200):  # if sst in Kelvin convert to Celsius
        sst = sst-CtoK
    # ************  cool skin constants  *******
    # density of water, specific heat capacity of water, water viscosity,
    # thermal conductivity of water
    rhow, cpw, visw, tcw = 1022, 4000, 1e-6, 0.6
    Al = 2.1e-5*np.power(sst+3.2, 0.79)
    be = 0.026
    bigc = 16*g*cpw*np.power(rhow*visw, 3)/(np.power(tcw, 2)*np.power(rho, 2))
    wetc = 0.622*lv*qsea/(287.1*np.power(sst+273.16, 2))
    Rns = 0.945*Rs  # albedo correction
    hsb = -rho*cp*usr*tsr
    hlb = -rho*lv*usr*qsr
    qout = Rnl+hsb+hlb
    dels = Rns*(0.065+11*tkt-6.6e-5/tkt*(1-np.exp(-tkt/8.0e-4)))
    qcol = qout-dels
    alq = Al*qcol+be*hlb*cpw/lv
    xlamx = 6*np.ones(sst.shape)
    xlamx = np.where(alq > 0, 6/(1+(bigc*alq/usr**4)**0.75)**0.333, 6)
    tkt = np.where(alq > 0, xlamx*visw/(np.sqrt(rho/rhow)*usr),
                   np.where(xlamx*visw/(np.sqrt(rho/rhow)*usr) > 0.01, 0.01,
                   xlamx*visw/(np.sqrt(rho/rhow)*usr)))
    dter = qcol*tkt/tcw
    dqer = wetc*dter
    return dter, dqer, tkt
# ---------------------------------------------------------------------


def get_gust(beta, Ta, usr, tsrv, zi, lat):
    """ Computes gustiness

    Parameters
    ----------
    beta : float
        constant
    Ta : float
        air temperature (K)
    usr : float
        friction velocity (m/s)
    tsrv : float
        star virtual temperature of air (K)
    zi : int
        scale height of the boundary layer depth (m)
    lat : float
        latitude

    Returns
    -------
    ug : float
    """
    if (np.nanmax(Ta) < 200):  # convert to K if in Celsius
        Ta = Ta+273.16
    g = gc(lat, None)
    Bf = (-g/Ta)*usr*tsrv
    ug = np.ones(np.shape(Ta))*0.2
    ug = np.where(Bf > 0, beta*np.power(Bf*zi, 1/3), 0.2)
    return ug
# ---------------------------------------------------------------------


def get_L(L, lat, usr, tsr, qsr, t10n, tv10n, qair, h_in, T, Ta, th, tv, sst,
          dt, dtv, dq, zo, wind, monob, meth):
    """
    calculates Monin-Obukhov length and virtual star temperature

    Parameters
    ----------
    L : int
        Monin-Obukhov length definition options
           "S80"  : default for S80, S88, LP82, YT96 and LY04
           "ERA5" : following ERA5 (IFS Documentation cy46r1), default for ERA5
    lat : float
        latitude
    usr : float
        friction wind speed (m/s)
    tsr : float
        star temperature (K)
    qsr : float
        star specific humidity (g/kg)
    t10n : float
        neutral temperature at 10m (K)
    tv10n : float
        neutral virtual temperature at 10m (K)
    qair : float
        air specific humidity (g/kg)
    h_in : float
        sensor heights (m)
    T : float
        air temperature (K)
    Ta : float
        air temperature (K)
    th : float
        potential temperature (K)
    tv : float
        virtual temperature (K)
    sst : float
        sea surface temperature (K)
    dt : float
        temperature difference (K)
    dq : float
        specific humidity difference (g/kg)
    wind : float
        wind speed (m/s)
    monob : float
        Monin-Obukhov length from previous iteration step (m)
    meth : str
        bulk parameterisation method option: "S80", "S88", "LP82", "YT96",
        "UA", "LY04", "C30", "C35", "C40", "ERA5"

    Returns
    -------
    tsrv : float
        virtual star temperature (K)
    monob : float
        M-O length (m)

    """
    g = gc(lat)
    if (L == "S80"):
        tsrv = tsr+0.61*t10n*qsr
        monob = ((tv10n*np.power(usr, 2))/(g*kappa*tsrv))
        monob = np.where(np.fabs(monob) < 1, np.where(monob < 0, -1, 1), monob)
    elif (L == "ERA5"):
        tsrv = tsr+0.61*t10n*qsr
        Rb = ((g*h_in[0]*((2*dt)/(Ta+sst-g*h_in[0])+0.61*dq)) /
              np.power(wind, 2))
        zo = (0.11*visc_air(Ta)/usr+0.018*np.power(usr, 2)/g)
        zot = 0.40*visc_air(Ta)/usr
        zol = (Rb*(np.power(np.log((h_in[0]+zo)/zo)-psim_calc((h_in[0]+zo) /
                                                              monob, meth) +
                            psim_calc(zo/monob, meth), 2) /
                   (np.log((h_in[0]+zo)/zot) -
                    psit_calc((h_in[0]+zo)/monob, meth) +
                    psit_calc(zot/monob, meth))))
        monob = h_in[0]/zol        
    return tsrv, monob
#------------------------------------------------------------------------------


def get_strs(h_in, monob, wind, zo, zot, zoq, dt, dq, dter, dqer, ct, cq,
             cskin, meth):
    """
    calculates star wind speed, temperature and specific humidity

    Parameters
    ----------
    h_in : float
        sensor heights (m)
    monob : float
        M-O length (m)
    wind : float
        wind speed (m/s)
    zo : float
        momentum roughness length (m)
    zot : float
        temperature roughness length (m)
    zoq : float
        moisture roughness length (m)
    dt : float
        temperature difference (K)
    dq : float
        specific humidity difference (g/kg)
    dter : float
        cskin temperature adjustment (K)
    dqer : float
        cskin q adjustment (g/kg)
    ct : float
        temperature exchange coefficient
    cq : float
        moisture exchange coefficient
    cskin : int
        cool skin adjustment switch
    meth : str
        bulk parameterisation method option: "S80", "S88", "LP82", "YT96", "UA",
        "LY04", "C30", "C35", "C40", "ERA5"

    Returns
    -------
    usr : float
        friction wind speed (m/s)
    tsr : float
        star temperature (K)
    qsr : float
        star specific humidity (g/kg)

    """
    if (meth == "UA"):
        usr = np.where(h_in[0]/monob <= -1.574, kappa*wind /
                       (np.log(-1.574*monob/zo)-psim_calc(-1.574, meth) +
                        psim_calc(zo/monob, meth) +
                        1.14*(np.power(-h_in[0]/monob, 1/3) -
                        np.power(1.574, 1/3))),
                       np.where(h_in[0]/monob < 0, kappa*wind /
                                (np.log(h_in[0]/zo) -
                                 psim_calc(h_in[0]/monob, meth) +
                                 psim_calc(zo/monob, meth)),
                                np.where(h_in[0]/monob <= 1, kappa*wind /
                                         (np.log(h_in[0]/zo) +
                                          5*h_in[0]/monob-5*zo/monob),
                                         kappa*wind/(np.log(monob/zo)+5 -
                                                     5*zo/monob +
                                                     5*np.log(h_in[0]/monob) +
                                                     h_in[0]/monob-1))))
                                # Zeng et al. 1998 (7-10)
        tsr = np.where(h_in[1]/monob < -0.465, kappa*(dt+dter*cskin) /
                       (np.log((-0.465*monob)/zot) -
                        psit_calc(-0.465, meth)+0.8*(np.power(0.465, -1/3) -
                        np.power(-h_in[1]/monob, -1/3))),
                       np.where(h_in[1]/monob < 0, kappa*(dt+dter*cskin) /
                                (np.log(h_in[1]/zot) -
                                 psit_calc(h_in[1]/monob, meth) +
                                 psit_calc(zot/monob, meth)),
                                np.where(h_in[1]/monob <= 1,
                                         kappa*(dt+dter*cskin) /
                                         (np.log(h_in[1]/zot) +
                                          5*h_in[1]/monob-5*zot/monob),
                                         kappa*(dt+dter*cskin) /
                                         (np.log(monob/zot)+5 -
                                          5*zot/monob+5*np.log(h_in[1]/monob) +
                                          h_in[1]/monob-1))))
                                # Zeng et al. 1998 (11-14)
        qsr = np.where(h_in[2]/monob < -0.465, kappa*(dq+dqer*cskin) /
                       (np.log((-0.465*monob)/zoq) -
                        psit_calc(-0.465, meth)+psit_calc(zoq/monob, meth) +
                        0.8*(np.power(0.465, -1/3) -
                             np.power(-h_in[2]/monob, -1/3))),
                       np.where(h_in[2]/monob < 0,
                                kappa*(dq+dqer*cskin)/(np.log(h_in[1]/zot) -
                                psit_calc(h_in[2]/monob, meth) +
                                psit_calc(zoq/monob, meth)),
                                np.where(h_in[2]/monob <= 1,
                                         kappa*(dq+dqer*cskin) /
                                         (np.log(h_in[1]/zoq)+5*h_in[2]/monob -
                                          5*zoq/monob),
                                         kappa*(dq+dqer*cskin)/
                                         (np.log(monob/zoq)+5-5*zoq/monob +
                                          5*np.log(h_in[2]/monob) +
                                          h_in[2]/monob-1))))
    elif (meth == "C30" or meth == "C35" or meth == "C40"):
        usr = (wind*kappa/(np.log(h_in[0]/zo)-psiu_26(h_in[0]/monob, meth)))
        tsr = ((dt+dter*cskin)*(kappa/(np.log(h_in[1]/zot) -
                                       psit_26(h_in[1]/monob))))
        qsr = ((dq+dqer*cskin)*(kappa/(np.log(h_in[2]/zoq) -
                                       psit_26(h_in[2]/monob))))
    else:
        usr = (wind*kappa/(np.log(h_in[0]/zo)-psim_calc(h_in[0]/monob, meth)))
        tsr = ct*wind*(dt+dter*cskin)/usr
        qsr = cq*wind*(dq+dqer*cskin)/usr
    return usr, tsr, qsr
# ---------------------------------------------------------------------
