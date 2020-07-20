import numpy as np
import sys
from VaporPressure import VaporPressure

CtoK = 273.16  # 273.15
""" Conversion factor for $^\circ\,$C to K """

kappa = 0.4  # NOTE: 0.41
""" von Karman's constant """
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
          meth == "C35" or meth == "C40"):
        cdn = cdn_from_roughness(u10n, Ta, None, lat, meth)
    elif (meth == "YT96"):
        # for u<3 same as S80
        cdn = np.where((u10n < 6) & (u10n >= 3),
                       (0.29+3.1/u10n+7.7/u10n**2)*0.001,
                       np.where((u10n <= 26) & (u10n >= 6),
                       (0.60 + 0.070*u10n)*0.001, (0.61+0.567/u10n)*0.001))
    elif (meth == "LY04"):
        cdn = np.where(u10n >= 0.5,
                       (0.142+(2.7/u10n)+(u10n/13.09))*0.001, np.nan)
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
            a = np.where(u10n > 10, 0.011+(u10n-10)/(18-10)*(0.018-0.011),
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
        elif (meth == "ERA5"):
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
    elif (meth == "ERA5"):
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
        meth == "UA" or meth == "ERA5" or meth == "C30" or meth == "C35" or
        meth == "C40"):
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
    else:
        psit = np.where(zol < 0, psi_conv(zol, meth),
                        psi_stab(zol, meth))
    return psit
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
# ---------------------------------------------------------------------


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


def get_init(spd, T, SST, lat, P, Rl, Rs, cskin, gust, L, tol, meth, qmeth):
    """
    Checks initial input values and sets defaults if needed

    Parameters
    ----------
    spd : float
        relative wind speed in m/s (is assumed as magnitude difference
        between wind and surface current vectors)
    T : float
        air temperature in K
    SST : float
        sea surface temperature in K
    lat : float
        latitude (deg), default 45deg
    P : float
        air pressure (hPa), default 1013hPa
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
    L : int
        Monin-Obukhov length definition options
        0 : default for S80, S88, LP82, YT96 and LY04
        1 : following UA (Zeng et al., 1998), default for UA
        2 : following ERA5 (IFS Documentation cy46r1), default for ERA5
        3 : COARE3.5 (Edson et al., 2013), default for C30, C35 and C40
    tol : float
        4x1 or 7x1 [option, lim1-3 or lim1-6]
        option : 'flux' to set tolerance limits for fluxes only lim1-3
        option : 'ref' to set tolerance limits for height adjustment lim-1-3
        option : 'all' to set tolerance limits for both fluxes and height
                 adjustment lim1-6 ['all', 0.01, 0.01, 5e-05, 0.01, 1, 1]
    meth : str
        "S80","S88","LP82","YT96","UA","LY04","C30","C35","C40","ERA5"
    qmeth : str
        is the saturation evaporation method to use amongst
        "HylandWexler","Hardy","Preining","Wexler","GoffGratch","CIMO",
        "MagnusTetens","Buck","Buck2","WMO","WMO2000","Sonntag","Bolton",
        "IAPWS","MurphyKoop"]
        default is Buck2

    Returns
    -------
    lat : float
        latitude
    P : float
        air pressure (hPa)
    Rl : float
        downward longwave radiation (W/m^2)
    Rs : float
        downward shortwave radiation (W/m^2)
    cskin : int
        cool skin adjustment switch
    gust : int
        gustiness switch
    tol : float
        tolerance limits
    L : int
        MO length switch

    """
    if ((type(spd) != np.ndarray) or (type(T) != np.ndarray) or
         (type(SST) != np.ndarray)):
        sys.exit("input type of spd, T and SST should be numpy.ndarray")
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
    if (np.all(lat == None)):  # set latitude to 45deg if empty
        lat = 45*np.ones(spd.shape)
    elif ((np.all(lat != None)) and (np.size(lat) == 1)):
        lat = np.ones(spd.shape)*np.copy(lat)
    if ((np.all(P == None)) or np.all(np.isnan(P))):
        P = np.ones(spd.shape)*1013
    elif (((np.all(P != None)) or np.all(~np.isnan(P))) and np.size(P) == 1):
        P = np.ones(spd.shape)*np.copy(P)
    if (np.all(Rl == None) or np.all(np.isnan(Rl))):
        Rl = np.ones(spd.shape)*370    # set to default for COARE3.5
    if (np.all(Rs == None) or np.all(np.isnan(Rs))):
        Rs = np.ones(spd.shape)*150  # set to default for COARE3.5
    if ((cskin == None) and (meth == "S80" or meth == "S88" or meth == "LP82"
                              or meth == "YT96")):
        cskin = 0
    elif ((cskin == None) and (meth == "UA" or meth == "LY04" or meth == "C30"
                                or meth == "C35" or meth == "C40"
                                or meth == "ERA5")):
        cskin = 1
    if ((gust == None) and (meth == "C30" or meth == "C35" or meth == "C40")):
        gust = [1, 1.2, 600]
    elif ((gust == None) and (meth == "UA" or meth == "ERA5")):
        gust = [1, 1, 1000]
    elif (gust == None):
        gust = [1, 1.2, 800]
    elif (np.size(gust) < 3):
        sys.exit("gust input must be a 3x1 array")
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
    if (tol == None):
        tol = ['flux', 0.01, 1, 1]
    elif (tol[0] not in ['flux', 'ref', 'all']):
        sys.exit("unknown tolerance input")
    return lat, P, Rl, Rs, cskin, gust, tol, L
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
    dqer : float

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
          dt, dq, wind, monob, meth):
    """
    calculates Monin-Obukhov length and virtual star temperature

    Parameters
    ----------
    L : int
        Monin-Obukhov length definition options
           0 : default for S80, S88, LP82, YT96 and LY04
           1 : following UA (Zeng et al., 1998), default for UA
           2 : following ERA5 (IFS Documentation cy46r1), default for ERA5
           3 : COARE3.5 (Edson et al., 2013), default for C30, C35 and C40
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
        bulk parameterisation method option: "S80", "S88", "LP82", "YT96", "UA",
        "LY04", "C30", "C35", "C40", "ERA5"

    Returns
    -------
    tsrv : float
        virtual star temperature (K)
    monob : float
        M-O length (m)

    """
    g = gc(lat)
    if (L == 0):
        tsrv = tsr+0.61*t10n*qsr
        monob = ((tv10n*np.power(usr, 2))/(g*kappa*tsrv))
        monob = np.where(np.fabs(monob) < 1, np.where(monob < 0, -1, 1), monob)
    elif (L == 1):
        tsrv = tsr*(1.+0.61*qair)+0.61*th*qsr
        monob = ((tv*np.power(usr, 2))/(kappa*g*tsrv))
    elif (L == 2):
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
    elif (L == 3):
        tsrv = tsr+0.61*(T+CtoK)*qsr
        zol = (kappa*g*h_in[0]/(T+CtoK)*(tsr+0.61*(T+CtoK)*qsr) /
               np.power(usr, 2))
        monob = h_in[0]/zol
    return tsrv, monob
#------------------------------------------------------------------------------


def get_hum(hum, T, sst, P, qmeth):
    """
    Get specific humidity output

    Parameters
    ----------
    hum : array
        humidity input switch 2x1 [x, values] default is relative humidity
            x='rh' : relative humidity in %
            x='q' : specific humidity (g/kg)
            x='Td' : dew point temperature (K)
    T : float
        air temperature in K
    sst : float
        sea surface temperature in K
    P : float
        air pressure at sea level in hPa
    qmeth : str
        method to calculate specific humidity from vapor pressure

    Returns
    -------
    qair : float
        specific humidity of air
    qsea : float
        specific humidity over sea surface

    """
    if (hum == None):
        RH = np.ones(sst.shape)*80
        qsea = qsat_sea(sst, P, qmeth)/1000     # surface water q (g/kg)
        qair = qsat_air(T, P, RH, qmeth)/1000   # q of air (g/kg)
    elif (hum[0] not in ['rh', 'q', 'Td']):
        sys.exit("unknown humidity input")
        qair, qsea = np.nan, np.nan
    elif (hum[0] == 'rh'):
        RH = hum[1]
        if (np.all(RH < 1)):
            sys.exit("input relative humidity units should be \%")
            qair, qsea = np.nan, np.nan
        qsea = qsat_sea(sst, P, qmeth)/1000    # surface water q (g/kg)
        qair = qsat_air(T, P, RH, qmeth)/1000  # q of air (g/kg)
    elif (hum[0] == 'q'):
        qair = hum[1]
        qsea = qsat_sea(sst, P, qmeth)/1000  # surface water q (g/kg)
    elif (hum[0] == 'Td'):
        Td = hum[1] # dew point temperature (K)
        Td = np.where(Td < 200, np.copy(Td)+CtoK, np.copy(Td))
        T = np.where(T < 200, np.copy(T)+CtoK, np.copy(T))
        esd = 611.21*np.exp(17.502*((Td-273.16)/(Td-32.19)))
        es = 611.21*np.exp(17.502*((T-273.16)/(T-32.19)))
        RH = 100*esd/es
        qair = qsat_air(T, P, RH, qmeth)/1000  # q of air (g/kg)
        qsea = qsat_sea(sst, P, qmeth)/1000    # surface water q (g/kg)
    return qair, qsea
#-------------------------------------------------------------------------


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
        cskin q adjustment (q/kg)
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
        usr = np.where(h_in[0]/monob < -1.574, kappa*wind /
                       (np.log(-1.574*monob/zo)-psim_calc(-1.574, meth) +
                        psim_calc(zo/monob, meth) +
                        1.14*(np.power(-h_in[0]/monob, 1/3) -
                        np.power(1.574, 1/3))),
                       np.where((h_in[0]/monob > -1.574) & (h_in[0]/monob < 0),
                                kappa*wind/(np.log(h_in[0]/zo) -
                                psim_calc(h_in[0]/monob, meth) +
                                psim_calc(zo/monob, meth)),
                                np.where((h_in[0]/monob > 0) &
                                (h_in[0]/monob < 1),
                                kappa*wind/(np.log(h_in[0]/zo) +
                                5*h_in[0]/monob-5*zo/monob),
                                kappa*wind/(np.log(monob/zo)+5-5*zo/monob +
                                5*np.log(h_in[0]/monob)+h_in[0]/monob-1))))
                                # Zeng et al. 1998 (7-10)
        tsr = np.where(h_in[1]/monob < -0.465, kappa*(dt+dter*cskin) /
                       (np.log((-0.465*monob)/zot) -
                        psit_calc(-0.465, meth)+0.8*(np.power(0.465, -1/3) -
                        np.power(-h_in[1]/monob, -1/3))),
                       np.where((h_in[1]/monob > -0.465) & (h_in[1]/monob < 0),
                                kappa*(dt+dter*cskin)/(np.log(h_in[1]/zot) -
                       psit_calc(h_in[1]/monob, meth) +
                       psit_calc(zot/monob, meth)),
                        np.where((h_in[1]/monob > 0) & (h_in[1]/monob < 1),
                                 kappa*(dt+dter*cskin)/(np.log(h_in[1]/zot) +
                                 5*h_in[1]/monob-5*zot/monob),
                                 kappa*(dt+dter*cskin)/(np.log(monob/zot)+5 -
                                 5*zot/monob+5*np.log(h_in[1]/monob) +
                                 h_in[1]/monob-1))))
                                # Zeng et al. 1998 (11-14)
        qsr = np.where(h_in[2]/monob < -0.465, kappa*(dq+dqer*cskin) /
                       (np.log((-0.465*monob)/zoq) -
                        psit_calc(-0.465, meth)+psit_calc(zoq/monob, meth) +
                        0.8*(np.power(0.465, -1/3) -
                             np.power(-h_in[2]/monob, -1/3))),
                       np.where((h_in[2]/monob > -0.465) & (h_in[2]/monob < 0),
                                kappa*(dq+dqer*cskin)/(np.log(h_in[1]/zot) -
                                psit_calc(h_in[2]/monob, meth) +
                                psit_calc(zoq/monob, meth)),
                                np.where((h_in[2]/monob > 0) &
                                         (h_in[2]/monob<1),
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
#------------------------------------------------------------------------------


def get_heights(h, dim_len):
    """ Reads input heights for velocity, temperature and humidity

    Parameters
    ----------
    h : float
        input heights (m)
    dim_len : int
        length dimension

    Returns
    -------
    hh : array
    """
    hh = np.zeros((3, dim_len))
    if (type(h) == float or type(h) == int):
        hh[0, :], hh[1, :], hh[2, :] = h, h, h
    elif (len(h) == 2 and np.ndim(h) == 1):
        hh[0, :], hh[1, :], hh[2, :] = h[0], h[1], h[1]
    elif (len(h) == 3 and np.ndim(h) == 1):
        hh[0, :], hh[1, :], hh[2, :] = h[0], h[1], h[2]
    elif (len(h) == 1 and np.ndim(h) == 2):
        hh = np.zeros((3, h.shape[1]))
        hh[0, :], hh[1, :], hh[2, :] = h[0, :], h[0, :], h[0, :]
    elif (len(h) == 2 and np.ndim(h) == 2):
        hh = np.zeros((3, h.shape[1]))
        hh[0, :], hh[1, :], hh[2, :] = h[0, :], h[1, :], h[1, :]
    elif (len(h) == 3 and np.ndim(h) == 2):
        hh = np.zeros((3, h.shape[1]))
        hh = np.copy(h)
    return hh
# ---------------------------------------------------------------------


def qsat_sea(T, P, qmeth):
    """ Computes surface saturation specific humidity (g/kg)

    Parameters
    ----------
    T : float
        temperature ($^\\circ$\\,C)
    P : float
        pressure (mb)
    qmeth : str
        method to calculate vapor pressure

    Returns
    -------
    qs : float
    """
    T = np.asarray(T)
    if (np.nanmin(T) > 200):  # if Ta in Kelvin convert to Celsius
        T = T-CtoK
    ex = VaporPressure(T, P, 'liquid', qmeth)
    es = 0.98*ex  # reduction at sea surface
    qs = 622*es/(P-0.378*es)
    return qs
# ------------------------------------------------------------------------------


def qsat_air(T, P, rh, qmeth):
    """ Computes saturation specific humidity (g/kg) as in C35

    Parameters
    ----------
    T : float
        temperature ($^\circ$\,C)
    P : float
        pressure (mb)
    rh : float
       relative humidity (%)
    qmeth : str
        method to calculate vapor pressure

    Returns
    -------
    q : float
    em : float
    """
    T = np.asarray(T)
    if (np.nanmin(T) > 200):  # if Ta in Kelvin convert to Celsius
        T = T-CtoK
    es = VaporPressure(T, P, 'liquid', qmeth)
    em = 0.01*rh*es
    q = 622*em/(P-0.378*em)
    return q
# ---------------------------------------------------------------------


def gc(lat, lon=None):
    """ Computes gravity relative to latitude

    Parameters
    ----------
    lat : float
        latitude ($^\circ$)
    lon : float
        longitude ($^\circ$, optional)

    Returns
    -------
    gc : float
        gravity constant (m/s^2)
    """
    gamma = 9.7803267715
    c1 = 0.0052790414
    c2 = 0.0000232718
    c3 = 0.0000001262
    c4 = 0.0000000007
    if lon is not None:
        lon_m, lat_m = np.meshgrid(lon, lat)
    else:
        lat_m = lat
    phi = lat_m*np.pi/180.
    xx = np.sin(phi)
    gc = (gamma*(1+c1*np.power(xx, 2)+c2*np.power(xx, 4)+c3*np.power(xx, 6) +
          c4*np.power(xx, 8)))
    return gc
# ---------------------------------------------------------------------


def visc_air(T):
    """ Computes the kinematic viscosity of dry air as a function of air temp.
    following Andreas (1989), CRREL Report 89-11.

    Parameters
    ----------
    Ta : float
        air temperature ($^\circ$\,C)

    Returns
    -------
    visa : float
        kinematic viscosity (m^2/s)
    """
    T = np.asarray(T)
    if (np.nanmin(T) > 200):  # if Ta in Kelvin convert to Celsius
        T = T-273.16
    visa = 1.326e-5*(1+6.542e-3*T+8.301e-6*np.power(T, 2) -
                     4.84e-9*np.power(T, 3))
    return visa
