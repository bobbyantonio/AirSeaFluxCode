import numpy as np
import sys

def get_init(spd, T, SST, lat, hum, P, Rl, Rs, cskin, skin, wl, gust, L, tol,
             n, meth, qmeth):
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
    hum : float
        relative humidity, if None is set to 80%
    P : float
        air pressure (hPa), default 1013hPa
    Rl : float
        downward longwave radiation (W/m^2)
    Rs : float
        downward shortwave radiation (W/m^2)
    cskin : int
        cool skin correction (0: off, 1: on)
    skin : str
        cool skin adjustment method "C35" (default), "ecmwf" or "Beljaars"
    wl : int
        warm layer correction (0: off default, 1: on)
    gust : int
        3x1 [x, beta, zi] x=1 to include the effect of gustiness, else 0
        beta gustiness parameter, beta=1 for UA, beta=1.2 for COARE
        zi PBL height (m) 600 for COARE, 1000 for UA and ecmwf, 800 default
        default for COARE [1, 1.2, 600]
        default for UA, ecmwf [1, 1, 1000]
        default else [1, 1.2, 800]
    L : int
        Monin-Obukhov length definition options
    tol : float
        4x1 or 7x1 [option, lim1-3 or lim1-6]
        option : 'flux' to set tolerance limits for fluxes only lim1-3
        option : 'ref' to set tolerance limits for height adjustment lim-1-3
        option : 'all' to set tolerance limits for both fluxes and height
                 adjustment lim1-6 ['all', 0.01, 0.01, 5e-05, 1e-3, 0.1, 0.1]
    n : int
        number of iterations
    meth : str
        "S80","S88","LP82","YT96","UA","NCAR","C30","C35","ecmwf",
        "Beljaars"
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
    skin : str
        cool skin adjustment method
    wl : int
        warm layer correction switch
    gust : int
        gustiness switch
    tol : float
        tolerance limits
    L : int
        MO length switch
    n : int
        number of iterations

    """
    # check if input is correct (type, size, value) and set defaults
    if ((type(spd) != np.ndarray) or (type(T) != np.ndarray) or
         (type(SST) != np.ndarray)):
        sys.exit("input type of spd, T and SST should be numpy.ndarray")
    elif ((spd.dtype not in ['float64', 'float32']) or
          (T.dtype not in ['float64', 'float32']) or
          (SST.dtype not in ['float64', 'float32'])):
        sys.exit("input dtype of spd, T and SST should be float")
    # if input values are nan break
    if meth not in ["S80", "S88", "LP82", "YT96", "UA", "NCAR", "C30", "C35",
                    "ecmwf", "Beljaars"]:
        sys.exit("unknown method")
    if qmeth not in ["HylandWexler", "Hardy", "Preining", "Wexler",
                     "GoffGratch", "WMO", "MagnusTetens", "Buck", "Buck2",
                     "WMO2018", "Sonntag", "Bolton", "IAPWS", "MurphyKoop"]:
        sys.exit("unknown q-method")
    if (np.all(np.isnan(spd)) or np.all(np.isnan(T)) or np.all(np.isnan(SST))):
        sys.exit("input wind, T or SST is empty")
    if (np.all(lat == None)):  # set latitude to 45deg if empty
        lat = 45*np.ones(spd.shape)
    elif ((np.all(lat != None)) and (np.size(lat) == 1)):
        lat = np.ones(spd.shape)*np.copy(lat)
    if (hum == None):
        RH = np.ones(SST.shape)*80
        hum = ['no', RH]
    else:
        hum = hum
    if ((np.all(P == None)) or np.all(np.isnan(P))):
        P = np.ones(spd.shape)*1013
    elif (((np.all(P != None)) or np.all(~np.isnan(P))) and np.size(P) == 1):
        P = np.ones(spd.shape)*np.copy(P)
    if ((cskin == None) and (meth == "S80" or meth == "S88" or meth == "LP82"
                             or meth == "YT96" or meth == "UA" or
                             meth == "NCAR")):
        cskin = 0
    elif ((cskin == None) and (meth == "C30" or meth == "C35"
                               or meth == "ecmwf" or meth == "Beljaars")):
        cskin = 1
        if ((skin == None) and (meth == "C30" or meth == "C35")):
            skin = "C35"
        elif ((skin == None) and (meth == "ecmwf")):
            skin = "ecmwf"
        elif ((skin == None) and (meth == "Beljaars")):
            skin = "Beljaars"
    if (wl == None):
        wl = 0
    if ((cskin == 1 or wl == 1) and (np.all(Rl == None) or np.all(np.isnan(Rl)))
        and ((np.all(Rs == None) or np.all(np.isnan(Rs))))):
        sys.exit("Cool skin/warm layer is switched ON; Radiation input should not be"
                 " empty")
    elif (cskin == 0):
        if (np.all(Rl == None)):
            Rl = np.ones(spd.shape)*np.nan
        if (np.all(Rs == None)):
            Rs = np.ones(spd.shape)*np.nan
    if (np.all(gust == None) and (meth == "C30" or meth == "C35")):
        gust = [1, 1.2, 600]
    elif (np.all(gust == None) and (meth == "UA" or meth == "ecmwf" or
                                    meth == "Beljaars")):
        gust = [1, 1, 1000]
    elif np.all(gust == None):
        gust = [1, 1.2, 800]
    elif ((np.size(gust) < 3) and (gust == 0)):
        gust = [0, 0, 0]
    elif (np.size(gust) < 3):
        sys.exit("gust input must be a 3x1 array")
    if (L not in [None, "tsrv", "Rb"]):
        sys.exit("L input must be either None, tsrv or Rb")
    if ((L == None) and ((meth != "ecmwf") and (meth != "Beljaars"))):
        L = "tsrv"
    elif ((L == None) and ((meth == "ecmwf") or (meth == "Beljaars"))):
        L = "Rb"
    if (tol == None):
        tol = ['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1]
    elif (tol[0] not in ['flux', 'ref', 'all']):
        sys.exit("unknown tolerance input")
    if (n < 5):
        n = 5
        print("Number of iterations should not be less than 5; n is set to 5")
    return lat, hum, P, Rl, Rs, cskin, skin, wl, gust, tol, L, n