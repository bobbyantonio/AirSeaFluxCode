import numpy as np

CtoK = 273.16  # 273.15
""" Conversion factor for $^\circ\,$C to K """

kappa = 0.4  # NOTE: 0.41
""" von Karman's constant """
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
