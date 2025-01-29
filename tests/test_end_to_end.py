import sys, os
import unittest
import numpy as np
import pandas as pd
from glob import glob
import xarray as xr
import numpy as np
import pandas as pd
from src import AirSeaFluxCode

DATA_FOLDER = os.path.join(os.getcwd(), 'tests', 'data')

class TestData(unittest.TestCase):
    
    def test_warm_start(self):
        
       
        flux_df = pd.read_csv(os.path.join(DATA_FOLDER, 'flux_df.csv'))
        sea_surface_df = pd.read_csv(os.path.join(DATA_FOLDER, 'sea_surface_df.csv'))

        meth = "ecmwf"
        spd=flux_df['wind_speed'].copy().to_numpy()
        T=flux_df['2m_temperature'].copy().to_numpy() 
        SST=sea_surface_df['sea_surface_temperature'].to_numpy()
        SST_fl="bulk"
        lat=flux_df['latitude'].to_numpy()
        hin=np.array([10, 2])
        hum=('q', flux_df['specific_humidity_surface'].copy().to_numpy())
        hout=10
        maxiter=10
        P=flux_df['mean_sea_level_pressure'].copy().to_numpy()
        cskin=1
        skin=None # Already set by model
        Rs=flux_df['mean_surface_downward_short_wave_radiation_flux'].copy().to_numpy()
        Rl=flux_df['mean_surface_downward_long_wave_radiation_flux'].copy().to_numpy()
        tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1]
        L="tsrv"
        wl=1
        qmeth="Buck2"
        gust=[4, 1.2, 600, 0.01]
        out_var = ("tau", "sensible", "latent", "cd", "cp", "ct", "cq", "rho", 'dter', 'dqer', 'dtwl', 'rh', 'lv', 'qsea')
        
        out=0
                     
        iclass = AirSeaFluxCode.method_lookup_dict[meth]()
        iclass.add_gust(gust=gust)
        iclass.add_variables(spd, T, SST, SST_fl, cskin=cskin, lat=lat, hum=hum,
                            P=P, L=L)

        iclass.get_heights(hin, hout)

        iclass.get_humidity_and_potential_temperature(qmeth=qmeth)

        iclass.set_coolskin_warmlayer(wl=wl, 
                                      cskin=cskin, 
                                      skin=skin, 
                                      Rl=Rl, Rs=Rs)
   
        iclass.iterate(tol=tol, maxiter=maxiter)
        res_sst_df = iclass.get_output(out_var=out_var, out=out)
        
        num_nans = res_sst_df['ct'].isna().sum() / len(res_sst_df['ct'])
        
        self.assertTrue(num_nans < 0.2)
