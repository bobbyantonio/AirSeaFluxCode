# Copyright 2023-2025, Stavroula Biri
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from .AirSeaFluxCode import AirSeaFluxCode

from .cs_wl_subs import (cs, cs_Beljaars, cs_C35,
                         cs_ecmwf, delta, get_dqer, wl_ecmwf)
from .flux_subs import (cd_calc, cdn_calc, cdn_from_roughness, ctq_calc,
                        ctqn_calc, get_LRb, get_Ltsrv, get_Rb, apply_GF,
                        get_gust, get_strs, get_tsrv, get_stabco, psim_calc,
                        psit_calc, psi_Bel, psi_ecmwf, psit_26, psi_conv,
                        psi_stab, psim_ecmwf, psiu_26, psim_conv, psim_stab)
from .hum_subs import gamma, get_hum, qsat_air, qsat_sea, VaporPressure
from .util_subs import (CtoK, kappa, get_heights, gc,
                        visc_air, set_flag, get_outvars, rho_air)

__all__ = ['AirSeaFluxCode', 'cs', 'cs_Beljaars', 'cs_C35', 'cs_ecmwf',
           'delta', 'get_dqer', 'wl_ecmwf', 'cd_calc', 'cdn_calc',
           'cdn_from_roughness', 'ctq_calc', 'ctqn_calc', 'get_LRb',
           'get_Ltsrv', 'get_Rb', 'apply_GF', 'get_gust', 'get_strs',
           'get_tsrv', 'get_stabco', 'psim_calc', 'psit_calc', 'psi_Bel',
           'psi_ecmwf', 'psit_26', 'psi_conv', 'psi_stab', 'psim_ecmwf',
           'psiu_26', 'psim_conv', 'psim_stab', 'gamma', 'get_hum', 'qsat_air',
           'qsat_sea', 'VaporPressure', 'CtoK', 'kappa', 'get_heights', 'gc',
           'visc_air', 'set_flag', 'get_outvars', 'rho_air']

__base__ = os.path.dirname(__file__)

__version__ = '1.2.0'