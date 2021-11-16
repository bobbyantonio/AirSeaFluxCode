import warnings
import numpy as np
import pandas as pd
import logging
from hum_subs import (get_hum, gamma)
from util_subs import *
from flux_subs import *
from cs_wl_subs import *


class S88:
    def _wind_iterate(self, ind):
        if self.gust[0] in [1, 2]:
            self.wind[ind] = np.sqrt(np.power(np.copy(self.spd[ind]), 2) +
                                     np.power(get_gust(self.gust[1],
                                                       self.theta[ind],
                                                       self.usr[ind],
                                            self.tsrv[ind], self.gust[2],
                                            self.grav[ind]), 2))
            # ratio of gusty to horizontal wind
            self.GustFact[ind] = self.wind[ind]/self.spd[ind]
            # if we update wind, also need to update u10n
            # remove effect of gustiness following Fairall et al. (2003)
            # usr is divided by (GustFact)^0.5
            self.u10n[ind] = self.wind[ind]-self.usr[ind]/kappa/np.sqrt(
                self.GustFact[ind])*(np.log(self.h_in[0, ind]/self.ref10) -
                                      self.psim[ind])
            # temporary as in C35
            # self.u10n[ind] = self.usr[ind]/kappa/self.GustFact[ind]*np.log(
            #     self.ref10/self.zo[ind])
            # temporary as in UA
            # self.u10n[ind] = self.usr[ind]/kappa/np.log(
            #     self.ref10/self.zo[ind])
            if self.gust[0] == 2:
                self.GustFact[ind] = 1
                # option to not remove GustFact
                self.u10n[ind] = self.wind[ind]-self.usr[ind]/kappa*(
                    np.log(self.h_in[0, ind]/self.ref10)-self.psim[ind])
            # these lines integrate up from the surface - doesn't work when
            # gustiness is on
            # self.u10n[ind] = self.usr[ind]/kappa/np.power(
            #     self.GustFact[ind], 0.5)*(np.log(self.ref10/self.zo[ind]))
            # self.u10n[ind] = self.usr[ind]/kappa * \
            #     (np.log(self.ref10/self.zo[ind]))
        else:
            # not sure this is needed - perhaps only to remove effects of
            # initalisation of wind
            self.wind[ind] = np.copy(self.spd[ind])
            self.u10n[ind] = self.wind[ind]-self.usr[ind]/kappa*(
                    np.log(self.h_in[0, ind]/self.ref10)-self.psim[ind])
        # temporary to check feed into iteration loop
        # print('Mean GF: {} | Du {} | u10n {}'.format(
        #     np.nanmean(self.GustFact), np.nanmean(self.spd-self.wind),
        #     np.nanmean(self.u10n)))


    def get_heights(self, hin, hout=10):
        self.hout = hout
        self.hin = hin
        self.h_in = get_heights(hin, len(self.spd))
        self.h_out = get_heights(self.hout, 1)

    def get_specHumidity(self, qmeth="Buck2"):
        self.qair, self.qsea = get_hum(self.hum, self.T, self.SST, self.P,
                                       qmeth)
        if (np.all(np.isnan(self.qsea)) or np.all(np.isnan(self.qair))):
            raise ValueError("qsea and qair cannot be nan")
        self.dq_in = self.qair-self.qsea
        self.dq_full = self.qair-self.qsea

        # Set lapse rate and Potential Temperature (now we have humdity)
        self.cp = 1004.67*(1+0.00084*self.qsea)
        self.tlapse = gamma("dry", self.SST, self.T, self.qair/1000, self.cp)
        self.theta = np.copy(self.T)+self.tlapse*self.h_in[1]
        self.dt_in = self.theta-self.SST
        self.dt_full = self.theta-self.SST

    def _fix_coolskin_warmlayer(self, wl, cskin, skin, Rl, Rs):
        skin = self.skin if skin is None else skin
        assert wl in [0, 1], "wl not valid"
        assert cskin in [0, 1], "cskin not valid"
        assert skin in ["C35", "ecmwf", "Beljaars"], "Skin value not valid"

        if ((cskin == 1 or wl == 1) and
            (np.all(Rl == None) or np.all(np.isnan(Rl))) and
            ((np.all(Rs == None) or np.all(np.isnan(Rs))))):
            print("Cool skin/warm layer is switched ON; "
                  "Radiation input should not be empty")
            raise

        self.wl = wl
        self.cskin = cskin
        self.skin = skin
        self.Rs = np.full(self.spd.shape, np.nan) if Rs is None else Rs
        self.Rl = np.full(self.spd.shape, np.nan) if Rl is None else Rl
        # print(self.meth, self.cskin, self.skin, self.wl)

    def set_coolskin_warmlayer(self, wl=0, cskin=0, skin=None, Rl=None,
                               Rs=None):
        wl = 0 if wl is None else wl
        if hasattr(self, "skin") == False:
            self.skin = "C35"
        self._fix_coolskin_warmlayer(wl, cskin, skin, Rl, Rs)


    def _update_coolskin_warmlayer(self, ind):
        if self.cskin == 1:
            if self.skin == "C35":
                self.dter[ind], self.tkt[ind] = cs_C35(np.copy(
                    self.SST[ind]), self.rho[ind], self.Rs[ind], self.Rnl[ind],
                    self.cp[ind], self.lv[ind], np.copy(self.tkt[ind]),
                    self.usr[ind], self.tsr[ind], self.qsr[ind], self.grav[ind])
            elif self.skin == "ecmwf":
                self.dter[ind] = cs_ecmwf(
                    self.rho[ind], self.Rs[ind], self.Rnl[ind], self.cp[ind],
                    self.lv[ind], self.usr[ind], self.tsr[ind], self.qsr[ind],
                    np.copy(self.SST[ind]), self.grav[ind])
            elif self.skin == "Beljaars":
                self.Qs[ind], self.dter[ind] = cs_Beljaars(
                    self.rho[ind], self.Rs[ind], self.Rnl[ind], self.cp[ind],
                    self.lv[ind], self.usr[ind], self.tsr[ind], self.qsr[ind],
                    self.grav[ind], np.copy(self.Qs[ind]))
            self.dqer[ind] = get_dqer(self.dter[ind], self.SST[ind],
                                      self.qsea[ind], self.lv[ind])
            self.skt[ind] = np.copy(self.SST[ind])+self.dter[ind]
            self.skq[ind] = np.copy(self.qsea[ind])+self.dqer[ind]
            if self.wl == 1:
                self.dtwl[ind] = wl_ecmwf(
                    self.rho[ind], self.Rs[ind], self.Rnl[ind], self.cp[ind],
                    self.lv[ind], self.usr[ind], self.tsr[ind], self.qsr[ind],
                    np.copy(self.SST[ind]), np.copy(self.skt[ind]),
                    np.copy(self.dter[ind]), self.grav[ind])
                self.skt[ind] = (np.copy(self.SST[ind])+self.dter[ind] +
                                 self.dtwl[ind])
                self.dqer[ind] = get_dqer(self.dter[ind], self.skt[ind],
                                          self.qsea[ind], self.lv[ind])
                self.skq[ind] = np.copy(self.qsea[ind])+self.dqer[ind]
        else:
            self.dter[ind] = np.zeros(self.SST[ind].shape)
            self.dqer[ind] = np.zeros(self.SST[ind].shape)
            self.dtwl[ind] = np.zeros(self.SST[ind].shape)
            self.tkt[ind] = np.zeros(self.SST[ind].shape)

    def _first_guess(self):
        # reference height1
        self.ref10 = 10

        #  first guesses
        self.t10n, self.q10n = np.copy(self.theta), np.copy(self.qair)
        self.rho = self.P*100/(287.1*self.t10n*(1+0.6077*self.q10n))
        self.lv = (2.501-0.00237*(self.SST-CtoK))*1e6  # J/kg

        #  Zeng et al. 1998
        self.tv = self.theta*(1+0.6077*self.qair)   # virtual potential T
        self.dtv = self.dt_in*(1+0.6077*self.qair)+0.6077*self.theta*self.dq_in

        # Set the wind array
        # self.wind = np.maximum(np.copy(self.spd), 3)
        self.wind = np.sqrt(np.power(np.copy(self.spd), 2)+0.25)
        self.GustFact = self.wind*0+1

        # Rb eq. 11 Grachev & Fairall 1997, use air temp height
        # use self.tv??  adjust wind to T-height?
        Rb = self.grav*self.h_in[1]*self.dtv/(self.T*np.power(self.wind, 2))
        self.monob = self.h_in[1]/12.0/Rb  # eq. 12 Grachev & Fairall 1997   # DO.THIS
        # where does 12.0 come from??

        # ------------

        # dummy_array = lambda val : np.full(self.T.shape, val)*self.msk
        def dummy_array(val): return np.full(self.T.shape, val)*self.msk
        if self.cskin + self.wl > 0:
            self.dter, self.tkt, self.dtwl = [
                dummy_array(x) for x in (-0.3, 0.001, 0.3)]
            self.dqer = get_dqer(self.dter, self.SST, self.qsea,
                                 self.lv)
            self.Rnl = 0.97*(self.Rl-5.67e-8*np.power(
                self.SST-0.3*self.cskin, 4))
            self.Qs = 0.945*self.Rs
        else:
            self.dter, self.dqer, self.dtwl = [
                dummy_array(x) for x in (0.0, 0.0, 0.0)]
            self.Rnl, self.Qs, self.tkt = [
                np.empty(self.arr_shp)*self.msk for _ in range(3)]
        self.skt = np.copy(self.SST)
        self.skq = np.copy(self.qsea)

        self.u10n = np.copy(self.wind)
        self.usr = 0.035*self.u10n
        self.cd10n, self.zo = cdn_calc(
            self.u10n, self.usr, self.theta, self.grav, self.meth)
        self.psim = psim_calc(self.h_in[0]/self.monob, self.meth)
        self.cd = cd_calc(self.cd10n, self.h_in[0], self.ref10, self.psim)
        self.usr = np.sqrt(self.cd*np.power(self.wind, 2))
        self.zot, self.zoq, self.tsr, self.qsr = [
            np.empty(self.arr_shp)*self.msk for _ in range(4)]
        self.ct10n, self.cq10n, self.ct, self.cq = [
            np.empty(self.arr_shp)*self.msk for _ in range(4)]
        self.tv10n = self.zot  # remove from output

    def iterate(self, maxiter=10, tol=None):
        if maxiter < 5:
            warnings.warn("Iteration number <5 - resetting to 5.")
            maxiter = 5

        # Decide which variables to use in tolerances based on tolerance
        # specification
        tol = ['all', 0.01, 0.01, 1e-05, 1e-3,
               0.1, 0.1] if tol is None else tol
        assert tol[0] in ['flux', 'ref', 'all'], "unknown tolerance input"

        old_vars = {"flux": ["tau", "sensible", "latent"],
                    "ref": ["u10n", "t10n", "q10n"]}
        old_vars["all"] = old_vars["ref"] + old_vars["flux"]
        old_vars = old_vars[tol[0]]

        new_vars = {"flux": ["tau", "sensible", "latent"],
                    "ref": ["u10n", "t10n", "q10n"]}
        new_vars["all"] = new_vars["ref"] + new_vars["flux"]
        new_vars = new_vars[tol[0]]
        # I'm sure there are better ways of doing this
        # extract tolerance values by deleting flag from tol
        tvals = np.delete(np.copy(tol), 0)
        tol_vals = list([float(tt) for tt in tvals])

        ind = np.where(self.spd > 0)
        it = 0

        # Setup empty arrays
        self.tsrv, self.psim, self.psit, self.psiq = [
            np.zeros(self.arr_shp)*self.msk for _ in range(4)]

        # extreme values for first comparison
        dummy_array = lambda val: np.full(self.arr_shp, val)*self.msk
        # you can use def instead of lambda
        # def dummy_array(val): return np.full(self.arr_shp, val)*self.msk
        self.itera, self.tau, self.sensible, self.latent = [
            dummy_array(x) for x in (-1, 1e+99, 1e+99, 1e+99)]

        # Generate the first guess values
        self._first_guess()

        #  iteration loop
        ii = True
        while ii & (it < maxiter):
            it += 1

            # Set the old variables (for comparison against "new")
            old = np.array([np.copy(getattr(self, i)) for i in old_vars])

            # Calculate cdn
            self.cd10n[ind], self.zo[ind] = cdn_calc(
                self.u10n[ind], self.usr[ind], self.theta[ind], self.grav[ind],
                self.meth)

            if np.all(np.isnan(self.cd10n)):
                logging.info('break %s at iteration %s cd10n<0', meth, it)
                break


            self.psim[ind] = psim_calc(
                self.h_in[0, ind]/self.monob[ind], self.meth)
            self.cd[ind] = cd_calc(
                self.cd10n[ind], self.h_in[0, ind], self.ref10, self.psim[ind])
            # remove effect of gustiness following Fairall et al. (2003)
            # usr is divided by (GustFact)^0.5 (updated in wind_iterate)
            # these 2 equations integrating from surface should be equivalent
            # - but they are not due to gustiness
            # self.u10n[ind] = self.usr[ind]/kappa/np.power(
            #     self.GustFact[ind],0.5)*(np.log(self.ref10/self.zo[ind]))
            # self.u10n[ind] = self.usr[ind]/kappa*(
            #     np.log(self.ref10/self.zo[ind]))
            # lines below with and without gustfactor
            # if self.gust[0] == 1:
            #     self.u10n[ind] = self.wind[ind]-self.usr[ind]/kappa/np.sqrt(
            #         self.GustFact[ind], 0.5)*(np.log(
            #             self.h_in[0, ind]/self.ref10)-self.psim[ind])
            # elif self.gust[0] == 2:
            #     self.u10n[ind] = self.wind[ind]-self.usr[ind]/kappa*(
            #         np.log(self.h_in[0, ind]/self.ref10)-self.psim[ind])
            # Update the wind values
            self._wind_iterate(ind)


            # temperature
            self.ct10n[ind], self.zot[ind] = ctqn_calc(
                "ct", self.h_in[1, ind]/self.monob[ind], self.cd10n[ind],
                self.usr[ind], self.zo[ind], self.theta[ind], self.meth)
            self.psit[ind] = psit_calc(
                self.h_in[1, ind]/self.monob[ind], self.meth)
            self.ct[ind] = ctq_calc(
                self.cd10n[ind], self.cd[ind], self.ct10n[ind],
                self.h_in[1, ind], self.ref10, self.psit[ind])
            # wind
            self.cq10n[ind], self.zoq[ind] = ctqn_calc(
                "cq", self.h_in[2, ind]/self.monob[ind], self.cd10n[ind],
                self.usr[ind], self.zo[ind], self.theta[ind], self.meth)
            self.psiq[ind] = psit_calc(
                self.h_in[2, ind]/self.monob[ind], self.meth)
            self.cq[ind] = ctq_calc(
                self.cd10n[ind], self.cd[ind], self.cq10n[ind],
                self.h_in[2, ind], self.ref10, self.psiq[ind])


            # Some parameterizations set a minimum on parameters
            try:
                self._minimum_params()
            except AttributeError:
                pass

            self.dt_full[ind] = self.dt_in[ind] - \
                self.dter[ind]*self.cskin - self.dtwl[ind]*self.wl
            self.dq_full[ind] = self.dq_in[ind] - self.dqer[ind]*self.cskin
            self.usr[ind], self.tsr[ind], self.qsr[ind] = get_strs(
                self.h_in[:, ind], self.monob[ind], self.wind[ind],
                self.zo[ind], self.zot[ind], self.zoq[ind], self.dt_full[ind],
                self.dq_full[ind], self.cd[ind], self.ct[ind], self.cq[ind],
                self.meth)

            # Update CS/WL parameters
            self._update_coolskin_warmlayer(ind)

            # Logging output
            log_vars = {"dter": 2, "dqer": 7, "tkt": 2,
                        "Rnl": 2, "usr": 3, "tsr": 4, "qsr": 7}
            log_vars = [np.round(np.nanmedian(getattr(self, V)), R)
                        for V, R in log_vars.items()]
            log_vars.insert(0, self.meth)
            logging.info(
                'method {} | dter = {} | dqer = {} | tkt = {} | Rnl = {} |'
                ' usr = {} | tsr = {} | qsr = {}'.format(*log_vars))

            if self.cskin + self.wl > 0:
                self.Rnl[ind] = 0.97*(self.Rl[ind]-5.67e-8 *
                                      np.power(self.SST[ind] +
                                               self.dter[ind]*self.cskin, 4))
            # not sure how to handle lapse/potemp
            # well-mixed in potential temperature ...
            self.t10n[ind] = self.theta[ind]-self.tlapse[ind]*self.ref10 - \
                self.tsr[ind]/kappa * \
                    (np.log(self.h_in[1, ind]/self.ref10)-self.psit[ind])
            self.q10n[ind] = self.qair[ind]-self.qsr[ind]/kappa * \
                (np.log(self.h_in[2, ind]/self.ref10)-self.psiq[ind])

            # update stability info
            self.tsrv[ind] = get_tsrv(
                self.tsr[ind], self.qsr[ind], self.theta[ind], self.qair[ind])
            self.Rb[ind] = get_Rb(
                self.grav[ind], self.usr[ind], self.h_in[0, ind],
                self.h_in[1, ind], self.tv[ind], self.dtv[ind], self.wind[ind],
                self.monob[ind], self.meth)
            if self.L == "tsrv":
                self.monob[ind] = get_Ltsrv(
                    self.tsrv[ind], self.grav[ind], self.tv[ind],
                    self.usr[ind])
            else:
                self.monob[ind] = get_LRb(
                    self.Rb[ind], self.h_in[1,ind], self.monob[ind],
                    self.zo[ind], self.zot[ind], self.meth)

            # Update the wind values
            self._wind_iterate(ind)

            # make sure you allow small negative values convergence
            if it == 1:
                self.u10n = np.where(self.u10n < 0, 0.5, self.u10n)

            self.itera[ind] = np.full(1, it)
            # remove effect of gustiness following Fairall et al. (2003)
            # usr is divided by (GustFact)^0.5 (here applied to sensible and
            # latent as well as tau)
            # GustFact should be 1 if gust is OFF
            self.tau = self.rho*np.power(self.usr, 2)/self.GustFact
            self.sensible = self.rho*self.cp*self.usr / \
                np.sqrt(self.GustFact)*self.tsr
            self.latent = self.rho*self.lv*self.usr / \
                np.sqrt(self.GustFact)*self.qsr
            # or leave as it is - gusty wind speed, or no gust
            # self.tau = self.rho*np.power(self.usr, 2)
            # self.sensible = self.rho*self.cp*self.usr*self.tsr
            # self.latent = self.rho*self.lv*self.usr*self.qsr
            # temporary as in C35, UA
            # self.tau = self.rho*np.power(self.usr, 2)/self.GustFact
            # self.sensible = self.rho*self.cp*self.usr*self.tsr
            # self.latent = self.rho*self.lv*self.usr*self.qsr

            # Set the new variables (for comparison against "old")
            new = np.array([np.copy(getattr(self, i)) for i in new_vars])

            if it > 2:  # force at least three iterations
                d = np.abs(new-old)  # change over this iteration
                for ii in range(0, len(tol_vals)):
                    d[ii, ] = d[ii, ]/tol_vals[ii]  # ratio to tolerance
                # identifies non-convergence
                ind = np.where(d.max(axis=0) >= 1)

            self.ind = np.copy(ind)
            ii = False if (ind[0].size == 0) else True
            # End of iteration loop

        self.itera[ind] = -1
        self.itera = np.where(self.itera > maxiter, -1, self.itera)
        logging.info('method %s | # of iterations:%s', self.meth, it)
        logging.info('method %s | # of points that did not converge :%s \n',
                     self.meth, self.ind[0].size)


    def _get_humidity(self):
        """Calculate RH used for flagging purposes & output."""
        if self.hum[0] in ('rh', 'no'):
            self.rh = self.hum[1]
        elif self.hum[0] == 'Td':
            Td = self.hum[1]  # dew point temperature (K)
            Td = np.where(Td < 200, np.copy(Td)+CtoK, np.copy(Td))
            T = np.where(self.T < 200, np.copy(self.T)+CtoK, np.copy(self.T))
            # T = np.copy(self.T)
            esd = 611.21*np.exp(17.502*((Td-CtoK)/(Td-32.19)))
            es = 611.21*np.exp(17.502*((T-CtoK)/(T-32.19)))
            self.rh = 100*esd/es

    def _flag(self, out=0):
        """Set the general flags."""
        flag = np.full(self.arr_shp, "n", dtype="object")

        if self.hum[0] == 'no':
            if self.cskin == 1:
                flag = np.where(
                    np.isnan(self.spd+self.T+self.SST+self.P+self.Rs+self.Rl),
                    "m", flag)
            else:
                flag = np.where(
                    np.isnan(self.spd+self.T+self.SST+self.P), "m", flag)
        else:
            if self.cskin == 1:
                flag = np.where(np.isnan(
                    self.spd+self.T+self.SST+self.hum[1]+self.P +
                    self.Rs+self.Rl), "m", flag)
            else:
                flag = np.where(
                    np.isnan(self.spd+self.T+self.SST+self.hum[1]+self.P),
                    "m", flag)

            flag = np.where(self.rh > 100, "r", flag)

        # u10n flag
        flag = np.where(((self.u10n < 0) | (self.u10n > 200)) & (flag == "n"),
                        "u",
                        np.where(((self.u10n < 0) | (self.u10n > 200)) &
                                 (np.char.find(flag.astype(str), 'u') == -1),
                                 flag+[","]+["u"], flag))
        # q10n flag
        flag = np.where(
            ((self.q10n < 0) | (self.q10n > 40*0.001)) & (flag == "n"), "q",
            np.where(
                ((self.q10n < 0) | (self.q10n > 40*0.001)) & (flag != "n"),
                flag+[","]+["q"], flag))

        # t10n flag
        flag = np.where(
            ((self.t10n < 173) | (self.t10n > 373)) & (flag == "n"), "t",
            np.where(((self.t10n < 173) | (self.t10n > 373)) & (flag != "n"),
                     flag+[","]+["t"], flag))

        flag = np.where(
            ((self.Rb < -0.5) | (self.Rb > 0.2) |
             ((self.hin[0]/self.monob) > 1000)) & (flag == "n"), "l",
            np.where(((self.Rb < -0.5) | (self.Rb > 0.2) |
                      ((self.hin[0]/self.monob) > 1000)) &
                     (flag != "n"), flag+[","]+["l"], flag))

        if out == 1:
            flag = np.where((self.itera == -1) & (flag == "n"), "i", np.where(
                (self.itera == -1) &
                ((flag != "n") & (np.char.find(flag.astype(str), 'm') == -1)),
                flag+[","]+["i"], flag))
        else:
            flag = np.where((self.itera == -1) & (flag == "n"), "i", np.where(
                (self.itera == -1) &
                ((flag != "n") & (np.char.find(flag.astype(str), 'm') == -1) &
                 (np.char.find(flag.astype(str), 'u') == -1)),
                flag+[","]+["i"], flag))
        self.flag = flag

    def get_output(self, out=0):

        assert out in [0, 1], "out must be either 0 or 1"

        self._get_humidity()  # Get the Relative humidity
        self._flag(out=out)  # Get flags

        # remove effect of gustiness following Fairall et al. (2003)
        # usr is divided by (GustFact)^0.5
        self.uref = self.spd-self.usr/kappa/np.sqrt(self.GustFact) * \
            (np.log(self.h_in[0]/self.h_out[0])-self.psim +
              psim_calc(self.h_out[0]/self.monob, self.meth))
        # include lapse rate adjustment as theta is well-mixed
        self.tref = self.theta-self.tlapse*self.h_out[1]-self.tsr/kappa * \
            (np.log(self.h_in[1]/self.h_out[1])-self.psit +
              psit_calc(self.h_out[1]/self.monob, self.meth))
        self.qref = self.qair-self.qsr/kappa * \
            (np.log(self.h_in[2]/self.h_out[2])-self.psiq +
              psit_calc(self.h_out[2]/self.monob, self.meth))
        # temporary as in C35
        # self.uref = self.spd+self.usr/kappa/self.GustFact*(
        #     np.log(self.h_out[0]/self.h_in[0]) -
        #     psim_calc(self.h_out[0]/self.monob, self.meth) +
        #     psim_calc(self.h_in[0]/self.monob, self.meth))
        # self.u10n = ((self.wind+self.usr/kappa*(
        #     np.log(self.ref10/self.h_in[0]) -
        #     psim_calc(self.ref10/self.monob, self.meth) +
        #     psim_calc(self.h_in[0]/self.monob, self.meth)))/self.GustFact +
        #     psim_calc(self.ref10/self.monob, self.meth) *
        #     self.usr/kappa/self.GustFact)
        # self.tref = (self.T+self.tsr/kappa*(
        #     np.log(self.h_out[1]/self.h_in[1]) -
        #     psit_calc(self.h_out[1]/self.monob, self.meth) +
        #     psit_calc(self.h_in[1]/self.monob, self.meth)) +
        #     self.tlapse*(self.h_in[1]-self.h_out[1]))
        # self.t10n = self.tref + \
        #     psit_calc(self.ref10/self.monob, self.meth)*self.tsr/kappa
        # self.qref = self.qair+self.qsr/kappa*(
        #     np.log(self.h_out[2]/self.h_in[2])-psit_calc(
        #         self.h_out[2]/self.monob, self.meth)+psit_calc(
        #             self.h_in[1]/self.monob, self.meth))
        # self.q10n = self.qref + \
        #     psit_calc(self.ref10/self.monob, self.meth)*self.qsr/kappa
        # temporary as in UA
        # self.uref = np.where(
        #     self.ref10/self.monob < 0, self.spd+(self.usr/kappa)*(
        #         np.log(self.ref10/self.h_in[0])-(psim_calc(
        #             self.ref10/self.monob, self.meth) -
        #             psim_calc(self.h_in[0]/self.monob, self.meth))),
        #     self.spd+(self.usr/kappa)*(np.log(self.ref10/self.h_in[0]) +
        #                                5*self.ref10/self.monob -
        #                                5*self.h_in[0]/self.monob))

        if self.wl == 0:
            self.dtwl = np.zeros(self.T.shape)*self.msk
            # reset to zero if not used

        # Do not calculate lhf if a measure of humidity is not input
        # This gets filled into a pd dataframe and so no need to specify y
        # dimension of array
        if self.hum[0] == 'no':
            self.latent, self.qsr, self.q10n = np.empty(3)
            self.qref, self.qair, self.rh = np.empty(3)

        # Set the final wind speed values
        # this seems to be gust (was wind_speed)
        self.ug = np.sqrt(np.power(self.wind, 2)-np.power(self.spd, 2))

        # Get class specific flags (will only work if self.u_hi and self.u_lo
        # have been set in the class)
        try:
            self._class_flag()
        except AttributeError:
            pass

        # Combine all output variables into a pandas array
        res_vars = ("tau", "sensible", "latent", "monob", "cd", "cd10n", "ct",
                    "ct10n", "cq", "cq10n", "tsrv", "tsr", "qsr", "usr",
                    "psim", "psit", "psiq", "u10n", "t10n", "q10n", "zo",
                    "zot", "zoq", "uref", "tref", "qref", "dter",
                    "dqer", "dtwl", "qair", "qsea", "Rl", "Rs", "Rnl", "ug",
                    "Rb", "rh", "tkt", "lv", "itera")

        res = np.zeros((len(res_vars), len(self.spd)))
        for i, value in enumerate(res_vars):
            res[i][:] = getattr(self, value)

        if out == 0:
            res[:, self.ind] = np.nan
            # set missing values where data have non acceptable values
            if self.hum[0] != 'no':
                res[:-1] = np.asarray([
                    np.where(self.q10n < 0, np.nan, res[i][:])
                    for i in range(len(res_vars)-1)])
                # len(res_vars)-1 instead of len(res_vars) in order to keep
                # itera= -1 for no convergence
            res[:-1] = np.asarray([
                np.where(self.u10n < 0, np.nan, res[i][:])
                for i in range(len(res_vars)-1)])
        else:
            warnings.warn("Warning: the output will contain values for points"
                          " that have not converged and negative values "
                          "(if any) for u10n/q10n")

        resAll = pd.DataFrame(data=res.T, index=range(self.nlen),
                              columns=res_vars)

        resAll["flag"] = self.flag

        return resAll

    def add_variables(self, spd, T, SST, lat=None, hum=None, P=None, L=None):

        # Add the mandatory variables
        assert type(spd) == type(T) == type(
            SST) == np.ndarray, "input type of spd, T and SST should be"
        " numpy.ndarray"
        self.L = "tsrv" if L is None else L
        self.arr_shp = spd.shape
        self.nlen = len(spd)
        self.spd = spd
        # self.T = T
        self.T = np.where(T < 200, np.copy(T)+CtoK, np.copy(T))
        self.hum = ['no', np.full(SST.shape, 80)] if hum is None else hum
        self.SST = np.where(SST < 200, np.copy(SST)+CtoK, np.copy(SST))
        self.lat = np.full(self.arr_shp, 45) if lat is None else lat
        self.grav = gc(self.lat)
        self.P = np.full(self.nlen, 1013) if P is None else P

        # mask to preserve missing values when initialising variables
        self.msk = np.empty(SST.shape)
        self.msk = np.where(np.isnan(spd+T+SST), np.nan, 1)
        self.Rb = np.empty(SST.shape)*self.msk

    def add_gust(self, gust=None):
        if np.all(gust is None):
            try:
                gust = self.default_gust
            except AttributeError:
                gust = [0, 0, 0]  # gustiness OFF
                # gust = [1, 1.2, 800]
        elif ((np.size(gust) < 3) and (gust == 0)):
            gust = [0, 0, 0]

        assert np.size(gust) == 3, "gust input must be a 3x1 array"
        assert gust[0] in [0, 1, 2], "gust at position 0 must be 0, 1 or 2"
        self.gust = gust

    def _class_flag(self):
        """A flag specific to this class - only used for certain classes where
         u_lo and u_hi are defined"""
        # flag.tmp = np.where(
        #     ((self.u10n < self.u_lo[0]) | (self.u10n > self.u_hi[0])),"o", "")
        # self.flag = np.where(
        #     self.flag == "n" & flag.tmp == "o", "o", self.flag)
        # self.flag = np.where(
        #     flag.tmp == "o" & self.flag != "n" & self.flag != "o" &
        #     self.flag != "m", self.flag+[","]+["o"], self.flag)
        # flag.add = ["u", "q", "t"]
        # self.flag = np.where(
        #     flag.tmp == "o" & (np.char.find(self.flag.astype(str), 'u') == -1 |
        #                        np.char.find(self.flag.astype(str), 'q') == -1 |
        #                        np.char.find(self.flag.astype(str), 't') == -1),
        #     self.flag+[","]+["o"], self.flag))
        self.flag = np.where(((self.u10n < self.u_lo[0]) |
                              (self.u10n > self.u_hi[0])) &
                             (self.flag == "n"), "o",
                             np.where(((self.u10n < self.u_lo[1]) |
                                       (self.u10n > self.u_hi[1])) &
                                      ((self.flag != "n") & (np.char.find(
                                          self.flag.astype(str), 'u') == -1) &
                                       (np.char.find(
                                           self.flag.astype(str), 'q') == -1)),
                                      self.flag+[","]+["o"], self.flag))

    def __init__(self):
        self.meth = "S88"

class S80(S88):

    def __init__(self):
        self.meth = "S80"
        self.u_lo = [6, 6]
        self.u_hi = [22, 22]


class YT96(S88):

    def __init__(self):
        self.meth = "YT96"
        # no limits to u range as we use eq. 21 for cdn
        # self.u_lo = [0, 3]
        # self.u_hi = [26, 26]

class LP82(S88):

    def __init__(self):
        self.meth = "LP82"
        self.u_lo = [3, 3]
        self.u_hi = [25, 25]


class NCAR(S88):


    def _minimum_params(self):
        self.cd = np.maximum(np.copy(self.cd), 1e-4)
        self.ct = np.maximum(np.copy(self.ct), 1e-4)
        self.cq = np.maximum(np.copy(self.cq), 1e-4)
        self.zo = np.minimum(np.copy(self.zo), 0.0025)

    def __init__(self):
        self.meth = "NCAR"
        self.u_lo = [0.5, 0.5]
        self.u_hi = [999, 999]

class UA(S88):

    def __init__(self):
        self.meth = "UA"
        self.default_gust = [1, 1, 1000]
        self.u_lo = [-999, -999]
        self.u_hi = [18, 18]


class C30(S88):
    # def set_coolskin_warmlayer(self, wl=0, cskin=1, skin="C35", Rl=None, Rs=None):
    #     self._fix_coolskin_warmlayer(wl, cskin, skin, Rl, Rs)

    def __init__(self):
        self.meth = "C30"
        self.default_gust = [1, 1.2, 600]
        self.skin = "C35"

class C35(C30):
    def __init__(self):
        self.meth = "C35"
        self.default_gust = [1, 1.2, 600]
        self.skin = "C35"

class ecmwf(C30):
    # def set_coolskin_warmlayer(self, wl=0, cskin=1, skin="ecmwf", Rl=None,
    #                            Rs=None):
    #     self._fix_coolskin_warmlayer(wl, cskin, skin, Rl, Rs)

    def __init__(self):
        self.meth = "ecmwf"
        self.default_gust = [1, 1, 1000]
        self.skin = "ecmwf"

class Beljaars(C30):
    # def set_coolskin_warmlayer(self, wl=0, cskin=1, skin="Beljaars", Rl=None,
    #                            Rs=None):
    #     self._fix_coolskin_warmlayer(wl, cskin, skin, Rl, Rs)

    def __init__(self):
        self.meth = "Beljaars"
        self.default_gust = [1, 1, 1000]
        self.skin = "Beljaars"


def AirSeaFluxCode(spd, T, SST, lat=None, hum=None, P=None, hin=18, hout=10,
                   Rl=None, Rs=None, cskin=0, skin=None, wl=0, gust=None,
                   meth="S88", qmeth="Buck2", tol=None, maxiter=10, out=0,
                   L=None):
    """
    Calculate turbulent surface fluxes using different parameterizations.

    Calculate height adjusted values for spd, T, q

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
            3x1 [x, beta, zi] x=0 gustiness is OFF, x=1 gustiness is ON and
            use gustiness factor, x=2 gustiness is ON and gustiness factor=1;
            beta gustiness parameter, beta=1 for UA, beta=1.2 for COARE
            zi PBL height (m) 600 for COARE, 1000 for UA and ecmwf
            default for COARE [1, 1.2, 600]
            default for UA, ecmwf [1, 1, 1000]
            else default is switched OFF
        meth : str
            "S80", "S88", "LP82", "YT96", "UA", "NCAR", "C30", "C35",
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
                    adjustment lim1-6
           default is tol=['all', 0.01, 0.01, 1e-05, 1e-3, 0.1, 0.1]
        maxiter : int
            number of iterations (default = 10)
        out : int
            set 0 to set points that have not converged, negative values of
                  u10n and q10n to missing (default)
            set 1 to keep points
        L : str
           Monin-Obukhov length definition options
           "tsrv"  : default for "S80", "S88", "LP82", "YT96", "UA", "NCAR",
                     "C30", "C35"
           "Rb" : following ecmwf (IFS Documentation cy46r1), default for
                  "ecmwf", "Beljaars"
    Returns
    -------
        res : array that contains
                       1. momentum flux       (N/m^2)
                       2. sensible heat       (W/m^2)
                       3. latent heat         (W/m^2)
                       4. Monin-Obhukov length (m)
                       5. drag coefficient (cd)
                       6. neutral drag coefficient (cd10n)
                       7. heat exchange coefficient (ct)
                       8. neutral heat exchange coefficient (ct10n)
                       9. moisture exhange coefficient (cq)
                       10. neutral moisture exchange coefficient (cq10n)
                       11. star virtual temperatcure (tsrv)
                       12. star temperature (tsr)
                       13. star specific humidity (qsr)
                       14. star wind speed (usr)
                       15. momentum stability function (psim)
                       16. heat stability function (psit)
                       17. moisture stability function (psiq)
                       18. 10m neutral wind speed (u10n)
                       19. 10m neutral temperature (t10n)
                       20. 10m neutral specific humidity (q10n)
                       21. surface roughness length (zo)
                       22. heat roughness length (zot)
                       23. moisture roughness length (zoq)
                       24. wind speed at reference height (uref)
                       25. temperature at reference height (tref)
                       26. specific humidity at reference height (qref)
                       27. cool-skin temperature depression (dter)
                       28. cool-skin humidity depression (dqer)
                       29. warm layer correction (dtwl)
                       30. specific humidity of air (qair)
                       31. specific humidity at sea surface (qsea)
                       32. downward longwave radiation (Rl)
                       33. downward shortwave radiation (Rs)
                       34. downward net longwave radiation (Rnl)
                       35. gust wind speed (ug)
                       36. Bulk Richardson number (Rib)
                       37. relative humidity (rh)
                       38. thickness of the viscous layer (delta)
                       39. lv latent heat of vaporization (Jkg−1)
                       40. number of iterations until convergence
                       41. flag ("n": normal, "o": out of nominal range,
                                 "u": u10n<0, "q":q10n<0
                                 "m": missing,
                                 "l": Rib<-0.5 or Rib>0.2 or z/L>1000,
                                 "r" : rh>100%,
                                 "t" : t10n<173K or t10n>373K
                                 "i": convergence fail at n)

    2021 / Author S. Biri
    2021 / Restructured by R. Cornes
    2021 / Simplified by E. Kent
    """
    logging.basicConfig(filename='flux_calc.log', filemode="w",
                        format='%(asctime)s %(message)s', level=logging.INFO)
    logging.captureWarnings(True)

    iclass = globals()[meth]()
    iclass.add_gust(gust=gust)
    iclass.add_variables(spd, T, SST, lat=lat, hum=hum, P=P, L=L)
    iclass.get_heights(hin, hout)
    iclass.get_specHumidity(qmeth=qmeth)
    iclass.set_coolskin_warmlayer(wl=wl, cskin=cskin, skin=skin, Rl=Rl, Rs=Rs)
    iclass.iterate(tol=tol, maxiter=maxiter)
    resAll = iclass.get_output(out=out)

    return resAll
