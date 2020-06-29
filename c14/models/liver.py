import numpy as np
import pandas as pd
from .base import model_base, Catm
from scipy.interpolate import UnivariateSpline
from ..tools import ImplicitParametersOutOfRange
from .minimal import POP1, POP1q

import pkg_resources

models_list = []

global_error = 0.5
global_limit = (10**-6, 10**1)
kappa_limit = (1e-3, 10)

default_path = \
    (pkg_resources.
     resource_filename(__name__,
                       "../data/kudryavtsev_et_al_1993_table_2.xlsx")
     )


class Hepatocyte(model_base):
    def __init__(self, path=default_path):
        self.Catm = Catm(delay=1)
        ploidy_data = pd.read_excel(path)
        ploidy_data['age'] = (ploidy_data[['age_min', 'age_max']].
                              mean(axis='columns'))
        ploidy_data /= 100.0
        ploidy_data['age'] *= 100

        self.ploidy = UnivariateSpline(ploidy_data['age'].values,
                                       ploidy_data['2C_mean'].values,
                                       ext=0, k=2)
        self.dtploidy = self.ploidy.derivative()
        self.ploidy2x2 = UnivariateSpline(ploidy_data['age'].values,
                                          ploidy_data['2Cx2_mean'].values,
                                          ext=0, k=2)
        self.dtploidy2x2 = self.ploidy2x2.derivative()

    def measurement_model(self, result_sim, data):
        mask_n2n4 = data.df.ploidy == '2n4n'
        mask_n2 = data.df.ploidy == '2n'
        mask_n4 = data.df.ploidy == '4n'

        try:
            f = self.f
        except AttributeError:
            f = 1.0

        wn2 = np.empty_like(data.age) * np.nan
        wn4 = np.empty_like(data.age) * np.nan
        wq = np.empty_like(data.age) * np.nan

        wn2[mask_n2n4] = f * (self.ploidy(data.age[mask_n2n4])
                              / (self.ploidy(data.age[mask_n2n4])
                                 + 2*(1-self.ploidy(data.age[mask_n2n4]))))
        wn2[mask_n2] = f * (self.ploidy(data.age[mask_n2])
                            / (self.ploidy(data.age[mask_n2])
                               + 2*self.ploidy2x2(data.age[mask_n2])))
        wn2[mask_n4] = f * 0

        wn4[mask_n2n4] = f * (2*(1-self.ploidy(data.age[mask_n2n4]))
                              / (self.ploidy(data.age[mask_n2n4])
                                 + 2*(1-self.ploidy(data.age[mask_n2n4]))))
        wn4[mask_n2] = f * (2*self.ploidy2x2(data.age[mask_n2])
                            / (self.ploidy(data.age[mask_n2])
                               + 2*self.ploidy2x2(data.age[mask_n2])))
        wn4[mask_n4] = f * 1

        wq[mask_n2n4] = 1-f
        wq[mask_n2] = 1-f
        wq[mask_n4] = 1-f

        average = (result_sim['n2']*wn2
                   + result_sim['n4']*wn4
                   + result_sim['q']*wq
                   )

        return average

    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        iparas = self.calc_implicit_parameters(t)
        try:
            r2 = self.r2
        except AttributeError:
            r2 = iparas['r2']
        try:
            r4 = self.r4
        except AttributeError:
            r4 = iparas['r4']
        try:
            kappa24 = self.kappa24
        except AttributeError:
            kappa24 = iparas['kappa24']
        try:
            kappa42 = self.kappa42
        except AttributeError:
            kappa42 = iparas['kappa42']
        try:
            delta2 = self.delta2
        except AttributeError:
            delta2 = iparas['delta2']
        try:
            delta4 = self.delta4
        except AttributeError:
            delta4 = iparas['delta4']

        rate_limit = 10.0
        r2 = r2 if r2 < rate_limit else rate_limit
        r4 = r4 if r4 < rate_limit else rate_limit
        kappa24 = kappa24 if kappa24 < rate_limit else rate_limit
        kappa42 = kappa42 if kappa42 < rate_limit else rate_limit
        delta2 = delta2 if delta2 < rate_limit else rate_limit
        delta4 = delta4 if delta4 < rate_limit else rate_limit

        ploidy = self.ploidy

        Dbirth = self.Dbirth

        M_new[self.n2] = 2*M[self.n2]*kappa42 - 2*M[self.n4]*kappa42 - M[self.n2]*r2 + r2*self.Catm.lin(Dbirth + t) + (2*(-M[self.n2] + M[self.n4])*kappa42)/ploidy(t)
        M_new[self.n4] = (2*M[self.n4]*r4 - (M[self.n2]*kappa24 - 2*M[self.n4]*kappa24 + 2*M[self.n4]*r4)*ploidy(t) + self.Catm.lin(Dbirth + t)*(-2*r4 - (kappa24 - 2*r4)*ploidy(t)))/(2.*(-1 + ploidy(t)))
        M_new[self.q] = 0

        return M_new.ravel()


class R(Hepatocyte):
    """ Liver model,
        r2 and r4 are implicitely time dependent
    """

    def __init__(self, kappa24=10**-3.0, kappa42=10**-2.0, delta2=10**-1.0, delta4=10**-1.0):
        default_parameters = dict(kappa24=kappa24, kappa42=kappa42,
                                  delta2=delta2, delta4=delta4)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        limit['kappa42'] = kappa_limit
        limit['kappa24'] = kappa_limit
        Hepatocyte.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        kappa24 = self.kappa24
        kappa42 = self.kappa42
        delta2 = self.delta2
        delta4 = self.delta4

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        r2 = delta2 + kappa24 + 2*kappa42 + (-2*kappa42 - (2*dtploidy(t))/(-2 + ploidy(t)))/ploidy(t)
        r4 = (-dtploidy(t) + (-2 + ploidy(t))*(-delta4 - kappa42 + (delta4 + kappa24 + kappa42)*ploidy(t)))/((-2 + ploidy(t))*(-1 + ploidy(t)))

        lambda2 = r2 + kappa24
        lambda4 = r4 + kappa42
        p2 = r2 / lambda2
        p4 = r4 / lambda4
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        f22 = r2 * ploidy(t)
        f42 = 2 * kappa42 * (1.0 - ploidy(t))
        f24 = kappa24 * ploidy(t)
        f44 = r4 * (1.0 - ploidy(t))

        rf22 = f22 / (f22 + 2 * f42)
        rf42 = f42 / (f22 + 2 * f42)
        rf24 = f24 / (f24 + f44)
        rf44 = f44 / (f24 + f44)

        ratio_f = f42 / f24

        iparas = {'r2': r2, 'r4': r4,
                  'lambda2': lambda2, 'lambda4': lambda4,
                  'ratio_r': ratio_r,
                  'ratio_kappa': ratio_kappa,
                  'p2': p2, 'p4': p4,
                  'f22': f22, 'f24': f24, 'f42': f42, 'f44': f44,
                  'rf22': rf22, 'rf24': rf24, 'rf42': rf42, 'rf44': rf44,
                  'ratio_f': ratio_f}

        if r2 < 0:
            raise ImplicitParametersOutOfRange(
                f'r2 ({r2})is negative at t = {t}', iparas)
        if r4 < 0:
            raise ImplicitParametersOutOfRange(
                f'r4 ({r4}) is negative at t = {t}', iparas)
        if kappa42 > r4:
            raise ImplicitParametersOutOfRange(
                f'kappa42 ({kappa42}) > r4 ({r4} at t={t})', iparas)
        return iparas


class Rq(R):
    """ Liver model,
        r2 and r4 are implicitely time dependent
        + quiescent population
    """

    def __init__(self, kappa24=10**-3.0, kappa42=10**-2.0, delta2=10**-1.0, delta4=10**-1.0,
                 f=0.9):
        default_parameters = dict(kappa24=kappa24, kappa42=kappa42,
                                  delta2=delta2, delta4=delta4,
                                  f=f)
        self.logparas = ['kappa24', 'kappa42', 'delta2', 'delta4']
        self.linparas = ['f']
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        limit['kappa42'] = kappa_limit
        limit['kappa24'] = kappa_limit
        limit['f'] = (0, 1)
        Hepatocyte.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

class POP2(POP1q):
    def __init__(self, lambda1=0.1, dlambda=0.1, f=0.5):
        default_parameters = dict(lambda1=lambda1, dlambda=dlambda, f=f)
        limit = {i: global_limit for i in default_parameters.keys()}
        self.logparas = ['lambda1', 'dlambda']
        self.linparas = ['f']
        limit['f'] = (0, 1)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['cells1', 'cells2'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)

    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        lambda1 = self.lambda1
        dlambda = self.dlambda

        M_new[self.cells1] = lambda1*(self.Catm.lin(t + self.Dbirth)
                                      - M[self.cells1])
        M_new[self.cells2] = (lambda1 + dlambda)*(self.Catm.lin(t + self.Dbirth)
                                      - M[self.cells2])

        return M_new.ravel()

    def calc_implicit_parameters(self, t):
        lambda2 = self.lambda1 + self.dlambda
        return {'lambda2': lambda2}

    def measurement_model(self, result_sim, data):
        return self.f*result_sim['cells1'] + (1-self.f)*result_sim['cells2']


models_list = [R, POP1q, POP1,POP2]