import numpy as np
import pandas as pd
from .base import model_base, Catm
from scipy.interpolate import UnivariateSpline
from ..tools import ImplicitParametersOutOfRange

import pkg_resources

models_list = []

global_error = 0.5
global_limit = (1e-99, 1e3)

default_path = \
    (pkg_resources.
     resource_filename(__name__,
                       "../data/kudryavtsev_et_al_1993_table_2.xlsx")
     )


class Liver(model_base):
    def __init__(self, path=default_path):
        self.Catm = Catm(delay=1)
        ploidy_data = pd.read_excel(path)
        ploidy_data['age'] = (ploidy_data[['age_min', 'age_max']].
                              mean(axis='columns'))
        ploidy_data /= 100.0
        ploidy_data['age'] *= 100

        self.ploidy = UnivariateSpline(ploidy_data['age'].values,
                                       ploidy_data['2C_mean'].values,
                                       ext=3, k=2)
        self.dtploidy = self.ploidy.derivative()
        self.ploidy2x2 = UnivariateSpline(ploidy_data['age'].values,
                                          ploidy_data['2Cx2_mean'].values,
                                          ext=3, k=2)
        self.dtploidy2x2 = self.ploidy2x2.derivative()

    def measurement_model(self, result_sim, data):
        mask_n2n4 = data.df.ploidy == '2n4n'
        mask_n2 = data.df.ploidy == '2n'
        mask_n4 = data.df.ploidy == '4n'

        wn2 = np.empty_like(data.age) * np.nan
        wn4 = np.empty_like(data.age) * np.nan

        wn2[mask_n2n4] = self.ploidy(data.age[mask_n2n4])
        wn2[mask_n2] = self.ploidy(data.age[mask_n2])
        wn2[mask_n4] = 0

        wn4[mask_n2n4] = 2*(1-self.ploidy(data.age[mask_n2n4]))
        wn4[mask_n2] = 2*self.ploidy2x2(data.age[mask_n2])
        wn4[mask_n4] = 1

        average = ((result_sim['n2']*wn2 + result_sim['n4']*wn4)
                   / np.sum([wn2, wn4], axis=0)
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
        dtploidy = self.dtploidy

        Dbirth = self.Dbirth

        M_new[self.n2] = -(M[self.n2]*(delta2 + kappa24)) - 2*M[self.n4]*kappa42 + r2*self.Catm.lin(Dbirth + t) + (2*M[self.n4]*kappa42)/ploidy(t) - (M[self.n2]*dtploidy(t))/ploidy(t)
        M_new[self.n4] = -(M[self.n4]*delta4) - M[self.n4]*kappa42 + (M[self.n4]*dtploidy(t))/(1 - ploidy(t)) + (M[self.n2]*kappa24*ploidy(t))/(2 - 2*ploidy(t)) + self.Catm.lin(Dbirth + t)*(r4 + (kappa24*ploidy(t))/(2 - 2*ploidy(t)))
        M_new[self.q] = 0

        return M_new.ravel()


class A(Liver):
    """ Liver model,
        r2 and kappa24 are implicitely time dependent
    """

    def __init__(self):
        default_parameters = dict(r4=-1.0, kappa42=-1.0,
                                  delta2=-1.0, delta4=-1.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r4 = self.r4
        kappa42 = self.kappa42
        delta2 = self.delta2
        delta4 = self.delta4

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        r2 = -((-delta4 + kappa42 + r4 - delta2*ploidy(t) + delta4*ploidy(t) - kappa42*ploidy(t) - r4*ploidy(t))/ploidy(t))
        kappa24 = -((-delta4 - kappa42 + r4 + dtploidy(t) + delta4*ploidy(t) + kappa42*ploidy(t) - r4*ploidy(t))/ploidy(t))

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        iparas = {'r2': r2, 'kappa24': kappa24,
                  'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                  'ratio_kappa': ratio_kappa}

        if r2 < 0:
            raise ImplicitParametersOutOfRange('r2 is negative', iparas)
        if kappa24 < 0:
            raise ImplicitParametersOutOfRange('kappa24 is negative', iparas)
        return iparas


class Al4s(A):
    """ Liver model,
        r2 and kappa24 are implicitely time dependent,
        r4 < 0.1
    """

    def __init__(self):
        default_parameters = dict(r4=-1.0, kappa42=-1.0,
                                  delta2=-1.0, delta4=-1.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        limit['r4'] = (1e-99, 0.1)
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)


class Ak0(Liver):
    """ Liver model,
        r2 and kappa24 are implicitely time dependent,
        kappa42 = 0
    """

    def __init__(self):
        default_parameters = dict(r4=-2.0, delta2=-1.0, delta4=-1.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r4 = self.r4
        delta2 = self.delta2
        delta4 = self.delta4

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        r2 = -((-delta4 + r4 - delta2*ploidy(t) + delta4*ploidy(t) - r4*ploidy(t))/ploidy(t))
        kappa24 = -((-delta4 + r4 + dtploidy(t) + delta4*ploidy(t) - r4*ploidy(t))/ploidy(t))
        kappa42 = 0

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if r2 < 0:
            raise ValueError('r2 is negative')
        if kappa24 < 0:
            raise ValueError('kappa24 is negative')
        return {'r2': r2, 'kappa24': kappa24, 'kappa42': kappa42,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Ar40(Liver):
    """ Liver model,
        r2 and kappa24 are implicitely time dependent,
        r4 = 0
    """

    def __init__(self):
        default_parameters = dict(kappa42=-1.0, delta2=-1.0, delta4=-1.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        kappa42 = self.kappa42
        delta2 = self.delta2
        delta4 = self.delta4

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        r2 = -((-delta4 + kappa42 - delta2*ploidy(t) + delta4*ploidy(t) - kappa42*ploidy(t))/ploidy(t))
        kappa24 = -((-delta4 - kappa42 + dtploidy(t) + delta4*ploidy(t) + kappa42*ploidy(t))/ploidy(t))
        r4 = 0

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if r2 < 0:
            raise ValueError('r2 is negative')
        if kappa24 < 0:
            raise ValueError('kappa24 is negative')
        return {'r2': r2, 'kappa24': kappa24, 'r4': r4,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Akr40(Liver):
    """ Liver model,
        r2 and kappa24 are implicitely time dependent,
        r4 = 0, kappa42 = 0
    """

    def __init__(self):
        default_parameters = dict(delta2=-1.0, delta4=-1.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        delta2 = self.delta2
        delta4 = self.delta4

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        r2 = -((-delta4 - delta2*ploidy(t) + delta4*ploidy(t))/ploidy(t))
        kappa24 = -((-delta4 + dtploidy(t) + delta4*ploidy(t))/ploidy(t))
        r4 = 0
        kappa42 = 0

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if r2 < 0:
            raise ValueError('r2 is negative')
        if kappa24 < 0:
            raise ValueError('kappa24 is negative')
        return {'r2': r2, 'kappa24': kappa24, 'r4': r4,
                'kappa42': kappa42,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Akr40_d2lin(Liver):
    """ Liver model,
        r2 and kappa24 are implicitely time dependent,
        r4 = 0, kappa42 = 0
        delta2 is linear in patient age
    """

    def __init__(self):
        default_parameters = dict(delta2_0=-1.0, delta2_100=-1.0, delta4=-1.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        delta2_0 = self.delta2_0
        delta2_100 = self.delta2_100
        delta4 = self.delta4

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        delta2 = np.interp(t, [0, 100], [delta2_0, delta2_100])
        r2 = -((-delta4 - delta2*ploidy(t) + delta4*ploidy(t))/ploidy(t))
        kappa24 = -((-delta4 + dtploidy(t) + delta4*ploidy(t))/ploidy(t))
        r4 = 0
        kappa42 = 0

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if r2 < 0:
            raise ValueError('r2 is negative')
        if kappa24 < 0:
            raise ValueError('kappa24 is negative')
        return {'r2': r2, 'kappa24': kappa24, 'r4': r4,
                'kappa42': kappa42, 'delta2': delta2,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Akr40q(Akr40):
    """ Liver model,
        r2 and kappa24 are implicitely time dependent,
        r4 = 0, kappa42 = 0
        quiescent population
    """

    def __init__(self):
        default_parameters = dict(delta2=-1.0, delta4=-1.0, f=1.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def transform_fit_to_physical(self, p_fit):
        p_phy = p_fit.copy()
        for p in p_fit.keys():
            p_phy[p] = 10**p_fit[p]
        p_phy['f'] = 0.5 * (np.tanh(p_fit['f'])+1)
        return p_phy

    def transform_physical_to_fit(self, p_phy):
        p_fit = p_phy.copy()
        for p in p_phy.keys():
            p_fit[p] = np.log10(p_phy[p])
        p_fit['f'] = -np.arctanh(1 - 2*p_phy['f'])
        return p_fit

    def measurement_model(self, result_sim, data):
        mask_n2 = data.df.ploidy == '2n'
        mask_n4 = data.df.ploidy == '4n'

        wn2 = self.ploidy(data.age)
        wn2[mask_n4] = 0

        wn4 = 2*(1-wn2)
        wn4[mask_n2] = 2*self.ploidy2x2(data.age[mask_n2])
        wn4[mask_n4] = 1

        result_sim_n2 = self.f*result_sim['n2'] + (1-self.f)*result_sim['q']

        average = ((result_sim_n2*wn2 + result_sim['n4']*wn4)
                   / np.sum([wn2, wn4], axis=0)
                   )

        return average


class Akr402x2n(Akr40):
    """ Liver model,
        r2 and kappa24 are implicitely time dependent,
        r4 = 0, kappa42 = 0,
        2x2n cells behave like 2n cells
    """

    def measurement_model(self, result_sim, data):
        mask_n2n4 = data.df.ploidy == '2n4n'
        mask_n2 = data.df.ploidy == '2n'
        mask_n4 = data.df.ploidy == '4n'

        wn2 = np.empty_like(data.age) * np.nan
        wn4 = np.empty_like(data.age) * np.nan

        wn2[mask_n2n4] = (self.ploidy(data.age[mask_n2n4])
                          + 2*self.ploidy2x2(data.age[mask_n2n4]))
        wn2[mask_n2] = (self.ploidy(data.age[mask_n2])
                        + 2*self.ploidy2x2(data.age[mask_n2]))
        wn2[mask_n4] = 0

        wn4[mask_n2n4] = 2*(1-self.ploidy(data.age[mask_n2n4])
                            - self.ploidy2x2(data.age[mask_n2n4]))
        wn4[mask_n2] = 0
        wn4[mask_n4] = 1

        average = ((result_sim['n2']*wn2 + result_sim['n4']*wn4)
                   / np.sum([wn2, wn4], axis=0)
                   )

        return average


class Ar2r4(Liver):
    """ Liver model,
        r2 and kappa24 are implicitely time dependent,
        lambda2 = r4
    """

    def __init__(self):
        default_parameters = dict(kappa42=-1.0, delta2=-1.0, delta4=-1.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        kappa42 = self.kappa42
        delta2 = self.delta2
        delta4 = self.delta4

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        r2 = -((-2*kappa42 + dtploidy(t) + delta2*ploidy(t) + delta4*ploidy(t) + 3*kappa42*ploidy(t) - dtploidy(t)*ploidy(t) - delta4*ploidy(t)**2 - kappa42*ploidy(t)**2)/((-2 + ploidy(t))*ploidy(t)))
        kappa24 = -((2*kappa42 - dtploidy(t) - delta2*ploidy(t) + delta4*ploidy(t) - 3*kappa42*ploidy(t) + delta2*ploidy(t)**2 - delta4*ploidy(t)**2 + kappa42*ploidy(t)**2)/((-2 + ploidy(t))*ploidy(t)))
        r4 = -((2*delta4 - dtploidy(t) + delta2*ploidy(t) - 2*delta4*ploidy(t))/(-2 + ploidy(t)))

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if r2 < 0:
            raise ValueError('r2 is negative')
        if kappa24 < 0:
            raise ValueError('kappa24 is negative')
        if r4 < 0:
            raise ValueError('r4 is negative')
        return {'r2': r2, 'kappa24': kappa24, 'r4': r4,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class B(Liver):
    """ Liver model,
        delta2 and delta4 are implicitely time dependent
    """

    def __init__(self):
        default_parameters = dict(r2=-1.0, r4=-1.0,
                                  kappa24=-1.0, kappa42=-1.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r2 = self.r2
        r4 = self.r4
        kappa24 = self.kappa24
        kappa42 = self.kappa42

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        delta2 = -((-2*kappa42 + dtploidy(t) + kappa24*ploidy(t) + 2*kappa42*ploidy(t) - r2*ploidy(t))/ploidy(t))
        delta4 = -((-kappa42 + r4 + dtploidy(t) + kappa24*ploidy(t) + kappa42*ploidy(t) - r4*ploidy(t))/(-1 + ploidy(t)))

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if delta2 < 0:
            raise ValueError('delta2 is negative')
        if delta4 < 0:
            raise ValueError('delta4 is negative')
        return {'delta2': delta2, 'delta4': delta4,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Bk0(Liver):
    """ Liver model,
        delta2 and delta4 are implicitely time dependent
        kappa42 = 0
    """

    def __init__(self):
        default_parameters = dict(r2=-1.0, r4=-1.0,
                                  kappa24=-2.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r2 = self.r2
        r4 = self.r4
        kappa24 = self.kappa24

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        delta2 = -((dtploidy(t) + kappa24*ploidy(t) - r2*ploidy(t))/ploidy(t))
        delta4 = -((r4 + dtploidy(t) + kappa24*ploidy(t) - r4*ploidy(t))/(-1 + ploidy(t)))
        kappa42 = 0

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if delta2 < 0:
            raise ValueError('delta2 is negative')
        if delta4 < 0:
            raise ValueError('delta4 is negative')
        return {'delta2': delta2, 'delta4': delta4, 'kappa42': kappa42,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Br40(Liver):
    """ Liver model,
        delta2 and delta4 are implicitely time dependent
        lambda 4 = 0
    """

    def __init__(self):
        default_parameters = dict(r2=-1.0, kappa24=-1.0, kappa42=-1.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r2 = self.r2
        kappa24 = self.kappa24
        kappa42 = self.kappa42

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        delta2 = -((-2*kappa42 + dtploidy(t) + kappa24*ploidy(t) + 2*kappa42*ploidy(t) - r2*ploidy(t))/ploidy(t))
        delta4 = -((-kappa42 + dtploidy(t) + kappa24*ploidy(t) + kappa42*ploidy(t))/(-1 + ploidy(t)))
        r4 = 0

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if delta2 < 0:
            raise ValueError('delta2 is negative')
        if delta4 < 0:
            raise ValueError('delta4 is negative')
        return {'delta2': delta2, 'delta4': delta4, 'r4': r4,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Bkr40(Liver):
    """ Liver model,
        delta2 and delta4 are implicitely time dependent,
        r4 = 0, kappa42 = 0
    """

    def __init__(self, r2=-1.0, kappa24=-1):
        default_parameters = dict(r2=r2, kappa24=kappa24)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r2 = self.r2
        kappa24 = self.kappa24

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        delta2 = -((dtploidy(t) + kappa24*ploidy(t) - r2*ploidy(t))/ploidy(t))
        delta4 = -((dtploidy(t) + kappa24*ploidy(t))/(-1 + ploidy(t)))
        r4 = 0
        kappa42 = 0

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if delta2 < 0:
            raise ValueError('delta2 is negative')
        if delta4 < 0:
            raise ValueError('delta4 is negative')
        return {'delta2': delta2, 'delta4': delta4,
                'r4': r4, 'kappa42': kappa42,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Bkr40_rlin(Liver):
    """ Liver model,
        delta2 and delta4 are implicitely time dependent,
        r4 = 0, kappa42 = 0
        r2(t) is linear in patient age
    """

    def __init__(self):
        default_parameters = dict(r2_0=-1.0, r2_100=-1.0, kappa24=-1)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r2_0 = self.r2_0
        r2_100 = self.r2_100
        kappa24 = self.kappa24

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        r2 = np.interp(t, [0, 100], [r2_0, r2_100])
        delta2 = -((dtploidy(t) + kappa24*ploidy(t) - r2*ploidy(t))/ploidy(t))
        delta4 = -((dtploidy(t) + kappa24*ploidy(t))/(-1 + ploidy(t)))
        r4 = 0
        kappa42 = 0

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if delta2 < 0:
            raise ValueError('delta2 is negative')
        if delta4 < 0:
            raise ValueError('delta4 is negative')
        return {'r2': r2, 'delta2': delta2, 'delta4': delta4,
                'r4': r4, 'kappa42': kappa42,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Bkr40_rstep(Liver):
    """ Liver model,
        delta2 and delta4 are implicitely time dependent,
        r4 = 0, kappa42 = 0
        r2(t) is linear in patient age
    """

    def __init__(self):
        default_parameters = dict(r2_0=-1.0, r2_1=-1.0, t_step=np.log10(50),
                                  kappa24=-1)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        limit['t_step'] = (1e-99, 1e100)
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r2_0 = self.r2_0
        r2_1 = self.r2_1
        t_step = self.t_step
        kappa24 = self.kappa24

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        r2 = np.interp(t, [0, t_step, t_step, t_step + 100],
                      [r2_0, r2_0, r2_1, r2_1])
        delta2 = -((dtploidy(t) + kappa24*ploidy(t) - r2*ploidy(t))/ploidy(t))
        delta4 = -((dtploidy(t) + kappa24*ploidy(t))/(-1 + ploidy(t)))
        r4 = 0
        kappa42 = 0

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if delta2 < 0:
            raise ValueError('delta2 is negative')
        if delta4 < 0:
            raise ValueError('delta4 is negative')
        return {'r2': r2, 'delta2': delta2, 'delta4': delta4,
                'r4': r4, 'kappa42': kappa42,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Br2r4(Br40):
    """ Liver model,
        delta2 and delta4 are implicitely time dependent,
        r4 = lambda2, kappa42 = 0
    """

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r2 = self.r2
        kappa24 = self.kappa24
        kappa42 = self.kappa42

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        delta2 = -((-2*kappa42 + dtploidy(t) + kappa24*ploidy(t) + 2*kappa42*ploidy(t) - r2*ploidy(t))/ploidy(t))
        delta4 = -((kappa24 - kappa42 + r2 + dtploidy(t) + kappa42*ploidy(t) - r2*ploidy(t))/(-1 + ploidy(t)))
        r4 = kappa24 + r2

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if delta2 < 0:
            raise ValueError('delta2 is negative')
        if delta4 < 0:
            raise ValueError('delta4 is negative')
        return {'delta2': delta2, 'delta4': delta4,
                'r4': r4,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class C(Liver):
    """ Liver model,
        kappa24 and delta4 are implicitely time dependent
    """

    def __init__(self):
        default_parameters = dict(r2=-1.0, r4=-1.0,
                                  kappa42=-1.0, delta2=-1.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r2 = self.r2
        r4 = self.r4
        kappa42 = self.kappa42
        delta2 = self.delta2

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        kappa24 = (2*kappa42 - dtploidy(t) - delta2*ploidy(t) - 2*kappa42*ploidy(t) + r2*ploidy(t))/ploidy(t)
        delta4 = (-kappa42 - r4 + delta2*ploidy(t) + kappa42*ploidy(t) + r4*ploidy(t) - r2*ploidy(t))/(-1 + ploidy(t))

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if kappa24 < 0:
            raise ValueError('kappa24 is negative')
        if delta4 < 0:
            raise ValueError('delta4 is negative')
        return {'kappa24': kappa24, 'delta4': delta4,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Ck0(Liver):
    """ Liver model,
        kappa24 and delta4 are implicitely time dependent
        kappa42 = 0
    """

    def __init__(self):
        default_parameters = dict(r2=-1.0, r4=-1.0,
                                  delta2=-2.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r2 = self.r2
        r4 = self.r4
        delta2 = self.delta2

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        kappa24 = -((dtploidy(t) + delta2*ploidy(t) - r2*ploidy(t))/ploidy(t))
        delta4 = -((r4 - delta2*ploidy(t) - r4*ploidy(t) + r2*ploidy(t))/(-1 + ploidy(t)))
        kappa42 = 0

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if r2 < 0:
            raise ValueError('r2 is negative')
        if kappa24 < 0:
            raise ValueError('kappa24 is negative')
        return {'kappa24': kappa24, 'delta4': delta4, 'kappa42': kappa42,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Cr40(Liver):
    """ Liver model,
        kappa24 and delta4 are implicitely time dependent
        r4 = 0
    """

    def __init__(self):
        default_parameters = dict(r2=-1.0,
                                  kappa42=-2.0, delta2=-2.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r2 = self.r2
        kappa42 = self.kappa42
        delta2 = self.delta2

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        kappa24 = -((-2*kappa42 + dtploidy(t) + delta2*ploidy(t) + 2*kappa42*ploidy(t) - r2*ploidy(t))/ploidy(t))
        delta4 = -((kappa42 - delta2*ploidy(t) - kappa42*ploidy(t) + r2*ploidy(t))/(-1 + ploidy(t)))
        r4 = 0

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if kappa24 < 0:
            raise ValueError('kappa24 is negative')
        if delta4 < 0:
            raise ValueError('delta4 is negative')
        return {'kappa24': kappa24, 'delta4': delta4, 'r4': r4,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Ckr40(Liver):
    """ Liver model,
        kappa24 and delta4 are implicitely time dependent
        kappa42 = 0 and r4 = 0
    """

    def __init__(self):
        default_parameters = dict(r2=-1.0, delta2=-2.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r2 = self.r2
        delta2 = self.delta2

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        kappa24 = -((dtploidy(t) + delta2*ploidy(t) - r2*ploidy(t))/ploidy(t))
        delta4 = -((-(delta2*ploidy(t)) + r2*ploidy(t))/(-1 + ploidy(t)))
        r4 = 0
        kappa42 = 0

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if kappa24 < 0:
            raise ValueError('kappa24 is negative')
        if delta4 < 0:
            raise ValueError('delta4 is negative')
        return {'kappa24': kappa24, 'delta4': delta4, 'r4': r4,
                'kappa42': kappa42,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Cr2r4(Cr40):
    """ Liver model,
        kappa24 and delta4 are implicitely time dependent
        r4 = 0
    """

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r2 = self.r2
        kappa42 = self.kappa42
        delta2 = self.delta2

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        kappa24 = -((-2*kappa42 + dtploidy(t) + delta2*ploidy(t) + 2*kappa42*ploidy(t) - r2*ploidy(t))/ploidy(t))
        delta4 = -((2*kappa42 - dtploidy(t) - delta2*ploidy(t) - 3*kappa42*ploidy(t) + 2*r2*ploidy(t) + dtploidy(t)*ploidy(t) + kappa42*ploidy(t)**2 - r2*ploidy(t)**2)/((-1 + ploidy(t))*ploidy(t)))
        r4 = -((-2*kappa42 + dtploidy(t) + delta2*ploidy(t) + 2*kappa42*ploidy(t) - 2*r2*ploidy(t))/ploidy(t))

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if kappa24 < 0:
            raise ValueError('kappa24 is negative')
        if delta4 < 0:
            raise ValueError('delta4 is negative')
        return {'kappa24': kappa24, 'delta4': delta4, 'r4': r4,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class H(Liver):
    """ Liver model,
        kappa24 is implicitely time dependent
        r2 == delta2, r4 == delta4 is assumed
    """

    def __init__(self, r2=-1, r4=-1):
        default_parameters = dict(r2=r2, r4=r4)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r2 = self.r2
        r4 = self.r4

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        delta2 = r2
        delta4 = r4
        kappa24 = -(dtploidy(t)/ploidy(t))
        kappa42 = 0

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24


        if kappa24 < 0:
            raise ValueError('kappa24 is negative')
        return {'delta2': delta2, 'delta4': delta4,
                'kappa24': kappa24, 'kappa42': kappa42,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


class Hr40(Liver):
    """ Liver model,
        kappa24 is implicitely time dependent
        r2 == delta2, r4 == delta4 is assumed
    """

    def __init__(self):
        default_parameters = dict(r2=-1.0)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        Liver.__init__(self)
        model_base.__init__(self, var_names=['n2', 'n4', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        r2 = self.r2

        ploidy = self.ploidy
        dtploidy = self.dtploidy

        # calculate implicit parameters
        delta2 = r2
        delta4 = 0
        kappa24 = -(dtploidy(t)/ploidy(t))
        kappa42 = 0
        r4 = 0

        lambda2 = r2 + kappa24
        p = r2 / lambda2
        ratio_r = r4 / r2
        ratio_kappa = kappa42 / kappa24

        if r2 < 0:
            raise ValueError('r2 is negative')
        if kappa24 < 0:
            raise ValueError('kappa24 is negative')
        return {'delta2': delta2, 'delta4': delta4,
                'kappa24': kappa24, 'kappa42': kappa42, 'r4': r4,
                'lambda2': lambda2, 'p': p, 'ratio_r': ratio_r,
                'ratio_kappa': ratio_kappa}


models_list = [A, Al4s, Ak0, Ar40, Akr40, Akr40_d2lin, Akr40q, Akr402x2n,
               Ar2r4,
               B, Bk0, Br40, Bkr40, Bkr40_rlin, Bkr40_rstep, Br2r4,
               C, Ck0, Cr40, Ckr40, Cr2r4, H, Hr40]
