import numpy as np
from .base import model_base, Catm
from ..tools import trans_arcsin, trans_sin

global_error = 0.5
global_limit = (1e-6, 1e1)


class NOT(model_base):
    """ No turnover at all
    """

    def __init__(self):
        default_parameters = {}
        limit = {i: global_limit for i in default_parameters.keys()}
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['cells'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)

    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        M_new[self.cells] = 0

        return M_new.ravel()

    def calc_implicit_parameters(self, t):
        return {'lambda': 0.0, 'delta': 0.0}

    def measurement_model(self, result_sim, data):
        return result_sim['cells']


class POP1(model_base):
    populations_DNA = {'cells':1}
    m_types = ['mean']
    populations = ['cells']
    populations_m = {'mean':populations}
    iparas = []

    def __init__(self):
        default_parameters = {'lambda_': 0.5}
        limit = {i: global_limit for i in default_parameters.keys()}
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['cells'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)

    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        lambda_ = self.lambda_

        M_new[self.cells] = lambda_*(self.Catm.lin(t + self.Dbirth)
                                     - M[self.cells])

        return M_new.ravel()

    def calc_implicit_parameters(self, t):
        delta = self.lambda_
        return {'delta': delta,'cells':1}

    def measurement_model(self, result_sim, data):
        return result_sim['cells']


class POP1q(model_base):
    def __init__(self):
        default_parameters = {'lambda_': 0.1, 'f': 0.5}
        limit = {i: global_limit for i in default_parameters.keys()}
        self.logparas = ['lambda_']
        self.linparas = ['f']
        limit['f'] = (0, 1)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['cells', 'q'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)

    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        lambda_ = self.lambda_

        M_new[self.cells] = lambda_*(self.Catm.lin(t + self.Dbirth)
                                     - M[self.cells])
        M_new[self.q] = 0

        return M_new.ravel()

    def calc_implicit_parameters(self, t):
        delta = self.lambda_
        return {'delta': delta}

    def measurement_model(self, result_sim, data):
        return self.f*result_sim['cells'] + (1-self.f)*result_sim['q']


models_list = [NOT, POP1, POP1q]
