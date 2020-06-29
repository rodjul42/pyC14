import numpy as np
from .base import model_base, Catm
from ..tools import ImplicitParametersOutOfRange
from .minimal import POP1, POP1q


models_list = []

global_error = 0.5
global_limit = (10**-6, 10**1)


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


models_list = [POP1, POP1q, POP2]
