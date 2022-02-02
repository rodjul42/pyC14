import numpy as np
from .base import model_base, Catm
from ..tools import trans_arcsin, trans_sin

global_error = 0.5

limit_rate = (1e-6, 0.5)
limit_f = (0.0, 1.0)
limit_s = (0, 100)


class N(model_base):
    """ No turnover at all
    """

    def __init__(self):
        default_parameters = {}
        limit = {}
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
        return {'beta': 0.0, 'gamma': 0.0}

    def measurement_model(self, result_sim, data):
        return result_sim['cells']


class A(model_base):
    def __init__(self):
        default_parameters = {'beta': 0.1}
        limit = {'beta': limit_rate}
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['cells'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)

    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        beta = self.beta
        if beta > 10.0:
            beta = 10.0

        M_new[self.cells] = beta*(self.Catm.lin(t + self.Dbirth)
                                  - M[self.cells])

        return M_new.ravel()

    def calc_implicit_parameters(self, t):
        gamma = self.beta
        return {'gamma': gamma}

    def measurement_model(self, result_sim, data):
        return result_sim['cells']


class x2POP(model_base):
    def __init__(self):
        default_parameters = {'beta': 0.1, 'f': 0.3}
        self.logparas = ['beta']
        self.linparas = ['f']
        limit = {'beta': limit_rate,
                 'f': limit_f}
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['cells', 'q'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)


    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        beta = self.beta
        if beta > 10.0:
            beta = 10.0

        M_new[self.cells] = beta*(self.Catm.lin(t + self.Dbirth)
                                  - M[self.cells])
        M_new[self.q] = 0

        return M_new.ravel()

    def calc_implicit_parameters(self, t):
        delta = self.beta
        return {'delta': delta}

    def measurement_model(self, result_sim, data):
        return self.f*result_sim['cells'] + (1-self.f)*result_sim['q']


class S(model_base):
    def __init__(self):
        default_parameters = {'beta': 0.1,
                              's': 20}
        self.logparas = ['beta']
        self.linparas = ['s']
        limit = {'beta': limit_rate,
                 's': limit_s}
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['cells'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)



    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        iparas = self.calc_implicit_parameters(t)
        betat = iparas['betat']

        M_new[self.cells] = betat * (self.Catm.lin(t + self.Dbirth)
                                     - M[self.cells])

        return M_new.ravel()

    def calc_implicit_parameters(self, t):
        beta = self.beta
        if beta > 10.0:
            beta = 10.0

        s = self.s

        betat = np.interp(t,
                            [0, s, s, 100],
                            [beta, beta, 0, 0])

        return {'betat': betat}

    def measurement_model(self, result_sim, data):
        return result_sim['cells']


class SB(S):
    def __init__(self):
        S.__init__(self)
        self.limit['s'] = (0,100)

class S2(model_base):
    def __init__(self):
        default_parameters = {'beta': 0.1,
                              's': 20, 'f': 0.3}
        self.logparas = ['beta']
        self.linparas = ['s', 'f']
        limit = {'beta': limit_rate,
                 's': limit_s,
                 'f': limit_f}
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['cells', 'q'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)



    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        iparas = self.calc_implicit_parameters(t)
        betat = iparas['betat']

        M_new[self.cells] = betat * (self.Catm.lin(t + self.Dbirth)
                                     - M[self.cells])
        M_new[self.q] = 0

        return M_new.ravel()

    def calc_implicit_parameters(self, t):
        beta = self.beta
        if beta > 10.0:
            beta = 10.0
        s = self.s

        betat = np.interp(t, [0, s, s, 100], [beta, beta, 0, 0])

        return {'betat': betat}

    def measurement_model(self, result_sim, data):
        return self.f*result_sim['cells'] + (1-self.f)*result_sim['q']


class S2B(S2):
    def __init__(self):
        S2.__init__(self)
        self.limit['s'] = (0,100)

class Lin(S):
    def __init__(self):
        default_parameters = {'beta0': 0.1,
                              'beta100': 0.0001}
        self.logparas = ['beta0','beta100']
        self.linparas = []
        limit = {'beta0': limit_rate,
                 'beta100': limit_rate}
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['cells'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)



    def calc_implicit_parameters(self, t):
        beta0 = self.beta0
        beta100 = self.beta100

        betat = np.interp(t,
                            [0, 100],
                            [beta0, beta100])

        return {'betat': betat}

class Lin2(S2):
    def __init__(self):
        default_parameters = {'beta0': 0.1,
                            'beta100': 0.0001,
                                  'f': 0.3}
        self.logparas = ['beta0','beta100']
        self.linparas = ['f']
        limit = {'beta0': limit_rate,
                 'beta100': limit_rate,
                 'f': limit_f}
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['cells', 'q'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)


    def calc_implicit_parameters(self, t):
        beta0 = self.beta0
        beta100 = self.beta100

        betat = np.interp(t,
                            [0, 100],
                            [beta0, beta100])
        return {'betat': betat}


class LinB(S):
    def __init__(self):
        default_parameters = {'beta0': 0.1,
                              'beta10': 0.1}
        self.logparas = ['beta0','beta10']
        self.linparas = []
        limit = {'beta0': limit_rate,
                 'beta10': limit_rate}
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['cells'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)



    def calc_implicit_parameters(self, t):
        beta0 = self.beta0
        beta10 = self.beta10
        betat = beta0 + (beta10-beta0)/10.0*t
        if betat>limit_rate[1]:
            betat=limit_rate[1]
        elif betat<0:
            betat=0
        return {'betat': betat}

class LinB2(S2):
    def __init__(self):
        default_parameters = {'beta0': 0.1,
                            'beta10': 0.1,
                                  'f': 0.3}
        self.logparas = ['beta0','beta10']
        self.linparas = ['f']
        limit = {'beta0': limit_rate,
                 'beta10': limit_rate,
                 'f': limit_f}
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['cells', 'q'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)



    def calc_implicit_parameters(self, t):
        beta0 = self.beta0
        beta10 = self.beta10
        betat = beta0 + (beta10-beta0)/10.0*t
        if betat>limit_rate[1]:
            betat=limit_rate[1]
        elif betat<0:
            betat=0
        return {'betat': betat}


import pandas as pd
from scipy.special import erf
from scipy.stats import lognorm

def log_params( mu, sigma):
        """ A transformation of paramteres such that mu and sigma are the 
            mean and variance of the log-normal distribution and not of
            the underlying normal distribution.
        """
        s2 = np.log(1.0 + sigma**2/mu**2)
        m = np.log(mu) - 0.5 * s2
        s = np.sqrt(s2)
        return m, s

def cdfl( X, mu, sigma):
        ''' lognormal pdf with actual miu and sigma
        '''
        #mu_tmp, sigma_tmp = log_params(mu, sigma)
        #return lognorm.cdf(X, s=sigma_tmp, scale = np.exp(mu_tmp))
        return (1+erf((X-mu)/(sigma*1.4142135623730951)))*0.5


def cdf(x,mu,sigma):
    return (1+erf((x-mu)/(sigma*1.4142135623730951)))*0.5


class x2POP_Gaus(model_base):
    def __init__(self,bins=100):
        default_parameters = {'beta_mean': 0.1, 'beta_width':0.01,'f': 0.9}
        self.logparas = ['beta_mean','beta_width']
        self.linparas = ['f']
        limit = {'beta_mean': limit_rate,
                 'f': limit_f,
                 'beta_width':(1e-5,10)}
        self.Catm = Catm(delay=1)
        self.max_age = 100
        self.bins = bins
        self.dist_vars = ['cells']
        model_base.__init__(self, var_names=['q','cells'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)
        num = 0
        for var_name in self.var_names:
            if var_name in self.dist_vars:
                self.__dict__.update({var_name+'_eff': slice(num,num+self.bins) })
                num=num+self.bins
            else:
                self.__dict__.update({var_name+'_eff': num })
                num=num+1
        self.nvars_eff = num


    def gen_init(self, Dbirth):
        if isinstance(Dbirth, pd.DataFrame):
            idx = Dbirth.index
        else:
            idx = None
        data = []
        for var_name in self.var_names:
            if var_name in self.dist_vars:
                data.append(pd.DataFrame(self.Catm.lin(Dbirth)[:, np.newaxis]*np.ones(self.bins), columns=pd.MultiIndex.from_product([[var_name],np.arange(self.bins)]), index=idx))
            else:
                data.append(pd.DataFrame(self.Catm.lin(Dbirth)[:, np.newaxis]*np.ones(1), columns=pd.MultiIndex.from_product([[var_name],[0]]), index=idx))
        return pd.concat(data,axis=1)


    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars_eff, -1))
        M_new = np.zeros_like(M)

        betas = self.betas
        M_new[self.cells_eff] = betas*(self.Catm.lin(t + self.Dbirth)
                                  - M[self.cells_eff])
        return M_new.ravel()


    def calc_initial_parameters(self):
        min_b =  max(limit_rate[0],self.beta_mean-3*self.beta_width)
        max_b =  min(limit_rate[1],self.beta_mean+3*self.beta_width)
        betas = np.linspace(min_b,max_b,self.bins)
        self.mask_betas = betas>10
        delta = betas[1]-betas[0]
        intb=betas+delta
        inta=betas-delta
        w =np.array([cdf(b,self.beta_mean,self.beta_width)-cdf(a,self.beta_mean,self.beta_width) for a,b in zip(inta,intb)])
        self.weights = w/w.sum()
        betas[self.mask_betas] = 0
        self.betas = betas[:,np.newaxis]

    def calc_implicit_parameters(self, t):
        return { }

    def measurement_model(self, result_sim, data):
        return self.f*result_sim['cells'] + (1-self.f)*result_sim['q']

    def bins_to_value(self,sol_res,d_len,t_len,Dcoll):
        newshape = sol_res.reshape(self.nvars_eff,d_len,t_len)
        reduced = np.zeros((self.nvars,d_len,t_len))
        if self.mask_betas.any():
            newshape[self.cells_eff][self.mask_betas] = self.Catm.lin(Dcoll)
        reduced[self.cells] = np.average(newshape[self.cells_eff],axis=0,weights=self.weights)
        reduced[self.q] = newshape[self.q_eff]
        return reduced.reshape(d_len*self.nvars,t_len)


#models_list = [N, A, x2POP, S, S2, SB, S2B, Lin, Lin2,LinB,LinB2,x2POP_Gaus]

models_list = [N, A, x2POP, SB, S2B, LinB,x2POP_Gaus]

