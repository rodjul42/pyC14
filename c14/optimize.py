import numpy as np
import pandas as pd
import scipy as sp
import scipy.integrate
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.stats import gaussian_kde, normaltest
import emcee
import copy
from multiprocessing import Pool

from .tools import RK45,ImplicitParametersOutOfRange


import logging
logger =  logging.getLogger(__name__)

__all__ = ['optimize']


class optimize():
    def __init__(self, model, data, step_size=0.1, rtol=0.1, min_step=0.001):
        self.model = model
        self.data = data
        self.data_leng = len(self.data.Dbirth)
        self.C_init = self.model.gen_init(self.data.df.Dbirth)
        self.pindex = np.arange(len(self.data.Dbirth)*self.model.nvars)
        self.tindex = np.tile(self.data.tindex, self.model.nvars)
        self.step_size = step_size
        self.rtol = rtol
        self.min_step = min_step



        lower_phy_limit = {k:model.limit[k][0] for k in model.limit.keys()}
        lower_fit_limit = self.model.transform_physical_to_fit(lower_phy_limit,mode='bayes')
        upper_phy_limit = {k:model.limit[k][1] for k in model.limit.keys()}
        upper_fit_limit = self.model.transform_physical_to_fit(upper_phy_limit,mode='bayes')

        dpa = [model.default_parameters[name] for name in model.parameter_names]
        x = [[lower_fit_limit[name] for name in model.parameter_names]] +\
                    [dpa]+ \
            [[upper_fit_limit[name] for name in model.parameter_names]]
        xT = list(map(list, zip(*x)) )

        y = [-1e11,-1e10,-1e11]
        self.interpolate_ll = [interp1d(i, y, fill_value="extrapolate") for i in xT]


    def calc_aic(self, p):
        return 2*(self.model.nparas) - 2*self.loglike_dict(p)

    def calc_bic(self, p):
        return np.log(self.data_leng)*(self.model.nparas) - 2*self.loglike_dict(p)

    def calc_aicc(self, p):
        return self.calc_aic(p) + (2*self.model.nparas**2 + 2*self.model.nparas)/(len(self.data.d14C)-self.model.nparas-1)


    def loglike_dict(self, parameters, model=None):
        if model is None:
            model = self.model
        return self.loglike([parameters[i] for i in model.parameter_names],model)

    def loglike(self, parameters, model):
        m_results = self.calc_sim_data(parameters, model)
        residual = m_results - self.data.df.d14C
        if residual.isnull().values.any():
            raise ValueError('Data and residual have different structures')
        sse = np.average( residual**2 )
        n = len(residual)
        loglike = -n*0.5*(2.837877066409345 + np.log(sse))
        return loglike

    def calc_sigma(self,parameters,model=None):
        if model is None:
            model = self.model
        m_results = self.calc_sim_data_dict(parameters, model)
        residual = m_results - self.data.df.d14C
        if residual.isnull().values.any():
            raise ValueError('Data and residual have different structures')
        sigma =np.sqrt( np.average( residual**2 ) )
        return sigma

    def residual_normaltest(self, parameters):
        resi = (self.calc_sim_data_dict(parameters) - self.data.d14C)
        return normaltest(resi.values)


    def calc_sim_data_dict(self, parameters_fit, model=None):
        if model is None:
            model = self.model
        return self.calc_sim_data([parameters_fit[i] for i in model.parameter_names],model)

    def calc_sim_data(self, parameters_fit, model):
        model.set_Dbirth(self.data.Dbirth)
        if not model.set_parameters_fit_array(parameters_fit):
            return -np.inf
        result_sim = self.odeint(model)
        if result_sim is None:
            return -np.inf
        return model.measurement_model(result_sim, self.data)

    def odeint(self, model):
        model.set_Dbirth(self.data.Dbirth)
        try:
            if self.step_size is None:
                sol = sp.integrate.solve_ivp(fun=model.rhs,
                            y0=self.C_init.unstack().values.copy(),
                            rtol=self.rtol,
                            min_step=self.min_step,
                            t_span=self.data.span,
                            t_eval=self.data.t_eval,
                            method=RK45)
            else:
                sol = sp.integrate.solve_ivp(fun=model.rhs,
                            y0=self.C_init.unstack().values.copy(),
                            max_step=self.step_size, atol=np.inf, rtol=np.inf,
                            t_span=self.data.span,
                            t_eval=self.data.t_eval,
                            method='RK45')
        except ValueError as e:
            logger.info("Value Error in ode int: %s", e)
            return None
        except ImplicitParametersOutOfRange as e:
            logger.info("ImplicitParametersOutOfRange Error in ode int: %s ;  return none", e)
            return None

        if sol['status'] < 0:
            logger.info("ode int failed with:  %s", sol['message'])
            return None
        if np.isnan(sol['y']).any():
            logger.info("NaN in ode int solution")
            return None
        sol_res =  model.bins_to_value(sol['y'],self.data_leng,len(self.data.t_eval),self.data.t_eval+self.data.Dbirth[:,np.newaxis])
        return pd.DataFrame( np.reshape(sol_res[(self.pindex, self.tindex)],(model.nvars,-1)).T,\
            columns=model.var_names, index=self.data.df.Dbirth.index)




    def Nloglike(self, parameters, model):
        model.set_Dbirth(self.data.Dbirth)
        #ingnore sigma
        parameter_names = self.model.parameter_names.copy()
        parameter_names.remove('sigma')
        parameters_dict = {name:val for name,val in zip(parameter_names,parameters)}
        parameters_dict['sigma'] = model.default_parameters['sigma']
        if not model.set_parameters_fit(parameters_dict):
            return np.inf
        result_sim = self.odeint(model)
        m_results = model.measurement_model(result_sim, self.data)
        residual = m_results - self.data.df.d14C
        if residual.isnull().values.any():
            raise ValueError('Data and residual have different structures')
        #sse = np.average( residual**2 ,weights=1/self.data.df.e14C )
        sse = np.average( residual**2  )
        n = len(residual)
        loglike = -n*0.5*(2.837877066409345 + np.log(sse))
        return -loglike





    def optimize_minuit_multistart(self, processes=4, return_with_error=True, mode='uniform',perror=None, init_limit=None, n=100, seed=42,\
        nwalkers=100, steps=100):
        #ingnore sigma
        parameter_names = self.model.parameter_names.copy()
        parameter_names.remove('sigma')
        if perror is None:
            error =  [self.model.error[pname] for pname in parameter_names]
        else:
            error = [perror[pname] for pname in parameter_names]

        if mode == 'uniform':
            np.random.seed(seed)
            if init_limit is None:
                phy_limit = self.model.limit
                lower_phy_limit = {k:phy_limit[k][0] for k in phy_limit.keys()}
                upper_phy_limit = {k:phy_limit[k][1] for k in phy_limit.keys()}

                lower_fit_limit = self.model.transform_physical_to_fit(lower_phy_limit,mode='freq')
                upper_fit_limit = self.model.transform_physical_to_fit(upper_phy_limit,mode='freq')

                fit_limit_list = [(lower_fit_limit[pname],upper_fit_limit[pname]) for pname in parameter_names]
            else:
                fit_limit_list = [init_limit[pname] for pname in parameter_names]

            print(fit_limit_list)
            p0s_untested = np.random.uniform([l[0] for l in fit_limit_list], [l[1] for l in fit_limit_list], (n, self.model.nparas-1)  )
            p0s = []
            for p0 in p0s_untested:
                p0_dict ={name:val for name,val in zip(parameter_names,p0)}
                p0_dict['sigma']=self.model.default_parameters['sigma']
                if self.model.set_parameters_fit(p0_dict, mode='freq'):
                    p0s.append(p0)
        elif mode == 'mcmc':
            if nwalkers is None:
                nwalkers=self.model.nparas*100
            _p, chain, le,_lep = self.optimize_emcee(mode='uniform',steps=steps, nwalkers=nwalkers, processes=processes, seed=seed)
            paras = []
            for i in range(nwalkers):
                p = chain[i,-1,:]
                l = le[i,-1]
                i = i+1
                if np.isfinite(l):
                    paras.append( np.delete(p, self.model.parameter_names.index('sigma'))  )
            print(len(paras),"parameters sets left starting minuit")
            p0s = paras
        else:
            raise NotImplementedError
        with Pool(processes=processes) as pool:
            res_pool = [pool.apply_async( self.optimize_minuit,(p0, error,)  ) for p0 in p0s]
            res = [res_p.get() for res_p in res_pool]
        res = [[{n:p for n, p in zip(parameter_names, p0)}] + list(r) for p0, r in zip(p0s, res)]
        res = pd.DataFrame(res, columns=['p0', 'values', 'errors', 'fval', 'valid', 'corr', 'cov'])
        res['n'] = len(self.data.df.d14C)
        res['fval'] = res['fval'].astype('float')
        res['valid'] = res['valid'].astype('bool')
        res.columns.name = self.model.__class__.__name__
        res = res[['p0', 'fval', 'values', 'errors', 'corr', 'cov', 'valid', 'n']].sort_values('fval')
        if (return_with_error):
            res_wE = res[res.errors.apply(lambda x: np.sum([np.isnan( x[i] ) for i in x.keys()])==0 )]
            return res_wE
        return res


    def optimize_minuit(self, pin=None, perror=None, plimit=None):
        #ignore sigma
        parameter_names = self.model.parameter_names.copy()
        parameter_names.remove('sigma')

        if pin is None:
            p0 =  [self.model.default_parameters[pname] for pname in parameter_names]
        elif isinstance(pin, dict):
            p0 = [pin[pname] for pname in parameter_names]
        else:
            p0 = pin
        if perror is None:
            error =  [self.model.error[pname] for pname in parameter_names]
        elif isinstance(pin, dict):
            error = [perror[pname] for pname in parameter_names]
        else:
            error = perror

        if plimit is None:
            limit = [None for pname in parameter_names]
        else:
            limit = [plimit[pname] for pname in parameter_names]

        try:
            res = sp.optimize.minimize(lambda x: self.Nloglike(x, copy.deepcopy(self.model)), p0)

            values = {name: value for name, value in zip(parameter_names, res['x'])}
            errors = {name: error for name, error in zip(parameter_names, np.sqrt(res['hess_inv'].diagonal()))}
            fval = res['fun']
            cov = res['hess_inv']
            corr = np.empty_like(cov)

            def cor(i, j):
                return cov[i, j] / (np.sqrt(cov[i, i] * cov[j, j]) + 1e-100)

            for i in range(len(parameter_names)):
                for j in range(len(parameter_names)):
                    corr[i, j] = cor(i, j)

            cov = pd.DataFrame(cov, index=parameter_names,
                               columns=parameter_names)
            corr = pd.DataFrame(corr, index=parameter_names,
                               columns=parameter_names) # not yet implemented

            return values, errors, fval, res['success'], corr, cov

        except Exception as e:
            logger.info("Optimization failed: %s", e)
            fill  = np.nan*np.ones((self.model.nparas-1, self.model.nparas-1))
            errors = {name: np.nan for name in parameter_names}
            return {n:np.nan for n in parameter_names}, errors, np.inf, False, \
                    pd.DataFrame(np.array(fill),columns=parameter_names, index=parameter_names), \
                    pd.DataFrame(np.array(fill),columns=parameter_names, index=parameter_names)



    def lnprob(self, parameters, model):
        data_elements = len(self.data.df.d14C)

        model.set_Dbirth(self.data.Dbirth)
        if not model.set_parameters_fit_array(parameters,mode='bayes'):
            if not self.gradient_to_default:
                return -np.inf,np.ones(data_elements)*-np.inf
            else:
                interpolate_to_default_para = np.array([i(f) for i, f in zip(self.interpolate_ll, parameters)]).sum()
                return interpolate_to_default_para,np.ones(data_elements)*interpolate_to_default_para

        try:
            if self.step_size is None:
                sol = sp.integrate.solve_ivp(fun=model.rhs,
                            y0=self.C_init.unstack().values.copy(),
                            rtol=self.rtol,
                            min_step=self.min_step,
                            t_span=self.data.span,
                            t_eval=self.data.t_eval,
                            method=RK45)
            else:
                sol = sp.integrate.solve_ivp(fun=model.rhs,
                            y0=self.C_init.unstack().values.copy(),
                            max_step=self.step_size, atol=np.inf, rtol=np.inf,
                            t_span=self.data.span,
                            t_eval=self.data.t_eval,
                            method='RK45')
        except ValueError as e:
            logger.info("Value Error in ode int: %s ; return -inf", e)
            if not self.gradient_to_default:
                return -np.inf,np.ones(data_elements)*-np.inf
            else:
                interpolate_to_default_para = np.array([i(f) for i, f in zip(self.interpolate_ll, parameters)]).sum()
                return interpolate_to_default_para,np.ones(data_elements)*interpolate_to_default_para
        except ImplicitParametersOutOfRange as e:
            logger.info("ImplicitParametersOutOfRange Error in ode int: %s ;  return inf", e)
            if not self.gradient_to_default:
                return -np.inf,np.ones(data_elements)*-np.inf
            else:
                interpolate_to_default_para = np.array([i(f) for i, f in zip(self.interpolate_ll, parameters)]).sum()
                return interpolate_to_default_para,np.ones(data_elements)*interpolate_to_default_para
        if sol['status'] < 0:
            logger.info("ode int failed with:  %s ;  return -inf",
                        sol['message'])
            if not self.gradient_to_default:
                return -np.inf,np.ones(data_elements)*-np.inf
            else:
                interpolate_to_default_para = np.array([i(f) for i, f in zip(self.interpolate_ll, parameters)]).sum()
                return interpolate_to_default_para,np.ones(data_elements)*interpolate_to_default_para

        if np.isnan(sol['y']).any():
            logger.info("NaN in ode int solution; return -inf")
            if not self.gradient_to_default:
                return -np.inf,np.ones(data_elements)*-np.inf
            else:
                interpolate_to_default_para = np.array([i(f) for i, f in zip(self.interpolate_ll, parameters)]).sum()
                return interpolate_to_default_para,np.ones(data_elements)*interpolate_to_default_para

        sol_res =  model.bins_to_value(sol['y'],self.data_leng,len(self.data.t_eval),self.data.t_eval+self.data.Dbirth[:,np.newaxis])
        result_sim = pd.DataFrame( np.reshape(sol_res[(self.pindex, self.tindex)],(model.nvars,-1)).T,\
            columns=model.var_names, index=self.data.df.Dbirth.index)

        m_results = model.measurement_model(result_sim, self.data)
        residual = m_results - self.data.df.d14C
        if residual.isnull().values.any():
            raise ValueError('Data and residual have different structures')
        #sse = np.average( residual**2 ,weights=1/self.data.df.e14C )
        ressqu = residual.values**2 
        #sse = np.average( ressqu )
        n = len(residual)
        #lnsse = np.log(sse)
        #loglike = -n*0.5*(2.837877066409345 + lnsse)
        #loglike_i = -0.9189385332046727 - 0.5*lnsse - ressqu/(2*sse)
        sigma = parameters[model.parameter_names.index('sigma')]
        if model.limit['sigma'][0] > sigma or sigma > model.limit['sigma'][1]:
            logger.info("sigma is outside its limits; return -inf")
            if not self.gradient_to_default:
                return -np.inf,np.ones(data_elements)*-np.inf
            else:
                interpolate_to_default_para = np.array([i(f) for i, f in zip(self.interpolate_ll, parameters)]).sum()
                return interpolate_to_default_para,np.ones(data_elements)*interpolate_to_default_para

        l_sigma = 0.5*np.log( sigma )
        s_sigma = 1/(2*sigma*sigma)
        loglike = -0.9189385332046727*n - n*l_sigma - s_sigma*np.sum(ressqu)
        loglike_i = -0.9189385332046727 - l_sigma - s_sigma*ressqu
        #print(sigma,loglike,loglike_i)
        return loglike,loglike_i


    def optimize_emcee(self, pin=None, mode="uniform",seed=42, error=None, steps=500, nwalkers=None, processes=1,init_fit_limit=None,fake_gradient=False,**opt_args):
        self.gradient_to_default = fake_gradient
        if pin is None:
            pin = self.model.default_parameters
        if nwalkers is None:
            nwalkers = self.model.nparas*2
        p0 = [pin[pname] for pname in self.model.parameter_names]
        if error is None:
            error = [self.model.error[pname] for pname in self.model.parameter_names]
        else:
            error = [error[pname] for pname in self.model.parameter_names]

        np.random.seed(seed)
        if mode == "uniform":
            if init_fit_limit is None:
                phy_limit = self.model.limit
            
                lower_phy_limit = {k:phy_limit[k][0] for k in phy_limit.keys()}
                upper_phy_limit = {k:phy_limit[k][1] for k in phy_limit.keys()}

                lower_fit_limit = self.model.transform_physical_to_fit(lower_phy_limit,mode='bayes')
                upper_fit_limit = self.model.transform_physical_to_fit(upper_phy_limit,mode='bayes')

                fit_limit_list = [(lower_fit_limit[pname],upper_fit_limit[pname]) for pname in self.model.parameter_names]
            else:
                fit_limit_list = [init_fit_limit[pname] for pname in self.model.parameter_names]
            p0s = np.random.uniform([l[0] for l in fit_limit_list], [l[1] for l in fit_limit_list], (nwalkers, self.model.nparas)  )

        elif mode=="gaus":
            p0s=[]
            for _w in range(nwalkers):
                tmp = p0 + error*np.random.randn(self.model.nparas)
                p0s.append(tmp)

        elif mode=="raw":
            p0s = opt_args['p0s']
            lnprob0 = opt_args['lnprob0']
            rstate0 = opt_args['rstate0']
            raise NotImplementedError
        else:
            raise NotImplementedError

        with Pool(processes=processes) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, self.model.nparas, self.lnprob, pool=pool, args=[copy.deepcopy(self.model)],blobs_dtype=float)
            sampler.run_mcmc(p0s, steps,skip_initial_state_check=True)
        
        
        chain = sampler.chain
        lnprobability = sampler.lnprobability
        lnprobability_i = np.array(sampler.blobs)
        return sampler, chain, lnprobability , lnprobability_i



    def calc_kde(self, chain, burnin):
        data = np.ravel(chain[:,burnin:,:]).reshape(-1, self.model.nparas).T.copy()
        return gaussian_kde(data)

    def maxkde_auto(self, kde, pin, perror):
        para = []
        logl = []
        for _ in range(10):
            newp=dict()
            for key in pin.keys():
                newp[key] = pin[key] + perror[key]*np.random.randn()
            pmaxkde = self.maxkde(kde, pin=newp, perror=perror)
            para.append(pmaxkde)
            logl.append(self.loglike_dict(pmaxkde, self.model))
        return para[np.argmax(logl)]
