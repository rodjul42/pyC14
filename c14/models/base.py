from copy import deepcopy
import logging
import numpy as np
import pandas as pd
import pkg_resources
from scipy import interp
from scipy.interpolate import interp1d, UnivariateSpline
from ..tools import  ImplicitParametersOutOfRange,FailedToCalulatePopulation, trans_arcsin, trans_sin

logger = logging.getLogger(__name__)

__all__ = ['Catm', 'exp_data', 'model_base']

c14atm_data = pkg_resources.resource_filename(__name__, "../data/c14atm.dat")


class Catm:
    def __init__(self, delay=1):
        self.delay = delay
        timep, c14p = np.genfromtxt(c14atm_data).T
        self.timep = timep + delay
        self.c14p = c14p
        self.len = len(self.timep)
        self.lin = interp1d(self.timep, self.c14p, bounds_error=False,
                            fill_value=(0, 0))
        self.spline = UnivariateSpline(self.timep, self.c14p, ext=1, k=1, s=0)

    def lin1(self, x):
        return interp(x, self.timep, self.c14p)


class exp_data:
    def __init__(self, df):
        self.df = df
        self.t_eval = np.unique(df.age.sort_values())
        self.tindex = [np.where(self.t_eval == i)[0][0] for i in df.age]
        self.span = (0, df.age.max())
        self.age = (df.Dcoll - df.Dbirth).values
        self.Dbirth = df.Dbirth.values
        self.d14C = df.d14C.values


class model_base:
    def __init__(self, var_names, default_parameters, error, limit):
        self.check_max_age = 80
        self.var_names = var_names
        self.nvars = len(var_names)
        for i_n, i in enumerate(var_names):
            self.__dict__.update({i: i_n})
       
        #if no logparas assume all are log except if there are linparas
        try:
            self.__dict__['logparas']
            self.logparas=list(self.logparas)
        except KeyError:
            logger.warning("Default logparas missing. ")
            try:
                linparas = self.__dict__['linparas']
                self.logparas=[]
                for para in default_parameters:
                    if para not in linparas:
                        self.logparas.append(para)
                logger.warning("linparas found -> assuming all other are logparas. ")
            except KeyError:
                self.logparas=list(default_parameters.keys())
                logger.warning("No linparas -> assuming all are logparas. ")
                

        #if no linparas assume no linparas
        try:
            self.__dict__['linparas']
            self.linparas=list(self.linparas)
        except KeyError:
            logger.error("Default linparas missing. Assuming all are logparas")
            self.linparas=list()

        #check for sigma
        try:
            default_parameters['sigma']
        except KeyError:
            logger.warning("Default parameters missing sigma added automatikcally with limits 0,0.2")
            default_parameters['sigma'] = 0.1
            limit['sigma'] = (0,0.2)
            error['sigma'] = 0.05
            self.linparas.append('sigma')

        self.default_parameters = default_parameters
        self.set_error(error)
        self.set_limit(limit)
        if not self.set_parameters_phy(default_parameters.copy()):
            logger.warning("Default parameters are outside of limits")
            self.set_parameters_phy(default_parameters.copy(),ignore_physics=True) 
        self.parameter_names = list(list(default_parameters.keys()))
        self.nparas = len(self.parameter_names)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        variab = [i + "\n" for i in self.var_names]
        dparas = [i + " = " + str(self.default_parameters[i]) + "\n" for i in self.default_parameters.keys()]
        dlimits = [i + " = " + str(self.limit[i]) + "\n" for i in self.limit.keys()]
        derror = [i + " = " + str(self.error[i]) + "\n" for i in self.error.keys()]

        paras = [i + " = " + str(self.__dict__[i]) + "\n" for i in self.parameter_names]
        try:
            iparas = self.get_implicit_parameters([0, 25, 50, 75, 100])
        except ValueError as e:
            print(e)
            iparas = {'Na': np.nan}
        iparas_str = [i + " = " + str(iparas[i]) + "\n" for i in iparas.keys()]
        return "\nParameters are:\n" + "".join(paras) + \
            "\nImplicit Parameters are:\n" + "".join(iparas_str) +\
            "\nVariables are t=[0,25,50,75,100]:\n" + "".join(variab) +\
            "\nDefault parameters are:\n" + "".join(dparas) +\
            "\nDefault limit are:\n" + "".join(dlimits) +\
            "\nDefault errors are:\n" + "".join(derror)

    def set_Dbirth(self, birth):
        if isinstance(birth, np.ndarray):
            self.Dbirth = birth
        else:
            self.Dbirth = birth.values

    def gen_init(self, Dbirth):
        if isinstance(Dbirth, pd.DataFrame):
            idx = Dbirth.index
        else:
            idx = None
        C_init = pd.DataFrame(self.Catm.lin(Dbirth)[:, np.newaxis]*np.ones(self.nvars), columns=self.var_names, index=idx)
        return C_init

    def calc_prior(self):
        return 1.0
    
    def rhs(self, t, y):
        raise NotImplementedError

    def measurement_model(self, result_sim, data):
        return result_sim[self.var_names[0]]

    def transform_fit_to_physical(self, p_fit,mode='freq'):
        p_phy = p_fit.copy()
        for p in self.logparas:
            p_phy[p] = 10**p_fit[p]
        for p in self.linparas:
            if mode=='freq' and p!='sigma':
                p_phy[p] = trans_sin(p_fit[p], self.limit[p])
            elif mode=='freq' and p=='sigma':
                pass
            elif mode=='bayes':
                pass
            else:
                raise NotImplementedError
        return p_phy

    def transform_fit_to_physical_array(self, p_fit_array,mode='freq'):
        p_fit = {name: value
                 for name, value in zip(self.parameter_names, p_fit_array)}
        return self.transform_fit_to_physical(p_fit,mode=mode)

    def transform_physical_to_fit(self, p_phy,mode='freq'):
        p_fit = p_phy.copy()
        for p in self.logparas:
            p_fit[p] = np.log10(p_phy[p])
        for p in self.linparas:
            if mode=='freq' and p!='sigma':
                p_fit[p] = trans_arcsin(p_phy[p], self.limit[p])
            elif mode=='freq' and p=='sigma':
                pass
            elif mode=='bayes':
                pass
            else:
                raise NotImplementedError
        return p_fit

    def check_limit(self, p_phy):
        for i in p_phy.keys():
            limit = self.limit[i]
            if limit[0] > p_phy[i] or p_phy[i] > limit[1]:
                logger.info("Parameter %s=%s is outside the limits %s",
                            i, p_phy[i], limit)
                return False
        return True

        

    def set_parameters_phy(self, p_phy, ignore_physics=False,mode='freq'):
        self.__dict__.update(p_phy)
        try:
            self.calc_initial_parameters()
        except ImplicitParametersOutOfRange as e:
            logging.debug(e)
            if not ignore_physics:
                return False
        except FailedToCalulatePopulation as e:
            logging.debug(e)
            return False
        if mode=='freq':
            try:
                p_phy['sigma']
            except KeyError:
                p_phy['sigma'] = 0.1
        
        if not ignore_physics:
            if not self.check_limit(p_phy):
                return False
        if not ignore_physics:
            try:
                self.calc_implicit_parameters(0)
                self.calc_implicit_parameters(self.check_max_age)
            except ValueError as e:
                logging.debug(e)
                return False
            except ImplicitParametersOutOfRange as e:
                logging.debug(e)
                return False
        
        return True

    def set_parameters_fit(self, p_fit, ignore_physics=False,mode='freq'):
        p_phy = self.transform_fit_to_physical(p_fit,mode=mode)
        return self.set_parameters_phy(p_phy, ignore_physics=False,mode=mode)

    def set_parameters_fit_array(self, p_fit_array, ignore_physics=False,mode='freq'):
        p_fit = dict(map(list, zip(self.parameter_names, p_fit_array)))
        return self.set_parameters_fit(p_fit, ignore_physics,mode=mode)

    def set_error(self, error):
        self.error = error

    def set_limit(self, limit):
        self.limit = limit

    def get_implicit_parameters(self, t):
        if isinstance(t, (list, np.ndarray, tuple)):
            iparas_name = self.calc_implicit_parameters(t[0]).keys()
            i_paras_t = {k: [] for k in iparas_name}
            for tp in t:
                iparas = self.calc_implicit_parameters(tp)
                for k in iparas_name:
                    i_paras_t[k].append(iparas[k])
            return i_paras_t
        else:
            return self.calc_implicit_parameters(t)

    def calc_implicit_parameters(self, t):
        return dict()

    def calc_initial_parameters(self):
        return
    
    
    def copy(self):
        return deepcopy(self)

    def bins_to_value(self,sol_res,d_len,t_len,Dcoll):
        return sol_res

class TEST(model_base):
    populations_DNA = {'cells':1}
    m_types = ['mean']
    populations = ['cells']
    populations_m = {'mean':populations}
    iparas = []

    def __init__(self):
        default_parameters = {'lambda_': 0.5,'sigma':0.1}
        self.linparas = ['sigma']
        self.logparas = ['lambda_']
        limit = {i: (1e-5,10) for i in default_parameters.keys()}
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['cells'],
                            default_parameters=default_parameters,
                            error={i: 0.5 for i in
                                   default_parameters.keys()},
                            limit=limit)



    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)


        M_new[self.cells] = self.lambda_ - M[self.cells] 

        return M_new.ravel()

    def calc_implicit_parameters(self, t):
        delta = self.lambda_
        return {'delta': delta,'cells':1}

    def measurement_model(self, result_sim, data):
        return result_sim['cells']
