import numpy as np
import pandas as pd
import collections
import arviz as az
import logging
logger =  logging.getLogger(__name__)

def find_point_estimate(chains,model):
    rands,chain,lnprob,lnprob_i = chains
    max_sample = np.unravel_index(np.argmax(lnprob, axis=None), lnprob.shape)
    return {n:v for v,n in zip(chain[max_sample],model.parameter_names)}

def convert_to_arviz(chains,model,burnin,remove_stuck=False,iparas_time=None,phy_space=False):
    rands,chain,lnprob,lnprob_i = chains
    para_names = model.parameter_names
    if remove_stuck:
        old_number_chains = lnprob.shape[0] 
        chain_mask = np.where(lnprob[:,burnin:].min(axis=1)>-1e10)[0]
        chain = chain[chain_mask,burnin:]
        lnprob_i = lnprob_i[burnin:,chain_mask]
        lnprob = lnprob[chain_mask,burnin:]
        print(old_number_chains - len(chain_mask),' chains are stuck')
    else:
        chain = chain[:,burnin:]
        lnprob_i = lnprob_i[burnin:,:]
        lnprob = lnprob[:,burnin:]

    if iparas_time is not None:
        chain_s = chain.shape
        iparas_name = list(model.calc_implicit_parameters(iparas_time).keys())
        chain_woth_ip = np.zeros((chain_s[0],chain_s[1],chain_s[2]+len(iparas_name)))
        for n_chain in range(chain_s[0]):
            for n_sample in range(chain_s[1]):
                model.set_parameters_fit_array(chain[n_chain,n_sample],mode='bayes')
                chain_woth_ip[n_chain,n_sample,chain_s[2]:] = [model.calc_implicit_parameters(iparas_time)[name] for name in iparas_name]
                sample_stats = {'log_likelihood':np.transpose(lnprob_i,axes=(1,0,2)),'loglike_values':lnprob}

        chain_dict = {name: chain[:,:,i] for i,name in enumerate(model.parameter_names) }
        chain_dict_phy = model.transform_fit_to_physical(chain_dict,mode='bayes')
        chain = np.transpose(np.array([chain_dict_phy[name] for name in model.parameter_names]),axes=(1,2,0))
        chain_woth_ip[:,:,:chain_s[2]] = chain
        
        return az.from_dict(posterior={'a':chain_woth_ip},sample_stats=sample_stats,dims={'a':['ac']},coords ={'ac':list(para_names)+iparas_name}),list(para_names)+iparas_name
    else:
        if phy_space:
            chain_dict = {name: chain[:,:,i] for i,name in enumerate(model.parameter_names) }
            chain_dict_phy = model.transform_fit_to_physical(chain_dict,mode='bayes')
            chain = np.transpose(np.array([chain_dict_phy[name] for name in model.parameter_names]),axes=(1,2,0))
        sample_stats = {'log_likelihood':np.transpose(lnprob_i,axes=(1,0,2)),'loglike_values':lnprob}
        return az.from_dict(posterior={'a':chain},sample_stats=sample_stats,dims={'a':['ac']},coords ={'ac':para_names}),list(para_names)


def run_convergence_checks(az_data):
    if az_data.posterior.dims['chain'] == 1:
        msg = ("Only one chain was sampled, this makes it impossible to "
               "run some convergence checks")
        return [msg]

    from arviz import rhat, ess

    ess = ess(az_data)
    rhat = rhat(az_data)
       
    warnings = []
    rhat_max = float(rhat.max()['a'].values)
    
    if rhat_max > 1.4:
        msg = ("ERROR: The rhat statistic is larger than 1.4 for some "
               "parameters. The sampler did not converge.")
        warnings.append(msg)
    elif rhat_max > 1.2:
        msg = ("WARN: The rhat statistic is larger than 1.2 for some "
               "parameters.")
        warnings.append(msg)
    elif rhat_max > 1.05:
        msg = ("INFO: The rhat statistic is larger than 1.05 for some "
               "parameters. This indicates slight problems during "
               "sampling.")
        warnings.append(msg)

    eff_min = float(ess.min()['a'].values)
    n_samples =  az_data.posterior.dims['draw'] * az_data.posterior.dims['chain']
    if eff_min < 200 and n_samples >= 500:
        msg = ("ERROR: The estimated number of effective samples is smaller than "
               "200 for some parameters.")
        warnings.append(msg)
    elif eff_min / n_samples < 0.1:
        msg = ("WARN: The number of effective samples is smaller than "
               "10% for some parameters.")
        warnings.append(msg)
    elif eff_min / n_samples < 0.25:
        msg = ("INFO: The number of effective samples is smaller than "
               "25% for some parameters.")

        warnings.append(msg)

    return warnings

class ImplicitParametersOutOfRange(Exception):
    def __init__(self, message, iparas):
        self.message = message
        self.iparas = iparas
    def __str__(self):
        return self.message


def col_name_levels(df, name):
    ''' get levels from level 'name' '''
    pos = df.columns.names.index(name)
    return df.columns.levels[pos]

def col_name_labels(df, name):
    ''' get labels from level 'name' '''
    pos = df.columns.names.index(name)
    return df.columns.labels[pos]

def col_name(df, name):
    ''' get name-level values from df '''
    return col_name_levels(df, name)[col_name_labels(df, name)]


def trans_arcsin(value_phy, limit):
    return np.arcsin(   (2*value_phy - limit[1]-limit[0]) / (limit[1]-limit[0])   )

def trans_sin(value_fit, limit):
    trans =  0.5*(  limit[1]+limit[0] + (limit[1]-limit[0])*np.sin(value_fit))
    if isinstance(value_fit, (collections.Sequence, np.ndarray)):
        mask = np.logical_or(value_fit < -np.pi*0.5,value_fit > np.pi*0.5)
        trans[mask] = limit[1]*2
    else:
        if value_fit < -np.pi*0.5 or value_fit > np.pi*0.5:
            trans = limit[1]*2
    return trans


def listofdict_to_dictoflist(listofdict):
    dictoflist = {n:[] for n in listofdict[0].keys()}
    for dic in listofdict:
        for n in dic.keys():
            dictoflist[n].append(dic[n])
    return dictoflist

def read_xls(path, useasindex=None):
    data = pd.read_excel(path)
    data.rename(columns={'Δ 14C':'d14C', 'Error, 2 s':'e14C', \
            'DOB mod':'Dbirth', 'DOB':'Dbirth', \
            'DOD mod':'Dcoll', 'DOD':'Dcoll', 'DOA':'Dcoll', \
            'no sorted':'N_cells', 'Pathology':'pathology', 'sort':'type', 'Code':'code'}, inplace=True)
    data.d14C = data.d14C/1000.0    # convert unit
    data.e14C = data.e14C/1000.0/2  # convert to 1s error
    data['age'] = data.Dcoll - data.Dbirth
    data.dropna(axis=0, how='all', inplace=True)
    if useasindex is not None:
        data.rename(columns={useasindex:'UID'}, inplace=True)
        data.set_index('UID', verify_integrity=True, inplace=True)
    else:
        data.index.name = 'UID'
    return data

from scipy.integrate._ivp.base import OdeSolver, DenseOutput
from scipy.integrate._ivp.common import (validate_max_step, validate_tol, select_initial_step,
                     norm, warn_extraneous, validate_first_step)

from scipy.integrate._ivp.rk import rk_step, RkDenseOutput

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.


class RungeKutta(OdeSolver):
    """Base class for explicit Runge-Kutta methods."""
    C = NotImplemented
    A = NotImplemented
    B = NotImplemented
    E = NotImplemented
    P = NotImplemented
    order = NotImplemented
    n_stages = NotImplemented

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, min_step=0, vectorized=False,
                 first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super(RungeKutta, self).__init__(fun, t0, y0, t_bound, vectorized,
                                         support_complex=True)
        self.min_step_user = min_step
        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        self.f = self.fun(self.t, self.y)
        if first_step is None:
            self.h_abs = select_initial_step(
                self.fun, self.t, self.y, self.f, self.direction,
                self.order, self.rtol, self.atol)
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)

    def _step_impl(self):
        t = self.t
        y = self.y

        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        order = self.order
        step_accepted = False

        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            y_new, f_new, error = rk_step(self.fun, t, y, self.f, h, self.A,
                                          self.B, self.C, self.E, self.K)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = norm(error / scale)
            if error_norm == 0.0:
                h_abs *= MAX_FACTOR
                step_accepted = True
            elif error_norm < 1:
                h_abs *= min(MAX_FACTOR,
                             max(1, SAFETY * error_norm ** (-1 / (order + 1))))
                step_accepted = True
            else:
                if (h_abs<self.min_step_user):
                    step_accepted = True
                    logger.debug("nmin step size reached")
                else:
                    h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm ** (-1 / (order + 1)))

        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        return True, None

    def _dense_output_impl(self):
        Q = self.K.T.dot(self.P)
        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)

class RK45(RungeKutta):
    order = 4
    n_stages = 6
    C = np.array([1/5, 3/10, 4/5, 8/9, 1])
    A = [np.array([1/5]),
         np.array([3/40, 9/40]),
         np.array([44/45, -56/15, 32/9]),
         np.array([19372/6561, -25360/2187, 64448/6561, -212/729]),
         np.array([9017/3168, -355/33, 46732/5247, 49/176, -5103/18656])]
    B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
                  1/40])
    # Corresponds to the optimum value of c_6 from [2]_.
    P = np.array([
        [1, -8048581381/2820520608, 8663915743/2820520608,
         -12715105075/11282082432],
        [0, 0, 0, 0],
        [0, 131558114200/32700410799, -68118460800/10900136933,
         87487479700/32700410799],
        [0, -1754552775/470086768, 14199869525/1410260304,
         -10690763975/1880347072],
        [0, 127303824393/49829197408, -318862633887/49829197408,
         701980252875 / 199316789632],
        [0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
        [0, 40617522/29380423, -110615467/29380423, 69997945/29380423]])
