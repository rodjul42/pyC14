import numpy as np
import pandas as pd
import collections
import logging
logger =  logging.getLogger(__name__)



class ImplicitParametersOutOfRange(Exception):
    def __init__(self, message, iparas):
        self.message = message
        self.iparas = iparas
    def __str__(self):
        return self.message

class FailedToCalulatePopulation(Exception):
    def __init__(self, message):
        self.message = message
    def __str__(self):
        return self.message

class NonFiniteValuesinIntegrate(Exception):
    def __init__(self):
        pass
    def __str__(self):
        return "Non Finite Values in Integrate of c14"

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

def listofdict_to_dictofarray(listofdict):
    dictoflist = {n:[] for n in listofdict[0].keys()}
    for dic in listofdict:
        for n in dic.keys():
            dictoflist[n].append(dic[n])
    for n in dictoflist.keys():
        dictoflist[n] = np.array(dictoflist[n])
    return dictoflist


def listofdict_to_dictofarray_f(listofdict,i):
    dictoflist = {n:[] for n in listofdict[0].keys()}
    for dic in listofdict:
        for n in dic.keys():
            if isinstance( dic[n], (collections.Sequence, np.ndarray)):
                dictoflist[n].append(dic[n][i])
            else:
                dictoflist[n].append(dic[n])
    for n in dictoflist.keys():
        dictoflist[n] = np.array(dictoflist[n])
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
    error_estimator_order = NotImplemented
    n_stages = NotImplemented

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, vectorized=False,
                 first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super(RungeKutta, self).__init__(fun, t0, y0, t_bound, vectorized,
                                         support_complex=True)
        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol, self.n)
        self.f = self.fun(self.t, self.y)
        if first_step is None:
            self.h_abs = select_initial_step(
                self.fun, self.t, self.y, self.f, self.direction,
                self.error_estimator_order, self.rtol, self.atol)
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_previous = None

    def _estimate_error(self, K, h):
        return np.dot(K.T, self.E) * h

    def _estimate_error_norm(self, K, h, scale):
        return norm(self._estimate_error(K, h) / scale)

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

        step_accepted = False
        step_rejected = False

        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            y_new, f_new = rk_step(self.fun, t, y, self.f, h, self.A,
                                   self.B, self.C, self.K)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = self._estimate_error_norm(self.K, h, scale)

            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR,
                                 SAFETY * error_norm ** self.error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                step_accepted = True
            else:
                h_abs *= max(MIN_FACTOR,
                             SAFETY * error_norm ** self.error_exponent)
                step_rejected = True

        self.h_previous = h
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
