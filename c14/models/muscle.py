import numpy as np
import pandas as pd
from .base import model_base, Catm
from scipy.interpolate import UnivariateSpline
from ..tools import trans_arcsin, trans_sin,ImplicitParametersOutOfRange
import pkg_resources
global_error = 0.5
global_limit = (10**-6, 10**3)

GROW_FACTOR = 0.85
GROW_TIME = 20
KTIME = 20
M_SHRINK = 0.02

models_list = []
models_list_fixed = []

class MB(model_base):
    def __init__(self):
        pass



    def transform_physical_to_fit(self, p_phy):
        p_fit = p_phy.copy()
        for p in self.linparas:
            p_fit[p] = trans_arcsin(p_phy[p],self.limit[p])
        for p in self.logparas:
            p_fit[p] = np.log10(p_phy[p])
        return p_fit

    def transform_fit_to_physical(self, p_fit):
        p_phy = p_fit.copy()
        for p in self.linparas:
            p_phy[p] = trans_sin(p_fit[p],self.limit[p])
        for p in self.logparas:
            p_phy[p] = 10**p_fit[p]
        return p_phy

    def measurement_model(self, result_sim, data):
        return result_sim['muscle']




class simple(MB):
    def __init__(self):
        default_parameters=dict(grow_factor=0, grow_time=0, Nm0 = 0, dm=-3)
        self.logparas = ['grow_factor','dm']
        self.linparas = ['grow_time','Nm0']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        limit['Nm0'] = (0.0000001, 1000)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Nm0 = self.Nm0
        growfactor = self.grow_factor
        growtime = self.grow_time
        dm = self.dm
        if (t<growtime):
            pm = growfactor + dm*Nm0 + dm*growfactor*t
            Nm = growfactor*t+Nm0
        else:
            pm = dm*(growfactor*growtime + Nm0)
            Nm = growfactor*growtime+Nm0
        iparas = {'pm' : pm,'Nm':Nm}
        if pm > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"pm"+"=" + str(pm) + " is outside the limits " + str(global_limit),iparas)

        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        muscle = self.muscle

        iparas = self.calc_implicit_parameters(t)
        pm = iparas['pm']
        Nm = iparas['Nm']


        M_new[muscle] = pm*(self.Catm.lin(t + self.Dbirth) - M[muscle]) / Nm

        return M_new.ravel()
models_list.append(simple)


class simple_fixed(simple):
    def __init__(self):
        default_parameters=dict(Nm0 = 0, dm=-3)
        self.grow_factor = GROW_FACTOR
        self.grow_time = GROW_TIME
        self.logparas = ['dm']
        self.linparas = ['Nm0']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['Nm0'] = (0.0000001, 1000)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(simple_fixed)


class dm0simple(MB):
    def __init__(self):
        default_parameters=dict(grow_factor=0, grow_time=0, Nm0 = 0)
        self.logparas = ['grow_factor']
        self.linparas = ['grow_time','Nm0']

        limit  = {i:global_limit for i in default_parameters.keys()}

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Nm0 = self.Nm0
        growfactor = self.grow_factor
        growtime = self.grow_time
        if (t<growtime):
            pm = growfactor
            Nm = growfactor*t+Nm0
        else:
            pm = 0
            Nm = growfactor*growtime+Nm0
        iparas = {'pm' : pm,'Nm':Nm}
        if pm > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"pm"+"=" + str(pm) + " is outside the limits " + str(global_limit),iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        muscle = self.muscle

        iparas = self.calc_implicit_parameters(t)
        pm = iparas['pm']
        Nm = iparas['Nm']

        M_new[muscle] = pm*(self.Catm.lin(t + self.Dbirth) - M[muscle]) / Nm

        return M_new.ravel()
models_list.append(dm0simple)


class dm0simple_fixed(dm0simple):
    def __init__(self):
        default_parameters=dict(Nm0 = 0)
        self.grow_factor = GROW_FACTOR
        self.grow_time = GROW_TIME
        self.logparas = []
        self.linparas = ['Nm0']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['Nm0'] = (0.0000001, 1000)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(dm0simple_fixed)



class simpleMshrink(MB):
    def __init__(self):
        default_parameters=dict(grow_factor=0, grow_time=0, Nm0 = 0, dm=-3, M_shrink=-2, ktime=0)
        self.logparas = ['grow_factor','dm','M_shrink']
        self.linparas = ['grow_time','Nm0','ktime']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        limit['Nm0'] = (0.0000001, 1000)
        limit['ktime'] = (1, 99)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Nm0 = self.Nm0
        growfactor = self.grow_factor
        growtime = self.grow_time
        ktime = self.ktime
        mshrink = self.M_shrink
        dm = self.dm
        if (t<growtime):
            pm = growfactor + dm*Nm0 + dm*growfactor*t
            Nm = growfactor*t+Nm0
        elif (t<growtime + ktime):
            pm = dm*(growfactor*growtime + Nm0)
            Nm = growfactor*growtime+Nm0
        else:
            pm = -mshrink + dm*(growfactor*growtime + Nm0 + mshrink*(growtime + ktime - t))
            Nm = growfactor*growtime + Nm0 - mshrink*(-growtime - ktime + t)
        iparas = {'pm' : pm,'Nm':Nm}
        if Nm <= 0:
            raise ('Nm is negative',iparas)
        if pm < 0:
            raise ImplicitParametersOutOfRange('pm is negative',iparas)
        if pm > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"pm"+"=" + str(pm) + " is outside the limits " + str(global_limit),iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        muscle = self.muscle

        iparas = self.calc_implicit_parameters(t)
        pm = iparas['pm']
        Nm = iparas['Nm']


        M_new[muscle] = pm*(self.Catm.lin(t + self.Dbirth) - M[muscle]) / Nm

        return M_new.ravel()
models_list.append(simpleMshrink)


class simpleMshrink_fixed(simpleMshrink):
    def __init__(self):
        default_parameters=dict(Nm0 = 0, dm=-3)
        self.logparas = ['dm']
        self.linparas = ['Nm0']
        self.grow_factor = GROW_FACTOR
        self.grow_time = GROW_TIME
        self.ktime = KTIME
        self.M_shrink = M_SHRINK

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['Nm0'] = (0.0000001, 1000)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(simpleMshrink_fixed)


class Skonst(MB):
    def __init__(self):
        default_parameters=dict(k=0, dm=-2, grow_factor=0, grow_time=0, Nm0=0)
        self.Ns0 = 1
        self.logparas = ['dm','grow_factor']
        self.linparas = ['k','grow_time','Nm0']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        limit['Nm0'] = (0.0000001, 1000)
        limit['k'] = (0, 1)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Ns0=self.Ns0
        Nm0 = self.Nm0
        growfactor = self.grow_factor
        growtime = self.grow_time
        k = self.k
        dm = self.dm

        if (t<growtime):
            ps = (growfactor + dm*Nm0 + dm*growfactor*t)/(Ns0 - k*Ns0)
            ds = (k*(growfactor + dm*Nm0 + dm*growfactor*t))/(Ns0 - k*Ns0)
            Nm = growfactor*t+Nm0
        else:
            ps = (dm*(growfactor*growtime + Nm0))/(Ns0 - k*Ns0)
            ds = (dm*k*(growfactor*growtime + Nm0))/(Ns0 - k*Ns0)
            Nm = growfactor*growtime + Nm0
        iparas = {'ps' : ps,'ds':ds,'Ns':Ns0,'Nm':Nm}
        if Nm <= 0:
            raise ImplicitParametersOutOfRange('Nm is negative',iparas)
        if ps < 0:
            raise ImplicitParametersOutOfRange('ps is negative',iparas)
        if ds < 0:
            raise ImplicitParametersOutOfRange('ds is negative',iparas)
        if ps > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        if ds > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ds"+"=" + str(ds) + " is outside the limits " + str(global_limit),iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        stem = self.stem
        muscle = self.muscle

        k = self.k
        iparas = self.calc_implicit_parameters(t)
        ps = iparas['ps']
        ds = iparas['ds']
        Nm = iparas['Nm']
        Ns = iparas['Ns']


        M_new[stem] = (Ns**2*((1 + k)*ps*self.Catm.lin(t + self.Dbirth) - 2*ds*M[stem] + (-1 + k)*ps*M[stem]) + 2*(ds - k*ps)*M[stem]*Ns**2)/(2.*Ns**2)
        M_new[muscle] = -((-1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)

        return M_new.ravel()
models_list.append(Skonst)

class Skonst_fixed(Skonst):
    def __init__(self):
        default_parameters=dict(k=0, dm=-2, Nm0=0)
        self.Ns0 = 1
        self.logparas = ['dm']
        self.linparas = ['k','Nm0']
        self.grow_factor = GROW_FACTOR
        self.grow_time = GROW_TIME

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['Nm0'] = (0.0000001, 1000)
        limit['k'] = (0, 1)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(Skonst_fixed)



class k0Skonst(MB):
    def __init__(self):
        default_parameters=dict(dm=-2, grow_factor=0, grow_time=0, Nm0=0)
        self.Ns0 = 1
        self.logparas = ['dm','grow_factor']
        self.linparas = ['grow_time','Nm0']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        limit['Nm0'] = (0.0000001, 1000)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Ns0=self.Ns0
        Nm0 = self.Nm0
        growfactor = self.grow_factor
        growtime = self.grow_time
        dm = self.dm

        if (t<growtime):
            ps = (growfactor + dm*Nm0 + dm*growfactor*t)/Ns0
            ds = 0
            Nm = growfactor*t+Nm0
        else:
            ps = (dm*(growfactor*growtime + Nm0))/Ns0
            ds = 0
            Nm = growfactor*growtime + Nm0
        iparas = {'ps' : ps,'ds':ds,'Ns':Ns0,'Nm':Nm}
        if Nm <= 0:
            raise ImplicitParametersOutOfRange('Nm is negative',iparas)
        if ps < 0:
            raise ImplicitParametersOutOfRange('ps is negative',iparas)
        if ds < 0:
            raise ImplicitParametersOutOfRange('ds is negative',iparas)
        if ps > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        if ds > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ds"+"=" + str(ds) + " is outside the limits " + str(global_limit),iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        stem = self.stem
        muscle = self.muscle

        iparas = self.calc_implicit_parameters(t)
        ps = iparas['ps']
        Nm = iparas['Nm']
        Ns = iparas['Ns']


        M_new[stem] = (ps*(self.Catm.lin(t + self.Dbirth) - M[stem]))/2.
        M_new[muscle] = (ps*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)

        return M_new.ravel()
models_list.append(k0Skonst)

class k0Skonst_fixed(k0Skonst):
    def __init__(self):
        default_parameters=dict(dm=-2, Nm0=0)
        self.Ns0 = 1
        self.logparas = ['dm']
        self.linparas = ['Nm0']
        self.grow_factor = GROW_FACTOR
        self.grow_time = GROW_TIME

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['Nm0'] = (0.0000001, 1000)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(k0Skonst_fixed)


class dm0Skonst(MB):
    def __init__(self):
        default_parameters=dict(k=0, grow_factor=0, grow_time=0, Nm0=0)
        self.Ns0 = 1
        self.logparas = ['grow_factor']
        self.linparas = ['grow_time','Nm0','k']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        limit['Nm0'] = (0.0000001, 1000)
        limit['k'] = (0, 1)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Ns0=self.Ns0
        Nm0 = self.Nm0
        growfactor = self.grow_factor
        growtime = self.grow_time
        k = self.k

        if (t<growtime):
            ps = growfactor/(Ns0 - k*Ns0)
            ds = (growfactor*k)/(Ns0 - k*Ns0)
            Nm = growfactor*t+Nm0
        else:
            ps = 0
            ds = 0
            Nm = growfactor*growtime + Nm0
        iparas = {'ps' : ps,'ds':ds,'Ns':Ns0,'Nm':Nm}
        if Nm <= 0:
            raise ImplicitParametersOutOfRange('Nm is negative',iparas)
        if ps < 0:
            raise ImplicitParametersOutOfRange('ps is negative',iparas)
        if ds < 0:
            raise ImplicitParametersOutOfRange('ds is negative',iparas)
        if ps > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        if ds > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ds"+"=" + str(ds) + " is outside the limits " + str(global_limit),iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        stem = self.stem
        muscle = self.muscle

        k = self.k
        iparas = self.calc_implicit_parameters(t)
        ps = iparas['ps']
        ds = iparas['ds']
        Nm = iparas['Nm']
        Ns = iparas['Ns']


        M_new[stem] = (Ns**2*((1 + k)*ps*self.Catm.lin(t + self.Dbirth) - 2*ds*M[stem] + (-1 + k)*ps*M[stem]) + 2*(ds - k*ps)*M[stem]*Ns**2)/(2.*Ns**2)
        M_new[muscle] = -((-1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)

        return M_new.ravel()
models_list.append(dm0Skonst)

class dm0Skonst_fixed(dm0Skonst):
    def __init__(self):
        default_parameters=dict(k=0, Nm0=0)
        self.Ns0 = 1
        self.logparas = []
        self.linparas = ['Nm0','k']
        self.grow_factor = GROW_FACTOR
        self.grow_time = GROW_TIME

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['Nm0'] = (0.0000001, 1000)
        limit['k'] = (0, 1)


        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(dm0Skonst_fixed)


class k0dm0Skonst(MB):
    def __init__(self):
        default_parameters=dict(grow_factor=0, grow_time=0, Nm0=0)
        self.Ns0 = 1
        self.logparas = ['grow_factor']
        self.linparas = ['grow_time','Nm0']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        limit['Nm0'] = (0.0000001, 1000)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Ns0=self.Ns0
        Nm0 = self.Nm0
        growfactor = self.grow_factor
        growtime = self.grow_time

        if (t<growtime):
            ps = growfactor/Ns0
            ds = 0
            Nm = growfactor*t+Nm0
        else:
            ps = 0
            ds = 0
            Nm = growfactor*growtime + Nm0
        iparas = {'ps' : ps,'ds':ds,'Ns':Ns0,'Nm':Nm}
        if Nm <= 0:
            raise ImplicitParametersOutOfRange('Nm is negative',iparas)
        if ps < 0:
            raise ImplicitParametersOutOfRange('ps is negative',iparas)
        if ds < 0:
            raise ImplicitParametersOutOfRange('ds is negative',iparas)
        if ps > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        if ds > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ds"+"=" + str(ds) + " is outside the limits " + str(global_limit),iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        stem = self.stem
        muscle = self.muscle

        iparas = self.calc_implicit_parameters(t)
        ps = iparas['ps']
        Nm = iparas['Nm']
        Ns = iparas['Ns']


        M_new[stem] = (ps*(self.Catm.lin(t + self.Dbirth) - M[stem]))/2.
        M_new[muscle] = (ps*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)

        return M_new.ravel()
models_list.append(k0dm0Skonst)

class k0dm0Skonst_fixed(k0dm0Skonst):
    def __init__(self):
        default_parameters=dict(Nm0=0)
        self.Ns0 = 1
        self.logparas = []
        self.linparas = ['Nm0']
        self.grow_factor = GROW_FACTOR
        self.grow_time = GROW_TIME

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['Nm0'] = (0.0000001, 1000)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(k0dm0Skonst_fixed)


class SlinshrinkMlinshrink(MB):
    def __init__(self):
        default_parameters=dict(k=0, dm=-2, grow_factor=0, grow_time=0, ktime=0, Nm0=0, M_shrink=-2, S_shrink=-4)
        self.Ns0 = 1
        self.logparas = ['dm','grow_factor','M_shrink','S_shrink']
        self.linparas = ['k','grow_time','Nm0','ktime']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        limit['Nm0'] = (0.0000001, 1000)
        limit['ktime'] = (1, 99)
        limit['k'] = (0, 1)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Ns0=self.Ns0
        Sshrink = self.S_shrink
        Mshrink = self.M_shrink
        Nm0 = self.Nm0
        growfactor = self.grow_factor
        growtime = self.grow_time
        ktime = self.ktime
        k = self.k
        dm = self.dm
        Ns =  Ns0 - Sshrink* t
        if Ns<=0:
            raise ValueError('Ns is negative or zero')

        if (t<growtime):
            ps = (growfactor + dm*Nm0 + dm*growfactor*t)/(Ns0 - k*Ns0 - Sshrink*t + k*Sshrink*t)
            ds = (Sshrink + k*(growfactor + dm*Nm0 - Sshrink + dm*growfactor*t))/((-1 + k)*(-Ns0 + Sshrink*t))
            Nm = growfactor*t+Nm0
        elif (t<growtime + ktime):
            ps = (dm*(growfactor*growtime + Nm0))/((-1 + k)*(-Ns0 + Sshrink*t))
            ds = (-(dm*k*(growfactor*growtime + Nm0)) + (-1 + k)*Sshrink)/((-1 + k)*(Ns0 - Sshrink*t))
            Nm = growfactor*growtime+Nm0
        else:
            ps = (Mshrink - dm*(growfactor*growtime + Nm0 + Mshrink*(growtime + ktime - t)))/((-1 + k)*(Ns0 - Sshrink*t))
            ds = (-Sshrink + k*(Mshrink + Sshrink - dm*(growfactor*growtime + Nm0 + Mshrink*(growtime + ktime - t))))/((-1 + k)*(Ns0 - Sshrink*t))
            Nm = growfactor*growtime + Nm0 - Mshrink*(-growtime - ktime + t)
        iparas = {'ps' : ps,'ds':ds,'Ns':Ns,'Nm':Nm}
        if Nm <= 0:
            raise ImplicitParametersOutOfRange('Nm is negative',iparas)
        if ps < 0:
            raise ImplicitParametersOutOfRange('ps is negative',iparas)
        if ds < 0:
            raise ImplicitParametersOutOfRange('ds is negative',iparas)
        if ps > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        if ds > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ds"+"=" + str(ds) + " is outside the limits " + str(global_limit),iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        stem = self.stem
        muscle = self.muscle

        k = self.k
        iparas = self.calc_implicit_parameters(t)
        ps = iparas['ps']
        ds = iparas['ds']
        Nm = iparas['Nm']
        Ns = iparas['Ns']


        M_new[stem] = (Ns**2*((1 + k)*ps*self.Catm.lin(t + self.Dbirth) - 2*ds*M[stem] + (-1 + k)*ps*M[stem]) + 2*(ds - k*ps)*M[stem]*Ns**2)/(2.*Ns**2)
        M_new[muscle] = -((-1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)

        return M_new.ravel()
models_list.append(SlinshrinkMlinshrink)

class SlinshrinkMlinshrink_fixed(SlinshrinkMlinshrink):
    def __init__(self):
        default_parameters=dict(k=0, dm=-2, Nm0=0, S_shrink=-4)
        self.Ns0 = 1
        self.logparas = ['dm','S_shrink']
        self.linparas = ['k','Nm0']

        self.grow_factor = GROW_FACTOR
        self.grow_time = GROW_TIME
        self.ktime = KTIME
        self.M_shrink = M_SHRINK

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['Nm0'] = (0.0000001, 1000)
        limit['k'] = (0, 1)


        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(SlinshrinkMlinshrink_fixed)



class k0SlinshrinkMlinshrink(MB):
    def __init__(self):
        default_parameters=dict(dm=-2, grow_factor=0, grow_time=0, ktime=0, Nm0=0, M_shrink=-2, S_shrink=-4)
        self.Ns0 = 1
        self.logparas = ['dm','grow_factor','M_shrink','S_shrink']
        self.linparas = ['grow_time','Nm0','ktime']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        limit['Nm0'] = (0.0000001, 1000)
        limit['ktime'] = (1, 99)


        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Ns0=self.Ns0
        Sshrink = self.S_shrink
        Mshrink = self.M_shrink
        Nm0 = self.Nm0
        growfactor = self.grow_factor
        growtime = self.grow_time
        ktime = self.ktime
        dm = self.dm
        Ns =  Ns0 - Sshrink* t
        ds = Sshrink/(Ns0 - Sshrink*t)
        if (t<growtime):
            ps = (growfactor + dm*Nm0 + dm*growfactor*t)/(Ns0 - Sshrink*t)
            Nm = growfactor*t+Nm0
        elif (t<growtime + ktime):
            ps = (dm*(growfactor*growtime + Nm0))/(Ns0 - Sshrink*t)
            Nm = growfactor*growtime+Nm0
        else:
            ps = (-Mshrink + dm*(growfactor*growtime + Nm0 + Mshrink*(growtime + ktime - t)))/(Ns0 - Sshrink*t)
            Nm = growfactor*growtime + Nm0 - Mshrink*(-growtime - ktime + t)
        iparas = {'ps' : ps,'ds':ds,'Ns':Ns,'Nm':Nm}
        if ds < 0:
            raise ImplicitParametersOutOfRange('ds is negative',iparas)
        if Ns<=0:
            raise ImplicitParametersOutOfRange('Ns is negative or zero',iparas)        
        if Nm <= 0:
            raise ImplicitParametersOutOfRange('Nm is negative',iparas)
        if ps < 0:
            raise ImplicitParametersOutOfRange('ps is negative',iparas)
        if ps > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        if ds > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ds"+"=" + str(ds) + " is outside the limits " + str(global_limit),iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        stem = self.stem
        muscle = self.muscle

        iparas = self.calc_implicit_parameters(t)
        ps = iparas['ps']
        ds = iparas['ds']
        Nm = iparas['Nm']
        Ns = iparas['Ns']


        M_new[stem] = (ps*self.Catm.lin(t + self.Dbirth) + M[stem]*(-2*ds - ps + (2*ds*Ns**2)/Ns**2))/2.
        M_new[muscle] = (ps*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)

        return M_new.ravel()
models_list.append(k0SlinshrinkMlinshrink)

class k0SlinshrinkMlinshrink_fixed(k0SlinshrinkMlinshrink):
    def __init__(self):
        default_parameters=dict(dm=-2, Nm0=0, S_shrink=-4)
        self.Ns0 = 1
        self.logparas = ['dm','S_shrink']
        self.linparas = ['Nm0']

        self.grow_factor = GROW_FACTOR
        self.grow_time = GROW_TIME
        self.ktime = KTIME
        self.M_shrink = M_SHRINK

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['Nm0'] = (0.0000001, 1000)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(k0SlinshrinkMlinshrink_fixed)





'''
class ms_nodeath(MB):
    def __init__(self):
        default_parameters=dict(grow_factor=1, grow_time=0, Nm0 = 0.01)
        self.Ns = 1
        self.logparas = ['grow_factor']
        self.linparas = ['grow_time','Nm0']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 50)
        limit['Nm0'] = (0.00001, 100)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        if (t<self.grow_time):
            ps = self.grow_factor/self.Ns
            Nm = self.grow_factor*t+self.Nm0
        else:
            ps = 0
            Nm = self.grow_factor*self.grow_time+self.Nm0
        return {'ps' : ps,'Nm':Nm}

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        stem = self.stem
        muscle = self.muscle

        Ns = self.Ns
        iparas = self.calc_implicit_parameters(t)
        ps = iparas['ps']
        Nm = iparas['Nm']

        M_new[stem] = ps*(self.Catm.lin(t + self.Dbirth) - M[stem])*0.5
        M_new[muscle] = (ps*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)

        return M_new.ravel()
models_list.append(ms_nodeath)


class ms_nodeath_fixed(ms_nodeath):
    def __init__(self):
        default_parameters=dict(Nm0 = 0.01)
        self.Ns = 1
        self.grow_factor = 0.85
        self.grow_time = 20
        self.logparas = []
        self.linparas = ['Nm0']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['Nm0'] = (0.00001, 100)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(ms_nodeath_fixed)

class MdeathSk(MB):
    def __init__(self):
        default_parameters=dict(dm=-2, grow_factor=0, grow_time=0, Nm0=0.001)
        self.Ns = 1
        self.logparas = ['dm','grow_factor']
        self.linparas = ['grow_time','Nm0']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 50)
        limit['Nm0'] = (0.00001, 100)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        if (t<self.grow_time):
            psm = (self.grow_factor+self.dm*self.grow_factor*t+self.dm*self.Nm0)/self.Ns
            Nm = self.grow_factor*t+self.Nm0
        else:
            psm = (self.dm*self.grow_factor+self.dm*self.Nm0)/self.Ns
            Nm = self.grow_factor*self.grow_time+self.Nm0
        return {'psm' : psm,'Nm':Nm}

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        stem = self.stem
        muscle = self.muscle

        Ns = self.Ns
        iparas = self.calc_implicit_parameters(t)
        psm = iparas['psm']
        Nm = iparas['Nm']

        M_new[stem] = psm*(self.Catm.lin(t + self.Dbirth) - M[stem])*0.5
        M_new[muscle] = (psm*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)

        return M_new.ravel()
models_list.append(MdeathSk)

class MdeathSk_fixed(MdeathSk):
    def __init__(self):
        default_parameters=dict(dm=-2, Nm0=0.001)
        self.Ns = 1
        self.grow_factor = 0.85
        self.grow_time = 20
        self.logparas = ['dm']
        self.linparas = ['Nm0']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['Nm0'] = (0.00001, 100)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(MdeathSk_fixed)

class MdeathSdeath(MB):
    def __init__(self):
        default_parameters=dict(dm=-2, grow_factor=0, grow_time=0, Nm0=0, shrink_factor=-4)
        self.Ns0 = 1
        self.logparas = ['dm','grow_factor','shrink_factor']
        self.linparas = ['grow_time','Nm0']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 50)
        limit['Nm0'] = (0.00001, 100)
        limit['shrink_factor'] = (0.0000001, 0.1)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Ns = self.Ns0 - self.shrink_factor * t
        if Ns<=0:
            raise ValueError('Ns is negative or zero')
        ds = self.shrink_factor/Ns
        if (t<self.grow_time):
            psm = (self.grow_factor+self.dm*self.grow_factor*t+self.dm*self.Nm0)/Ns
            Nm = self.grow_factor*t+self.Nm0
        else:
            psm = (self.dm*self.grow_factor+self.dm*self.Nm0)/Ns
            Nm = self.grow_factor*self.grow_time+self.Nm0


        return {'psm' : psm,'ds':ds,'Ns':Ns,'Nm':Nm}

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        stem = self.stem
        muscle = self.muscle

        iparas = self.calc_implicit_parameters(t)
        psm = iparas['psm']
        ds = iparas['ds']
        Nm = iparas['Nm']
        Ns = iparas['Ns']


        M_new[stem] = (psm*self.Catm.lin(t + self.Dbirth) + M[stem]*(-2*ds - psm + (2*ds*Ns**2)/Ns**2))*0.5
        M_new[muscle] = (psm*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)

        return M_new.ravel()
models_list.append(MdeathSdeath)




class MdeathSdeath_fixed(MdeathSdeath):
    def __init__(self):
        default_parameters=dict(dm=-2, Nm0=0, shrink_factor=-4)
        self.Ns0 = 1
        self.grow_factor = 0.85
        self.grow_time = 20
        self.logparas = ['dm','shrink_factor']
        self.linparas = ['Nm0']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['Nm0'] = (0.00001, 100)
        limit['shrink_factor'] = (0.0000001, 0.1)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)

models_list_fixed.append(MdeathSdeath_fixed)
'''
