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
MGROW = 17
KTIME = 20

NMSRATE = 0.004

NM2SRATE = 0.008
NSSRATE = 0.001
RESTTIME = 20

STEMPP = 2.4/100.0

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

class MB2(MB):
    def calc_initial_parameters(self):
        p0 = STEMPP*GROW_TIME*GROW_FACTOR/(1-STEMPP + GROW_TIME*GROW_FACTOR*STEMPP )
        self.Nm0 = (1-p0) * self.N0
        self.Ns0 = p0 * self.N0

class MBt2(model_base):
    def calc_initial_parameters(self):
        p0 = STEMPP*GROW_TIME*GROW_FACTOR/(1-STEMPP + GROW_TIME*GROW_FACTOR*STEMPP )
        self.Nm01 = (1-p0) * self.N0*0.5
        self.Nm02 = (1-p0) * self.N0*0.5
        self.Ns0 = p0 * self.N0


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
        return (result_sim['muscle1'] + result_sim['muscle2'])*0.5



class simple(MB):
    def __init__(self):
        default_parameters=dict(mgrow=0, grow_time=0, dm=-3)
        self.logparas = ['mgrow','dm']
        self.linparas = ['grow_time']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        mgrow = self.mgrow
        growtime = self.grow_time
        dm = self.dm
        if (t<growtime):
            pm = ((-1 + mgrow + dm*(growtime + (-1 + mgrow)*t)))/growtime
            Nm = (mgrow - 1)/growtime*t + 1
        else:
            pm = dm*mgrow
            Nm = mgrow
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
        default_parameters=dict(dm=-3)
        self.mgrow = MGROW
        self.grow_time = GROW_TIME
        self.logparas = ['dm']
        self.linparas = []

        limit  = {i:global_limit for i in default_parameters.keys()}

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(simple_fixed)


'''

class dm0simple(MB2):
    def __init__(self):
        default_parameters=dict(mgrow=0, grow_time=0)
        self.logparas = ['mgrow','N0']
        self.linparas = ['grow_time']

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        limit['N0'] = (0.001, 1000)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Nm0 = self.Nm0
        mgrow = self.mgrow
        growtime = self.grow_time
        if (t<growtime):
            pm = ((-1 + mgrow)*Nm0)/growtime
            Nm = Nm0*(mgrow - 1)/growtime*t+Nm0
        else:
            pm = 0
            Nm = mgrow*Nm0
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
        default_parameters=dict(N0 = 0)
        self.mgrow = MGROW
        self.grow_time = GROW_TIME
        self.logparas = ['N0']
        self.linparas = []

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['N0'] = (0.0000001, 1000)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(dm0simple_fixed)
'''

class Skonst(MB2):
    def __init__(self):
        default_parameters=dict(k=0, dm=-2, mgrow=0, grow_time=0)
        self.logparas = ['dm','mgrow']
        self.linparas = ['k','grow_time']
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
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
        mgrow = self.mgrow
        growtime = self.grow_time
        k = self.k
        dm = self.dm

        if (t<growtime):
            ps = -((Nm0*(-1 + mgrow + dm*(growtime + (-1 + mgrow)*t)))/(growtime*(-1 + k)*Ns0))
            ds = -((k*Nm0*(-1 + mgrow + dm*(growtime + (-1 + mgrow)*t)))/(growtime*(-1 + k)*Ns0))
            Nm = Nm0*(mgrow - 1)/growtime*t+Nm0
        else:
            ps = (dm*mgrow*Nm0)/(Ns0 - k*Ns0)
            ds = (dm*k*mgrow*Nm0)/(Ns0 - k*Ns0)
            Nm = Nm0*mgrow
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


        M_new[stem] = ((1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - M[stem]))/2
        M_new[muscle] = -((-1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)

        return M_new.ravel()
models_list.append(Skonst)

class Skonst_fixed(Skonst):
    def __init__(self):
        default_parameters=dict(k=0, dm=-2)
        self.logparas = ['dm']
        self.linparas = ['k']
        self.mgrow = MGROW
        self.grow_time = GROW_TIME
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k'] = (0, 1)
        
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(Skonst_fixed)


class dm0Skonst(MB2):
    def __init__(self):
        default_parameters=dict(k=0, mgrow=0, grow_time=0)
        self.logparas = ['mgrow']
        self.linparas = ['k','grow_time']
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
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
        mgrow = self.mgrow
        growtime = self.grow_time
        k = self.k

        if (t<growtime):
            ps = -((Nm0 - mgrow*Nm0) /(growtime*Ns0 - growtime*k*Ns0))
            ds = (k*Nm0 - k*mgrow*Nm0)/(-(growtime*Ns0) + growtime*k*Ns0)
            Nm = Nm0*(mgrow - 1)/growtime*t+Nm0
        else:
            ps = 0
            ds = 0
            Nm = Nm0*mgrow
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


        M_new[stem] = ((1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - M[stem]))/2
        M_new[muscle] = -((-1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)

        return M_new.ravel()
models_list.append(dm0Skonst)

class dm0Skonst_fixed(dm0Skonst):
    def __init__(self):
        default_parameters=dict(k=0)
        self.logparas = []
        self.linparas = ['k']
        self.mgrow = MGROW
        self.grow_time = GROW_TIME
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k'] = (0, 1)
        
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(dm0Skonst_fixed)



class Skonst2(MB2):
    def __init__(self):
        default_parameters=dict(ds=-2, dm=-2, mgrow=0, grow_time=0)
        self.logparas = ['dm','mgrow','ds']
        self.linparas = ['grow_time']
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Ns0=self.Ns0
        Nm0 = self.Nm0
        mgrow = self.mgrow
        growtime = self.grow_time
        ds = self.ds
        dm = self.dm

        if (t<growtime):
            ps = (ds*growtime*Ns0 + Nm0*(-1 + mgrow + dm*(growtime + (-1 + mgrow)*t)))/(growtime*Ns0)
            k = (ds*growtime*Ns0)/(ds*growtime*Ns0 + Nm0*(-1 + mgrow + dm*(growtime + (-1 + mgrow)*t)))
            Nm = Nm0*(mgrow - 1)/growtime*t+Nm0
        else:
            ps = ds + (dm*mgrow*Nm0)/Ns0
            k =  (ds*Ns0)/(dm*mgrow*Nm0 + ds*Ns0)
            Nm = Nm0*mgrow
        iparas = {'ps' : ps,'k':k,'Ns':Ns0,'Nm':Nm}
        if Nm <= 0:
            raise ImplicitParametersOutOfRange('Nm is negative',iparas)
        if ps < 0:
            raise ImplicitParametersOutOfRange('ps is negative',iparas)
        if k < 0 or k > 1:
            raise ImplicitParametersOutOfRange('k is not in 0-1',iparas)
        if ps > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        stem = self.stem
        muscle = self.muscle

        ds = self.ds
        dm = self.dm

        iparas = self.calc_implicit_parameters(t)
        ps = iparas['ps']
        k = iparas['k']
        Nm = iparas['Nm']
        Ns = iparas['Ns']


        M_new[stem] =  ((1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - M[stem]))/2
        M_new[muscle] = -((-1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)
                        
        return M_new.ravel()
models_list.append(Skonst2)

class Skonst2_fixed(Skonst2):
    def __init__(self):
        default_parameters=dict(ds=-2, dm=-2)
        self.logparas = ['dm','ds']
        self.linparas = []
        self.mgrow = MGROW
        self.grow_time = GROW_TIME
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(Skonst2_fixed)




class dm0Skonst2(MB2):
    def __init__(self):
        default_parameters=dict(ds=-2, mgrow=0, grow_time=0)
        self.logparas = ['mgrow','ds']
        self.linparas = ['grow_time']
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Ns0=self.Ns0
        Nm0 = self.Nm0
        mgrow = self.mgrow
        growtime = self.grow_time
        ds = self.ds

        if (t<growtime):
            ps = ds + ((-1 + mgrow)*Nm0)/(growtime*Ns0)
            k = (ds*growtime*Ns0)/((-1 + mgrow)*Nm0 + ds*growtime*Ns0)
            Nm = Nm0*(mgrow - 1)/growtime*t+Nm0
        else:
            ps = ds
            k = 1
            Nm = Nm0*mgrow
        iparas = {'ps' : ps,'k':k,'Ns':Ns0,'Nm':Nm}
        if Nm <= 0:
            raise ImplicitParametersOutOfRange('Nm is negative',iparas)
        if ps < 0:
            raise ImplicitParametersOutOfRange('ps is negative',iparas)
        if k < 0 or k > 1:
            raise ImplicitParametersOutOfRange('k is not in 0-1',iparas)
        if ps > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        stem = self.stem
        muscle = self.muscle

        ds = self.ds
        iparas = self.calc_implicit_parameters(t)
        ps = iparas['ps']
        k = iparas['k']
        Nm = iparas['Nm']
        Ns = iparas['Ns']


        M_new[stem] = ((1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - M[stem]))/2.
        M_new[muscle] = -((-1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)

        return M_new.ravel()
models_list.append(dm0Skonst2)

class dm0Skonst2_fixed(dm0Skonst2):
    def __init__(self):
        default_parameters=dict(ds=0)
        self.logparas = ['ds']
        self.linparas = []
        self.mgrow = MGROW
        self.grow_time = GROW_TIME
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}        
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(dm0Skonst2_fixed)



class Type2Skonst(MBt2):
    def __init__(self):
        default_parameters=dict(k=0,dm1=-2,dm2=-2, mgrow=0, grow_time=0)
        self.logparas = ['mgrow','dm1','dm2']
        self.linparas = ['grow_time','k']
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        limit['k'] = (0, 1)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Ns0=self.Ns0
        Nm01 = self.Nm01
        Nm02 = self.Nm02
        mgrow = self.mgrow
        growtime = self.grow_time
        k = self.k
        dm1 = self.dm1
        dm2 = self.dm2

        if (t<growtime):
            ps = -((Nm01*(-1 + mgrow + dm1*(growtime + (-1 + mgrow)*t)) + Nm02*(-1 + mgrow + dm2*(growtime + (-1 + mgrow)*t)))/(growtime*(-1 + k)*Ns0))
            ds = -((k*(Nm01*(-1 + mgrow + dm1*(growtime + (-1 + mgrow)*t)) + Nm02*(-1 + mgrow + dm2*(growtime + (-1 + mgrow)*t))))/(growtime*(-1 + k)*Ns0))
            t1 = (Nm01*(-1 + mgrow + dm1*(growtime + (-1 + mgrow)*t)))/(Nm01*(-1 + mgrow + dm1*(growtime + (-1 + mgrow)*t)) + Nm02*(-1 + mgrow + dm2*(growtime + (-1 + mgrow)*t)))
            Nm1 = Nm01*(mgrow - 1)/growtime*t+Nm01
            Nm2 = Nm02*(mgrow - 1)/growtime*t+Nm02
        else:
            ps = (mgrow*(dm1*Nm01 + dm2*Nm02))/(Ns0 - k*Ns0)
            ds = (k*mgrow*(dm1*Nm01 + dm2*Nm02))/(Ns0 - k*Ns0)
            t1 = (dm1*Nm01)/(dm1*Nm01 + dm2*Nm02)
            Nm1 = Nm01*mgrow
            Nm2 = Nm02*mgrow
        iparas = {'ps' : ps,'ds':ds,'t1':t1,'Ns':Ns0,'Nm1':Nm1,'Nm2':Nm2}
        if Nm1 <= 0:
            raise ImplicitParametersOutOfRange('Nm1 is negative',iparas)
        if Nm2 <= 0:
            raise ImplicitParametersOutOfRange('Nm2 is negative',iparas)
        if ps < 0:
            raise ImplicitParametersOutOfRange('ps is negative',iparas)
        if t1 < 0 or t1 > 1:
            raise ImplicitParametersOutOfRange('t1 is not in 0-1',iparas)
        if ds < 0:
            raise ImplicitParametersOutOfRange('ds is negative',iparas)
        if ps > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        if ds > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ds"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        stem = self.stem
        muscle1 = self.muscle1
        muscle2 = self.muscle2

        k = self.k
        dm1 = self.dm1
        dm2 = self.dm2
        iparas = self.calc_implicit_parameters(t)
        ps = iparas['ps']
        ds = iparas['ds']
        t1 = iparas['t1']
        Nm1 = iparas['Nm1']
        Nm2 = iparas['Nm2']
        Ns = iparas['Ns']


        M_new[stem] = ((1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - M[stem]))/2.
        M_new[muscle1] = -((-1 + k)*ps*t1*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle1] + M[stem])*Ns)/(2.*Nm1)
        M_new[muscle2] =((-1 + k)*ps*(-1 + t1)*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle2] + M[stem])*Ns)/(2.*Nm2)

        return M_new.ravel()
models_list.append(Type2Skonst)

class Type2Skonst_fixed(Type2Skonst):
    def __init__(self):
        default_parameters=dict(k=0,dm1=-2,dm2=-2)
        self.logparas = ['dm1','dm2']
        self.linparas = ['k']
        self.mgrow = MGROW
        self.grow_time = GROW_TIME
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k'] = (0, 1)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(Type2Skonst_fixed)

class dm1_0Type2Skonst_fixed(Type2Skonst):
    def __init__(self):
        default_parameters=dict(k=0,dm2=-2)
        self.logparas = ['dm2']
        self.linparas = ['k']
        self.mgrow = MGROW
        self.grow_time = GROW_TIME
        self.dm1 = 0
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k'] = (0, 1)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(dm1_0Type2Skonst_fixed)

class dm1_0dm2_0Type2Skonst_fixed(Type2Skonst):
    def __init__(self):
        default_parameters=dict(k=0)
        self.logparas = []
        self.linparas = ['k']
        self.mgrow = MGROW
        self.grow_time = GROW_TIME
        self.dm1 = 0
        self.dm2 = 0
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k'] = (0, 1)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)


    def calc_implicit_parameters(self, t):
        Ns0=self.Ns0
        Nm01 = self.Nm01
        Nm02 = self.Nm02
        mgrow = self.mgrow
        growtime = self.grow_time
        k = self.k
        dm1 = self.dm1
        dm2 = self.dm2

        if (t<growtime):
            ps = -((Nm01*(-1 + mgrow + dm1*(growtime + (-1 + mgrow)*t)) + Nm02*(-1 + mgrow + dm2*(growtime + (-1 + mgrow)*t)))/(growtime*(-1 + k)*Ns0))
            ds = -((k*(Nm01*(-1 + mgrow + dm1*(growtime + (-1 + mgrow)*t)) + Nm02*(-1 + mgrow + dm2*(growtime + (-1 + mgrow)*t))))/(growtime*(-1 + k)*Ns0))
            t1 = (Nm01*(-1 + mgrow))/(Nm01*(-1 + mgrow) + Nm02*(-1 + mgrow))
            Nm1 = Nm01*(mgrow - 1)/growtime*t+Nm01
            Nm2 = Nm02*(mgrow - 1)/growtime*t+Nm02
        else:
            ps = (mgrow*(dm1*Nm01 + dm2*Nm02))/(Ns0 - k*Ns0)
            ds = (k*mgrow*(dm1*Nm01 + dm2*Nm02))/(Ns0 - k*Ns0)
            t1 = 0.5
            Nm1 = Nm01*mgrow
            Nm2 = Nm02*mgrow
        iparas = {'ps' : ps,'ds':ds,'t1':t1,'Ns':Ns0,'Nm1':Nm1,'Nm2':Nm2}
        if Nm1 <= 0:
            raise ImplicitParametersOutOfRange('Nm1 is negative',iparas)
        if Nm2 <= 0:
            raise ImplicitParametersOutOfRange('Nm2 is negative',iparas)
        if ps < 0:
            raise ImplicitParametersOutOfRange('ps is negative',iparas)
        if t1 < 0 or t1 > 1:
            raise ImplicitParametersOutOfRange('t1 is not in 0-1',iparas)
        if ds < 0:
            raise ImplicitParametersOutOfRange('ds is negative',iparas)
        if ps > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        if ds > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ds"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        return iparas
models_list_fixed.append(dm1_0dm2_0Type2Skonst_fixed)


class SchrinkType2Skonst(MBt2):
    def __init__(self):
        default_parameters=dict(k=0,dm1=-2,dm2=-2, mgrow=0, grow_time=0, resttime=0,Nm2srate=-3,Nssrate=-3)
        self.logparas = ['mgrow','dm1','dm2','Nm2srate','Nssrate']
        self.linparas = ['grow_time','k','resttime']
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        limit['k'] = (0, 1)
        limit['resttime'] = (0, 30)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        Ns0=self.Ns0
        Nm01 = self.Nm01
        Nm02 = self.Nm02
        mgrow = self.mgrow
        Nm2srate = self.Nm2srate
        Nssrate = self.Nssrate
        growtime = self.grow_time
        resttime = self.rest_time
        k = self.k
        dm1 = self.dm1
        dm2 = self.dm2

        if (t<growtime):
            ps = -(((-1 + mgrow)*Nm02*(1 + dm2*t) + Nm01*(-1 + mgrow + dm1*(growtime + (-1 + mgrow)*t)))/(growtime*(-1 + k)*Ns0))
            ds = -((k*((-1 + mgrow)*Nm02*(1 + dm2*t) + Nm01*(-1 + mgrow + dm1*(growtime + (-1 + mgrow)*t))))/(growtime*(-1 + k)*Ns0))
            t1 = (Nm01*(-1 + mgrow + dm1*(growtime + (-1 + mgrow)*t)))/((-1 + mgrow)*Nm02*(1 + dm2*t) + Nm01*(-1 + mgrow + dm1*(growtime + (-1 + mgrow)*t)))
            Nm1 = Nm01*(mgrow - 1)/growtime*t+Nm01
            Nm2 = Nm02*(mgrow - 1)/growtime*t+Nm02
            Ns = Ns0
        elif (t<growtime+resttime):
            ps = (mgrow*(dm1*Nm01 + dm2*Nm02))/(Ns0 - k*Ns0)
            ds = (k*mgrow*(dm1*Nm01 + dm2*Nm02))/(Ns0 - k*Ns0)
            t1 = (dm1*Nm01)/(dm1*Nm01 + dm2*Nm02)
            Nm1= Nm01*mgrow
            Nm2= Nm02*mgrow
            Ns = Ns0 
        else:
            ps = -((mgrow*(dm1*Nm01 - Nm02*Nm2srate + dm2*Nm02*(1 + Nm2srate*(growtime + resttime - t))))/((-1 + k)*Ns0*(1 + Nssrate*(growtime + resttime - t))))
            ds = (-(dm1*k*mgrow*Nm01) + k*mgrow*Nm02*Nm2srate + (-1 + k)*Ns0*Nssrate - dm2*k*mgrow*Nm02*(1 + Nm2srate*(growtime + resttime - t)))/((-1 + k)*Ns0*(1 + Nssrate*(growtime + resttime - t)))
            t1 = (dm1*Nm01)/(dm1*Nm01 + Nm02*(-Nm2srate + dm2*(1 + Nm2srate*(growtime + resttime - t))))
            Nm1 = Nm01*mgrow
            Nm2 = Nm02*mgrow  - Nm02*mgrow*Nm2srate*(-growtime - resttime + t)
            Ns = Ns0  -Ns0*Nssrate*(-growtime - resttime + t)
        iparas = {'ps' : ps,'ds':ds,'t1':t1,'Ns':Ns,'Nm1':Nm1,'Nm2':Nm2}
        if Nm1 <= 0:
            raise ImplicitParametersOutOfRange('Nm1 is negative',iparas)
        if Nm2 <= 0:
            raise ImplicitParametersOutOfRange('Nm2 is negative',iparas)
        if Ns <= 0:
            raise ImplicitParametersOutOfRange('Ns is negative',iparas)
        if ps < 0:
            raise ImplicitParametersOutOfRange('ps is negative',iparas)
        if t1 < 0 or t1 > 1:
            raise ImplicitParametersOutOfRange('t1 is not in 0-1',iparas)
        if ds < 0:
            raise ImplicitParametersOutOfRange('ds is negative',iparas)
        if ps > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        if ds > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ds"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        stem = self.stem
        muscle1 = self.muscle1
        muscle2 = self.muscle2

        k = self.k
        dm1 = self.dm1
        dm2 = self.dm2
        iparas = self.calc_implicit_parameters(t)
        ps = iparas['ps']
        ds = iparas['ds']
        t1 = iparas['t1']
        Nm1 = iparas['Nm1']
        Nm2 = iparas['Nm2']
        Ns = iparas['Ns']


        M_new[stem] = ((1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - M[stem]))/2.
        M_new[muscle1] = -((-1 + k)*ps*t1*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle1] + M[stem])*Ns)/(2.*Nm1)
        M_new[muscle2] =((-1 + k)*ps*(-1 + t1)*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle2] + M[stem])*Ns)/(2.*Nm2)

        return M_new.ravel()
models_list.append(SchrinkType2Skonst)

class SchrinkType2Skonst_fixed(SchrinkType2Skonst):
    def __init__(self):
        default_parameters=dict(k=0,dm1=-2,dm2=-1.5)
        self.logparas = ['dm1','dm2']
        self.linparas = ['k']
        self.mgrow = MGROW
        self.grow_time = GROW_TIME
        self.Nm2srate = NM2SRATE
        self.Nssrate = NSSRATE
        self.rest_time = RESTTIME
        self.N0 = 1

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k'] = (0, 1)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(SchrinkType2Skonst_fixed)


class dm1_0SchrinkType2Skonst_fixed(SchrinkType2Skonst):
    def __init__(self):
        default_parameters=dict(k=0,dm2=-1)
        self.logparas = ['dm2']
        self.linparas = ['k']
        self.mgrow = MGROW
        self.grow_time = GROW_TIME
        self.Nm2srate = NM2SRATE
        self.Nssrate = NSSRATE
        self.rest_time = RESTTIME
        self.dm1 = 0
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k'] = (0, 1)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(dm1_0SchrinkType2Skonst_fixed)


'''
class SlinshrinkMlinshrink(MB2):
    def __init__(self):
        default_parameters=dict(k=0, dm=-2, grow_factor=0, grow_time=0, ktime=0, N0=0, M_shrink=-2, S_shrink=-4)
        self.Ns0 = 1
        self.logparas = ['dm','grow_factor','M_shrink','S_shrink','N0']
        self.linparas = ['k','grow_time','ktime']
        
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        limit['N0'] = (0.001, 1000)
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
        default_parameters=dict(k=0, dm=-2, N0=0, S_shrink=-4)
        self.Ns0 = 1
        self.logparas = ['dm','S_shrink','N0']
        self.linparas = ['k']

        self.grow_factor = GROW_FACTOR
        self.grow_time = GROW_TIME
        self.ktime = KTIME
        self.M_shrink = M_SHRINK

        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['N0'] = (0.001, 1000)
        limit['k'] = (0, 1)


        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(SlinshrinkMlinshrink_fixed)


'''


