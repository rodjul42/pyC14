import numpy as np
import pandas as pd
from .base import model_base, Catm
from scipy.interpolate import UnivariateSpline
from ..tools import trans_arcsin, trans_sin,ImplicitParametersOutOfRange
import pkg_resources
global_error = 0.5
global_limit = (10**-6, 10**3)

GROW_TIME = 26
MGROW = 9.18
REST_TIME = 20

NMSRATE = 0.004

NM2SRATE = 0.008
NSSRATE = 0.001

KMAX = 0.8

#EXPA= 8.37706842364683
#EXPB= 3.54099043111506
#EXPX0=13.1412672478863
#EXPY0=0.801045348966687
EXPA = 19.111480330228474 
EXPB = 2.9184430157974512
EXPX0= 11.745328842637585
EXPY0= 0.6644364078454283

DEFAULT_FIXED = dict(grow_time=GROW_TIME,mgrow=MGROW,rest_time=REST_TIME,Nmsrate=NMSRATE,Nm2srate=NM2SRATE,Nssrate=NSSRATE,
            EXPa=EXPA,EXPb=EXPB,EXPx0=EXPX0,EXPy0=EXPY0)


STEMPP = 2.4/100.0

models_list = []
models_list_fixed = []

class MB(model_base):
    def __init__(self):
        pass



    def measurement_model(self, result_sim, data):
        return result_sim['muscle']

class MB2(MB):
    def calc_initial_parameters(self):
        self.Nm0 = self.N0
        Nmfunc1_0,_ = self.get_Nm(0,self.Nm0)
        Nmfunc1_40,_ = self.get_Nm(40,self.Nm0)
        growf = Nmfunc1_40/Nmfunc1_0
        p0 = STEMPP*growf/(1-STEMPP + growf*STEMPP )
        self.Ns0 = p0 * self.N0/(1-p0)

class MBt2(model_base):
    def calc_initial_parameters(self):
        self.Nm01 = self.N0*0.33333333
        self.Nm02 = self.N0*0.66666666
        Nmfunc1_0,_ = self.get_Nm(0,self.Nm01)
        Nmfunc2_0,_ = self.get_Nm2(0,self.Nm02)
        Nmfunc1_40,_ = self.get_Nm(40,self.Nm01)
        Nmfunc2_40,_ = self.get_Nm2(40,self.Nm02)
        growf = (Nmfunc1_40+Nmfunc2_40)/(Nmfunc1_0+Nmfunc2_0)
        p0 = STEMPP*growf/(1-STEMPP + growf*STEMPP )
        self.Ns0 = p0 * self.N0 / (1-p0)



    def measurement_model(self, result_sim, data):
        Nm1 = np.array([self.get_Nm(t,self.Nm01)[0] for t in data.age])
        Nm2 = np.array([self.get_Nm2(t,self.Nm01)[0] for t in data.age])
        return (result_sim['muscle1']*Nm1 + result_sim['muscle2']*Nm2)/(Nm1+Nm2)

class MBt2s2(MBt2):
    def calc_initial_parameters(self):
        self.Nm01 = self.N0*0.33333333
        self.Nm02 = self.N0*0.66666666
        Nmfunc1_0,_ = self.get_Nm(0,self.Nm01)
        Nmfunc2_0,_ = self.get_Nm2(0,self.Nm02)
        Nmfunc1_40,_ = self.get_Nm(40,self.Nm01)
        Nmfunc2_40,_ = self.get_Nm2(40,self.Nm02)
        growf = (Nmfunc1_40)/(Nmfunc1_0)
        p0 = STEMPP*growf/(1-STEMPP + growf*STEMPP )

        growf2 = (Nmfunc2_40)/(Nmfunc2_0)
        p02 = STEMPP*growf2/(1-STEMPP + growf2*STEMPP )
        self.Ns01 = p0 * self.Nm01 / (1-p0)
        self.Ns02 = p02 * self.Nm02 / (1-p02)



class MBt2_simple(model_base):
    def calc_initial_parameters(self):
        self.Nm01 = 0.33333333
        self.Nm02 = 0.66666666




    def measurement_model(self, result_sim, data):
        Nm1 = np.array([self.get_Nm(t,self.Nm01)[0] for t in data.age])
        Nm2 = np.array([self.get_Nm2(t,self.Nm01)[0] for t in data.age])
        return (result_sim['muscle1']*Nm1 + result_sim['muscle2']*Nm2)/(Nm1+Nm2)

'''
------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------------
'''
def grow_lin(t,N0,mgrow,growtime):
    if (t<growtime):
        Nm = N0*(mgrow - 1)/growtime*t + N0
        dtNm = N0*(mgrow - 1)/growtime
    else:
        Nm =  mgrow*N0
        dtNm = 0
    return Nm,dtNm

def growshrink_lin(t,N0,mgrow,growtime,resttime,shrinkrate):
    if (t<growtime+resttime):
        Nm,dtNm = grow_lin(t,N0,mgrow,growtime)
    else:
        Nm = (mgrow - (t-growtime-resttime)*shrinkrate)*N0
        dtNm = - shrinkrate*N0
    return Nm,dtNm

def growS_lin(t,N0,mgrow,growtime):
    st=5
    a= -(-1 + mgrow)/(4.*growtime*st)
    b= ((-1 + mgrow)*(growtime + st))/(2.*growtime*st)
    c = (2 + 2*mgrow + (growtime - growtime*mgrow)/st + (st - mgrow*st)/growtime)/4.
    if (t<growtime-st):
        Nm = (mgrow - 1)/growtime*t + 1
        dtNm = (mgrow - 1)/growtime
    elif (t<growtime+st):
        Nm = a*t*t+b*t+c
        dtNm = 2*a*t+b
    else:
        Nm =  mgrow
        dtNm = 0
    return N0*Nm,N0*dtNm


def growshrinkS_lin(t,N0,mgrow,growtime,resttime,shrinkrate):
    if (t<growtime+resttime):
        Nm,dtNm = growS_lin(t,N0,mgrow,growtime)
    else:
        Nm = (mgrow - (t-growtime-resttime)*shrinkrate)*N0
        dtNm = - shrinkrate*N0
    return Nm,dtNm


def growlog(t,N0,EXPa,EXPb,EXPx0,EXPy0):
    tmp=(1+np.exp(-(t-EXPx0)/EXPb))
    f = EXPy0+EXPa/tmp
    df= EXPa*np.exp(-(t-EXPx0)/EXPb)/(tmp*tmp*EXPb)
    return f*N0,df*N0

def grow_old(t,N0,EXPa,EXPb,EXPx0,EXPy0,T0,A,B):
    tmp2=(1+np.exp(-(t-T0)/A))

    tmp=(1+np.exp(-(t-EXPx0)/EXPb))
    f = EXPy0+EXPa/tmp + B/tmp2
    df= EXPa*np.exp(-(t-EXPx0)/EXPb)/(tmp*tmp*EXPb) + B*np.exp(-(t-T0)/A)/(tmp2*tmp2*A)
    return f*N0,df*N0

class Nm_grow():
    Nm_paras = dict(mgrow=14, grow_time=20)
    Nm_limit = dict(grow_time=(1,40))
    Nm_logparas = {'mgrow'}
    Nm_linparas = {'grow_time'}
    def __init__(self):
        pass

    def get_Nm(self,t,Nm0):
        growtime = self.grow_time
        mgrow = self.mgrow
        return grow_lin(t,Nm0,mgrow,growtime)

class Nm_growshrink():
    Nm_paras = dict(mgrow=14, grow_time=20,Nmsrate=0.002,rest_time=10)
    Nm_limit = dict(grow_time=(1,40),rest_time=(1,40))
    Nm_logparas = {'mgrow','Nmsrate'}
    Nm_linparas = {'grow_time','rest_time'}
    def __init__(self):
        pass

    def get_Nm(self,t,Nm0):
        growtime = self.grow_time
        mgrow = self.mgrow
        resttime = self.rest_time
        Nmsrate = self.Nmsrate
        return growshrink_lin(t,Nm0,mgrow,growtime,resttime,Nmsrate)


class Nm2_grow():
    Nm2_paras = dict(mgrow=14, grow_time=20)
    Nm2_limit = dict(grow_time=(1,40))
    Nm2_logparas = {'mgrow'}
    Nm2_linparas = {'grow_time'}
    def __init__(self):
        pass

    def get_Nm2(self,t,Nm0):
        growtime = self.grow_time
        mgrow = self.mgrow
        return grow_lin(t,Nm0,mgrow,growtime)

class Nm2_growshrink():
    Nm2_paras = dict(mgrow=0, grow_time=0,Nm2srate=0.001,rest_time=0)
    Nm2_limit = dict(grow_time=(1,40),rest_time=(1,40))
    Nm2_logparas = {'mgrow','Nm2srate'}
    Nm2_linparas = {'grow_time','rest_time'}
    def __init__(self):
        pass

    def get_Nm2(self,t,Nm0):
        growtime = self.grow_time
        mgrow = self.mgrow
        resttime = self.rest_time
        Nm2srate = self.Nm2srate
        return growshrink_lin(t,Nm0,mgrow,growtime,resttime,Nm2srate)

class Nm_growS():
    Nm_paras = dict(mgrow=14, grow_time=20)
    Nm_limit = dict(grow_time=(1,40))
    Nm_logparas = {'mgrow'}
    Nm_linparas = {'grow_time'}
    def __init__(self):
        pass

    def get_Nm(self,t,Nm0):
        growtime = self.grow_time
        mgrow = self.mgrow
        return growS_lin(t,Nm0,mgrow,growtime)


class Nm_growshrinkS():
    Nm_paras = dict(mgrow=14, grow_time=20,Nmsrate=0.002,rest_time=10)
    Nm_limit = dict(grow_time=(1,40),rest_time=(1,40))
    Nm_logparas = {'mgrow','Nmsrate'}
    Nm_linparas = {'grow_time','rest_time'}
    def __init__(self):
        pass

    def get_Nm(self,t,Nm0):
        growtime = self.grow_time
        mgrow = self.mgrow
        resttime = self.rest_time
        Nmsrate = self.Nmsrate
        return growshrinkS_lin(t,Nm0,mgrow,growtime,resttime,Nmsrate)


class Nm2_growshrinkS():
    Nm2_paras = dict(mgrow=14, grow_time=20,Nm2srate=0.001,rest_time=10)
    Nm2_limit = dict(grow_time=(1,40),rest_time=(1,40))
    Nm2_logparas = {'mgrow','Nm2srate'}
    Nm2_linparas = {'grow_time','rest_time'}
    def __init__(self):
        pass

    def get_Nm2(self,t,Nm0):
        growtime = self.grow_time
        mgrow = self.mgrow
        resttime = self.rest_time
        Nm2srate = self.Nm2srate
        return growshrinkS_lin(t,Nm0,mgrow,growtime,resttime,Nm2srate)


class Nm_growlog():
    Nm_paras = dict(EXPa=0, EXPb=0,EXPx0=0.001,EXPy0=0)
    Nm_limit = dict(EXPa=(0.1,40),EXPb=(0.1,40),EXPx0=(0.1,40),EXPy0=(-10,10))
    Nm_logparas = {}
    Nm_linparas = {'EXPa','EXPb','EXPx0','EXPy0'}
    def __init__(self):
        pass

    def get_Nm(self,t,Nm0):
        EXPa = self.EXPa
        EXPb = self.EXPb
        EXPx0 = self.EXPx0
        EXPy0 = self.EXPy0
        return growlog(t,Nm0,EXPa,EXPb,EXPx0,EXPy0)

class Nm_grow_old():
    Nm_paras = dict(EXPa=0, EXPb=0,EXPx0=0.001,EXPy0=0,T0=30,A=10,B=1)
    Nm_limit = dict(EXPa=(0.1,40),EXPb=(0.1,40),EXPx0=(0.1,40),EXPy0=(-10,10),T0=(20,50),A=(1,40),B=(0.01,40))
    Nm_logparas = {}
    Nm_linparas = {'EXPa','EXPb','EXPx0','EXPy0','T0','A','B'}
    def __init__(self):
        pass

    def get_Nm(self,t,Nm0):
        EXPa = self.EXPa
        EXPb = self.EXPb
        EXPx0 = self.EXPx0
        EXPy0 = self.EXPy0
        T0 = self.T0
        A = self.A
        B = self.B
        return grow_old(t,Nm0,EXPa,EXPb,EXPx0,EXPy0,T0,A,B)

class Nm2_growlog():
    Nm2_paras = dict(EXPa=0, EXPb=0,EXPx0=0.001,EXPy0=0)
    Nm2_limit = dict(EXPa=(0.1,40),EXPb=(0.1,40),EXPx0=(0.1,40),EXPy0=(-10,10))
    Nm2_logparas = {}
    Nm2_linparas = {'EXPa','EXPb','EXPx0','EXPy0'}
    def __init__(self):
        pass

    def get_Nm2(self,t,Nm0):
        EXPa = self.EXPa
        EXPb = self.EXPb
        EXPx0 = self.EXPx0
        EXPy0 = self.EXPy0
        return growlog(t,Nm0,EXPa,EXPb,EXPx0,EXPy0)


class Ns_const():
    Ns_paras = dict()
    Ns_limit = dict()
    Ns_logparas = set()
    Ns_linparas = set()
    def __init__(self):
        pass

    def get_Ns(self,t,Ns0):
        return Ns0,0


class Ns2_const():
    Ns2_paras = dict()
    Ns2_limit = dict()
    Ns2_logparas = set()
    Ns2_linparas = set()
    def __init__(self):
        pass

    def get_Ns2(self,t,Ns0):
        return Ns0,0

class Ns_shrink():
    Ns_paras = dict(Nssrate=0.001,rest_time=0,grow_time=0)
    Ns_limit = dict(grow_time=(1,40),rest_time=(1,40))
    Ns_logparas = {'Nssrate'}
    Ns_linparas = set()
    def __init__(self):
        pass

    def get_Ns(self,t,Ns0):
        growtime = self.grow_time
        resttime = self.rest_time
        Nssrate = self.Nssrate
        if (t<growtime+resttime):
            Ns = Ns0
            dtNs = 0
        else:
            Ns =  Ns0 - (t-growtime-resttime)*Nssrate*Ns0
            dtNs = -Nssrate*Ns0
        return Ns,dtNs


'''
------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------
Models
------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------
'''

class simple(MB):
    def __init__(self):
        default_parameters=dict(dm=10e-3)
        default_parameters.update(self.Nm_paras)
        self.logparas = {'dm'}
        self.logparas = self.logparas.union(self.Nm_logparas)
        self.linparas = set()
        self.linparas = self.linparas.union(self.Nm_linparas)
        
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit.update(self.Nm_limit)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        dm = self.dm
        Nmfunc,dtNmfunc = self.get_Nm(t,1)
        
        pm = dm*Nmfunc + dtNmfunc
        Nm = Nmfunc
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



class simpleM2(MBt2_simple):
    def __init__(self):
        default_parameters=dict(dm1=10e-2,dm2=10e-2)
        default_parameters.update(self.Nm_paras)
        default_parameters.update(self.Nm2_paras)
        self.logparas = {'dm1','dm2'}
        self.linparas = set()
        self.logparas = self.logparas.union(self.Nm_logparas)
        self.logparas = self.logparas.union(self.Nm2_logparas)
        self.linparas = self.linparas.union(self.Nm_linparas)
        self.linparas = self.linparas.union(self.Nm2_linparas)
        
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit.update(self.Nm_limit)
        limit.update(self.Nm2_limit)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        dm1 = self.dm1
        dm2 = self.dm2

        Nmfunc1,dtNmfunc1 = self.get_Nm(t,self.Nm01)
        Nmfunc2,dtNmfunc2 = self.get_Nm2(t,self.Nm02)

        fm1 = dm1*Nmfunc1 + dtNmfunc1
        fm2 = dm2*Nmfunc2 + dtNmfunc2
        iparas = {'fm1' :fm1,'fm2':fm2,'Nm1':Nmfunc1,'Nm2':Nmfunc2}
        if Nmfunc1 <= 0:
            raise ImplicitParametersOutOfRange('Nm1 is negative',iparas)
        if Nmfunc2 <= 0:
            raise ImplicitParametersOutOfRange('Nm2 is negative',iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        muscle1 = self.muscle1
        muscle2 = self.muscle2

        iparas = self.calc_implicit_parameters(t)
        fm1 = iparas['fm1']
        fm2 = iparas['fm2']
        Nm1 = iparas['Nm1']
        Nm2 = iparas['Nm2']

        M_new[muscle1] = (fm1*(self.Catm.lin(t + self.Dbirth) - M[muscle1]))/Nm1
        M_new[muscle2] = (fm2*(self.Catm.lin(t + self.Dbirth) - M[muscle2]))/Nm2

        return M_new.ravel()

class dm1_0simpleM2(MBt2_simple):
    def __init__(self):
        default_parameters=dict(dm2=10e-2)
        default_parameters.update(self.Nm_paras)
        default_parameters.update(self.Nm2_paras)
        self.logparas = {'dm2'}
        self.linparas = set()
        self.logparas = self.logparas.union(self.Nm_logparas)
        self.logparas = self.logparas.union(self.Nm2_logparas)
        self.linparas = self.linparas.union(self.Nm_linparas)
        self.linparas = self.linparas.union(self.Nm2_linparas)
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit.update(self.Nm_limit)
        limit.update(self.Nm2_limit)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        dm2 = self.dm2

        Nmfunc1,dtNmfunc1 = self.get_Nm(t,self.Nm01)
        Nmfunc2,dtNmfunc2 = self.get_Nm2(t,self.Nm02)

        fm1 = dtNmfunc1
        fm2 = dm2*Nmfunc2 + dtNmfunc2
        iparas = {'fm1' :fm1,'fm2':fm2,'Nm1':Nmfunc1,'Nm2':Nmfunc2}
        if Nmfunc1 <= 0:
            raise ImplicitParametersOutOfRange('Nm1 is negative',iparas)
        if Nmfunc2 <= 0:
            raise ImplicitParametersOutOfRange('Nm2 is negative',iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        muscle1 = self.muscle1
        muscle2 = self.muscle2

        iparas = self.calc_implicit_parameters(t)
        fm1 = iparas['fm1']
        fm2 = iparas['fm2']
        Nm1 = iparas['Nm1']
        Nm2 = iparas['Nm2']

        M_new[muscle1] = (fm1*(self.Catm.lin(t + self.Dbirth) - M[muscle1]))/Nm1
        M_new[muscle2] = (fm2*(self.Catm.lin(t + self.Dbirth) - M[muscle2]))/Nm2

        return M_new.ravel()

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


class SM(MB2):
    def __init__(self):
        default_parameters=dict(dm=10e-2,k=0.5)
        default_parameters.update(self.Nm_paras)
        default_parameters.update(self.Ns_paras)
        self.logparas = {'dm'}
        self.logparas = self.logparas.union(self.Nm_logparas)
        self.logparas = self.logparas.union(self.Ns_logparas)
        self.linparas = {'k'}
        self.linparas = self.linparas.union(self.Nm_linparas)
        self.linparas = self.linparas.union(self.Ns_linparas)
        self.N0 = 1  
        
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit.update(self.Nm_limit)
        limit.update(self.Ns_limit)
        limit['k'] = (0, KMAX)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        k = self.k
        dm = self.dm

        Nmfunc,dtNmfunc = self.get_Nm(t,self.Nm0)
        Nsfunc,dtNsfunc = self.get_Ns(t,self.Ns0)

        ps = (dm*Nmfunc + dtNmfunc)/(Nsfunc - k*Nsfunc)
        ds = (dtNsfunc - k*(dm*Nmfunc + dtNmfunc + dtNsfunc))/((-1 + k)*Nsfunc)

        iparas = {'ps' : ps,'ds':ds,'Ns':Nsfunc,'Nm':Nmfunc}
        if Nmfunc <= 0:
            raise ImplicitParametersOutOfRange('Nm is negative',iparas)
        if Nsfunc <= 0:
            raise ImplicitParametersOutOfRange('Ns is negative',iparas)
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



class k0SM(SM):
    def __init__(self):
        default_parameters=dict(dm=10e-2)
        default_parameters.update(self.Nm_paras)
        default_parameters.update(self.Ns_paras)
        self.logparas = {'dm'}
        self.logparas = self.logparas.union(self.Nm_logparas)
        self.logparas = self.logparas.union(self.Ns_logparas)
        self.linparas = set()
        self.linparas = self.linparas.union(self.Nm_linparas)
        self.linparas = self.linparas.union(self.Ns_linparas)
        self.N0 = 1  
        self.k = 0        
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit.update(self.Nm_limit)
        limit.update(self.Ns_limit)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



class dm0SM(MB2):
    def __init__(self):
        default_parameters=dict(k=0.5)
        default_parameters.update(self.Nm_paras)
        default_parameters.update(self.Ns_paras)
        self.logparas = set()
        self.logparas = self.logparas.union(self.Nm_logparas)
        self.logparas = self.logparas.union(self.Ns_logparas)
        self.linparas = {'k'}
        self.linparas = self.linparas.union(self.Nm_linparas)
        self.linparas = self.linparas.union(self.Ns_linparas)
        self.N0 = 1  
        
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit.update(self.Nm_limit)
        limit.update(self.Ns_limit)
        limit['k'] = (0, KMAX)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        k = self.k

        Nmfunc,dtNmfunc = self.get_Nm(t,self.Nm0)
        Nsfunc,dtNsfunc = self.get_Ns(t,self.Ns0)

  
        ps = dtNmfunc/(Nsfunc - k*Nsfunc)
        ds = -((k*dtNmfunc + (-1 + k)*dtNsfunc)/((-1 + k)*Nsfunc))
        
        iparas = {'ps' : ps,'ds':ds,'Ns':Nsfunc,'Nm':Nmfunc}
        if Nmfunc <= 0:
            raise ImplicitParametersOutOfRange('Nm is negative',iparas)
        if Nsfunc <= 0:
            raise ImplicitParametersOutOfRange('Ns is negative',iparas)
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


class SM2(MB2):
    def __init__(self):
        default_parameters=dict(dm=10e-2,ds=0.1)
        default_parameters.update(self.Nm_paras)
        default_parameters.update(self.Ns_paras)
        self.logparas = {'dm','ds'}
        self.logparas = self.logparas.union(self.Nm_logparas)
        self.logparas = self.logparas.union(self.Ns_logparas)
        self.linparas = set()
        self.linparas = self.linparas.union(self.Nm_linparas)
        self.linparas = self.linparas.union(self.Ns_linparas)
        self.N0 = 1  
        
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit.update(self.Nm_limit)
        limit.update(self.Ns_limit)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        ds = self.ds
        dm = self.dm

        Nmfunc,dtNmfunc = self.get_Nm(t,self.Nm0)
        Nsfunc,dtNsfunc = self.get_Ns(t,self.Ns0)

  
        ps = (dm*Nmfunc + ds*Nsfunc + dtNmfunc + dtNsfunc)/Nsfunc
        k = (ds*Nsfunc + dtNsfunc)/(dm*Nmfunc + ds*Nsfunc + dtNmfunc + dtNsfunc)
        
        iparas = {'ps' : ps,'k':k,'Ns':Nsfunc,'Nm':Nmfunc}
        if Nmfunc <= 0:
            raise ImplicitParametersOutOfRange('Nm is negative',iparas)
        if Nsfunc <= 0:
            raise ImplicitParametersOutOfRange('Ns is negative',iparas)
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


class dm0SM2(SM2):
    def __init__(self):
        default_parameters=dict(ds=0.1)
        default_parameters.update(self.Nm_paras)
        default_parameters.update(self.Ns_paras)
        self.logparas = {'ds'}
        self.logparas = self.logparas.union(self.Nm_logparas)
        self.logparas = self.logparas.union(self.Ns_logparas)
        self.linparas = set()
        self.linparas = self.linparas.union(self.Nm_linparas)
        self.linparas = self.linparas.union(self.Ns_linparas)
        self.N0 = 1  
        
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit.update(self.Nm_limit)
        limit.update(self.Ns_limit)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        ds = self.ds

        Nmfunc,dtNmfunc = self.get_Nm(t,self.Nm0)
        Nsfunc,dtNsfunc = self.get_Ns(t,self.Ns0)

  
        ps = (ds*Nsfunc + dtNmfunc + dtNsfunc)/Nsfunc
        k = 1 - dtNmfunc/(ds*Nsfunc + dtNmfunc + dtNsfunc)
        
        iparas = {'ps' : ps,'k':k,'Ns':Nsfunc,'Nm':Nmfunc}
        if Nmfunc <= 0:
            raise ImplicitParametersOutOfRange('Nm is negative',iparas)
        if Nsfunc <= 0:
            raise ImplicitParametersOutOfRange('Ns is negative',iparas)
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


        M_new[stem] =  ((1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - M[stem]))/2
        M_new[muscle] = -((-1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)
                        
        return M_new.ravel()



class S2M(MBt2):
    def __init__(self):
        default_parameters=dict(k=0.5,dm1=10e-2,dm2=10e-2)
        default_parameters.update(self.Nm_paras)
        default_parameters.update(self.Nm2_paras)
        default_parameters.update(self.Ns_paras)
        self.logparas = {'dm1','dm2'}
        self.logparas = self.logparas.union(self.Nm_logparas)
        self.logparas = self.logparas.union(self.Nm2_logparas)
        self.logparas = self.logparas.union(self.Ns_logparas)
        self.linparas = {'k'}
        self.linparas = self.linparas.union(self.Nm_linparas)
        self.linparas = self.linparas.union(self.Nm2_linparas)
        self.linparas = self.linparas.union(self.Ns_linparas)
        self.N0 = 1  
        
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit.update(self.Nm_limit)
        limit.update(self.Nm2_limit)
        limit.update(self.Ns_limit)
        limit['k'] = (0, KMAX)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        k = self.k
        dm1 = self.dm1
        dm2 = self.dm2

        Nmfunc1,dtNmfunc1 = self.get_Nm(t,self.Nm01)
        Nmfunc2,dtNmfunc2 = self.get_Nm2(t,self.Nm02)
        Nsfunc,dtNsfunc = self.get_Ns(t,self.Ns0)


        
        ps = (dm1*Nmfunc1 + dm2*Nmfunc2 + dtNmfunc1 + dtNmfunc2)/(Nsfunc - k*Nsfunc)
        ds = (dtNsfunc - k*(dm1*Nmfunc1 + dm2*Nmfunc2 + dtNmfunc1 + dtNmfunc2 + dtNsfunc))/((-1 + k)*Nsfunc)
        t1 = (dm1*Nmfunc1 + dtNmfunc1)/(dm1*Nmfunc1 + dm2*Nmfunc2 + dtNmfunc1 + dtNmfunc2)
        
        iparas = {'ps' : ps,'ds':ds,'t1':t1,'Ns':Nsfunc,'Nm1':Nmfunc1,'Nm2':Nmfunc2}
        if Nmfunc1 <= 0:
            raise ImplicitParametersOutOfRange('Nm1 is negative',iparas)
        if Nmfunc2 <= 0:
            raise ImplicitParametersOutOfRange('Nm2 is negative',iparas)
        if Nsfunc <= 0:
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


class k0S2M(S2M):
    def __init__(self):
        default_parameters=dict(dm1=10e-2,dm2=10e-2)
        default_parameters.update(self.Nm_paras)
        default_parameters.update(self.Nm2_paras)
        default_parameters.update(self.Ns_paras)
        self.logparas = {'dm1','dm2'}
        self.logparas = self.logparas.union(self.Nm_logparas)
        self.logparas = self.logparas.union(self.Nm2_logparas)
        self.logparas = self.logparas.union(self.Ns_logparas)
        self.linparas = set()
        self.linparas = self.linparas.union(self.Nm_linparas)
        self.linparas = self.linparas.union(self.Nm2_linparas)
        self.linparas = self.linparas.union(self.Ns_linparas)
        self.N0 = 1  
        self.k = 0
        
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit.update(self.Nm_limit)
        limit.update(self.Nm2_limit)
        limit.update(self.Ns_limit)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



class dm1_0S2M(MBt2):
    def __init__(self):
        default_parameters=dict(k=0.5,dm2=10e-2)
        default_parameters.update(self.Nm_paras)
        default_parameters.update(self.Nm2_paras)
        default_parameters.update(self.Ns_paras)
        self.logparas = {'dm2'}
        self.logparas = self.logparas.union(self.Nm_logparas)
        self.logparas = self.logparas.union(self.Nm2_logparas)
        self.logparas = self.logparas.union(self.Ns_logparas)
        self.linparas = {'k'}
        self.linparas = self.linparas.union(self.Nm_linparas)
        self.linparas = self.linparas.union(self.Nm2_linparas)
        self.linparas = self.linparas.union(self.Ns_linparas)
        self.N0 = 1  
        
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit.update(self.Nm_limit)
        limit.update(self.Nm2_limit)
        limit.update(self.Ns_limit)
        limit['k'] = (0, KMAX)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        k = self.k
        dm2 = self.dm2

        Nmfunc1,dtNmfunc1 = self.get_Nm(t,self.Nm01)
        Nmfunc2,dtNmfunc2 = self.get_Nm2(t,self.Nm02)
        Nsfunc,dtNsfunc = self.get_Ns(t,self.Ns0)
        
        ps = (dm2*Nmfunc2 + dtNmfunc1 + dtNmfunc2)/(Nsfunc - k*Nsfunc)
        ds = (dtNsfunc - k*(dm2*Nmfunc2 + dtNmfunc1 + dtNmfunc2 + dtNsfunc))/((-1 + k)*Nsfunc)
        t1 = dtNmfunc1/(dm2*Nmfunc2 + dtNmfunc1 + dtNmfunc2)
        
        iparas = {'ps' : ps,'ds':ds,'t1':t1,'Ns':Nsfunc,'Nm1':Nmfunc1,'Nm2':Nmfunc2}
        if Nmfunc1 <= 0:
            raise ImplicitParametersOutOfRange('Nm1 is negative',iparas)
        if Nmfunc2 <= 0:
            raise ImplicitParametersOutOfRange('Nm2 is negative',iparas)
        if Nsfunc <= 0:
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

class dm1_0_dm2_0S2M(MBt2):
    def __init__(self):
        default_parameters=dict(k=0.5)
        default_parameters.update(self.Nm_paras)
        default_parameters.update(self.Nm2_paras)
        default_parameters.update(self.Ns_paras)
        self.logparas = set()
        self.logparas = self.logparas.union(self.Nm_logparas)
        self.logparas = self.logparas.union(self.Nm2_logparas)
        self.logparas = self.logparas.union(self.Ns_logparas)
        self.linparas = {'k'}
        self.linparas = self.linparas.union(self.Nm_linparas)
        self.linparas = self.linparas.union(self.Nm2_linparas)
        self.linparas = self.linparas.union(self.Ns_linparas)
        self.N0 = 1  
        
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit.update(self.Nm_limit)
        limit.update(self.Nm2_limit)
        limit.update(self.Ns_limit)
        limit['k'] = (0, KMAX)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



    def calc_implicit_parameters(self, t):
        k = self.k

        Nmfunc1,dtNmfunc1 = self.get_Nm(t,self.Nm01)
        Nmfunc2,dtNmfunc2 = self.get_Nm2(t,self.Nm02)
        Nsfunc,dtNsfunc = self.get_Ns(t,self.Ns0)
        
        ps = (dtNmfunc1 + dtNmfunc2)/(Nsfunc - k*Nsfunc)
        ds = (dtNsfunc - k*(dtNmfunc1 + dtNmfunc2 + dtNsfunc))/((-1 + k)*Nsfunc)
        if dtNmfunc1 + dtNmfunc2 == 0:
            t1 = 0.5 #should not matter since no change in muscle population
        else:
            t1 = dtNmfunc1/(dtNmfunc1 + dtNmfunc2)
        
        iparas = {'ps' : ps,'ds':ds,'t1':t1,'Ns':Nsfunc,'Nm1':Nmfunc1,'Nm2':Nmfunc2}
        if Nmfunc1 <= 0:
            raise ImplicitParametersOutOfRange('Nm1 is negative',iparas)
        if Nmfunc2 <= 0:
            raise ImplicitParametersOutOfRange('Nm2 is negative',iparas)
        if Nsfunc <= 0:
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



class S2_M2(MBt2s2):
    def __init__(self):
        default_parameters=dict(k1=0.5,k2=0.5,dm1=10e-2,dm2=10e-2)
        default_parameters.update(self.Nm_paras)
        default_parameters.update(self.Nm2_paras)
        default_parameters.update(self.Ns_paras)
        default_parameters.update(self.Ns2_paras)
        self.logparas = {'dm1','dm2'}
        self.logparas = self.logparas.union(self.Nm_logparas)
        self.logparas = self.logparas.union(self.Nm2_logparas)
        self.logparas = self.logparas.union(self.Ns_logparas)
        self.logparas = self.logparas.union(self.Ns2_logparas)
        self.linparas = {'k1','k2'}
        self.linparas = self.linparas.union(self.Nm_linparas)
        self.linparas = self.linparas.union(self.Nm2_linparas)
        self.linparas = self.linparas.union(self.Ns_linparas)
        self.linparas = self.linparas.union(self.Ns2_linparas)
        self.N0 = 1  
        
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit.update(self.Nm_limit)
        limit.update(self.Nm2_limit)
        limit.update(self.Ns_limit)
        limit.update(self.Ns2_limit)
        limit['k1'] = (0, KMAX)
        limit['k2'] = (0, KMAX)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem1','stem2','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)


    def calc_implicit_parameters(self, t):
        k = self.k1
        dm = self.dm1

        k2 = self.k2
        dm2 = self.dm2

        Nmfunc,dtNmfunc = self.get_Nm(t,self.Nm01)
        Nsfunc,dtNsfunc = self.get_Ns(t,self.Ns01)

        Nmfunc2,dtNmfunc2 = self.get_Nm2(t,self.Nm02)
        Nsfunc2,dtNsfunc2 = self.get_Ns2(t,self.Ns02)

        ps = (dm*Nmfunc + dtNmfunc)/(Nsfunc - k*Nsfunc)
        ds = (dtNsfunc - k*(dm*Nmfunc + dtNmfunc + dtNsfunc))/((-1 + k)*Nsfunc)

        ps2 = (dm2*Nmfunc2 + dtNmfunc2)/(Nsfunc2 - k2*Nsfunc2)
        ds2 = (dtNsfunc2 - k2*(dm2*Nmfunc2 + dtNmfunc2 + dtNsfunc2))/((-1 + k2)*Nsfunc2)

        iparas = {'ps1' : ps,'ds1':ds,'Ns1':Nsfunc,'Nm1':Nmfunc,'ps2' : ps2,'ds2':ds2,'Ns2':Nsfunc2,'Nm2':Nmfunc2}
        if Nmfunc <= 0:
            raise ImplicitParametersOutOfRange('Nm is negative',iparas)
        if Nsfunc <= 0:
            raise ImplicitParametersOutOfRange('Ns is negative',iparas)
        if ps < 0:
            raise ImplicitParametersOutOfRange('ps is negative',iparas)
        if ds < 0:
            raise ImplicitParametersOutOfRange('ds is negative',iparas)
        if ps > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        if ds > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ds"+"=" + str(ds) + " is outside the limits " + str(global_limit),iparas)
        if Nmfunc2 <= 0:
            raise ImplicitParametersOutOfRange('Nm2 is negative',iparas)
        if Nsfunc2 <= 0:
            raise ImplicitParametersOutOfRange('Ns2 is negative',iparas)
        if ps2 < 0:
            raise ImplicitParametersOutOfRange('ps2 is negative',iparas)
        if ds2 < 0:
            raise ImplicitParametersOutOfRange('ds2 is negative',iparas)
        if ps2 > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps2"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        if ds2 > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ds2"+"=" + str(ds) + " is outside the limits " + str(global_limit),iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y,(self.nvars,-1))
        M_new = np.zeros_like(M)
        stem = self.stem1
        muscle = self.muscle1

        stem2 = self.stem2
        muscle2 = self.muscle2



        k = self.k1
        k2 = self.k2
        iparas = self.calc_implicit_parameters(t)
        ps = iparas['ps1']
        ds = iparas['ds1']
        Nm = iparas['Nm1']
        Ns = iparas['Ns1']
        ps2 = iparas['ps2']
        ds2 = iparas['ds2']
        Nm2 = iparas['Nm2']
        Ns2 = iparas['Ns2']


        M_new[stem] = ((1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - M[stem]))/2
        M_new[muscle] = -((-1 + k)*ps*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle] + M[stem])*Ns)/(2.*Nm)

        M_new[stem2] = ((1 + k2)*ps2*(self.Catm.lin(t + self.Dbirth) - M[stem2]))/2
        M_new[muscle2] = -((-1 + k2)*ps2*(self.Catm.lin(t + self.Dbirth) - 2*M[muscle2] + M[stem2])*Ns2)/(2.*Nm2)

        return M_new.ravel()

'''
------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------
FIXED
------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------
'''
class simple_fixed(simple):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(dm=10**-3)
        self.__dict__.update(default_fixed)
        self.logparas = ['dm']
        self.linparas = []

        limit  = {i:global_limit for i in default_parameters.keys()}

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



class simpleM2_fixed(simpleM2):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(dm1=10**-2,dm2=10**-3)
        self.__dict__.update(default_fixed)
        self.logparas = ['dm2','dm1']
        self.linparas = []

        limit  = {i:global_limit for i in default_parameters.keys()}

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)



class dm1_0simpleM2_fixed(dm1_0simpleM2):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(dm2=10**-3)
        self.__dict__.update(default_fixed)
        self.logparas = ['dm2']
        self.linparas = []

        limit  = {i:global_limit for i in default_parameters.keys()}

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)

class dm2_0simpleM2_fixed(simpleM2):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(dm1=10**-2)
        self.__dict__.update(default_fixed)
        self.logparas = ['dm1']
        self.linparas = []
        self.dm2=0
        limit  = {i:global_limit for i in default_parameters.keys()}

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)

class dm1_dm2simpleM2_fixed(simpleM2):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(dm=10**-2)
        self.__dict__.update(default_fixed)
        self.logparas = ['dm']
        self.linparas = []

        limit  = {i:global_limit for i in default_parameters.keys()}

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)


    def calc_implicit_parameters(self, t):
        dm1 = self.dm
        dm2 = self.dm

        Nmfunc1,dtNmfunc1 = self.get_Nm(t,self.Nm01)
        Nmfunc2,dtNmfunc2 = self.get_Nm2(t,self.Nm02)

        fm1 = dm1*Nmfunc1 + dtNmfunc1
        fm2 = dm2*Nmfunc2 + dtNmfunc2
        iparas = {'fm1' :fm1,'fm2':fm2,'Nm1':Nmfunc1,'Nm2':Nmfunc2}
        if Nmfunc1 <= 0:
            raise ImplicitParametersOutOfRange('Nm1 is negative',iparas)
        if Nmfunc2 <= 0:
            raise ImplicitParametersOutOfRange('Nm2 is negative',iparas)
        return iparas




class SM_fixed(SM):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(k=0.5, dm=10**-2)
        self.logparas = ['dm']
        self.linparas = ['k']
        self.__dict__.update(default_fixed)
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k'] = (0, KMAX)
        
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)

class dm0SM_fixed(dm0SM):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(k=0.5)
        self.logparas = []
        self.linparas = ['k']
        self.__dict__.update(default_fixed)
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k'] = (0, KMAX)
        
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)

class k0SM_fixed(k0SM):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict( dm=10**-2)
        self.logparas = ['dm']
        self.linparas = []
        self.__dict__.update(default_fixed)
        self.N0 = 1
        self.k = 0
        limit  = {i:global_limit for i in default_parameters.keys()}
        
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)

class SM2_fixed(SM2):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(ds=0.5, dm=10**-2)
        self.logparas = ['dm','ds']
        self.linparas = []
        self.__dict__.update(default_fixed)
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        
        
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)

class dm0SM2_fixed(dm0SM2):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(ds=0.1)
        self.logparas = ['ds']
        self.linparas = []
        self.__dict__.update(default_fixed)
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}        
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)

class S2M_fixed(S2M):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(k=0.5,dm1=10**-2,dm2=10**-2)
        self.logparas = ['dm1','dm2']
        self.linparas = ['k']
        self.__dict__.update(default_fixed)
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k'] = (0, KMAX)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)


class k0S2M_fixed(k0S2M):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(dm1=10**-2,dm2=10**-2)
        self.logparas = ['dm1','dm2']
        self.linparas = []
        self.__dict__.update(default_fixed)
        self.N0 = 1
        self.k = 0
        limit  = {i:global_limit for i in default_parameters.keys()}

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)


class dm1_0S2M_fixed(dm1_0S2M):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(k=0.5,dm2=10**-2)
        self.logparas = ['dm2']
        self.linparas = ['k']
        self.__dict__.update(default_fixed)
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k'] = (0, KMAX)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)

class dm1_0_dm2_0S2M_fixed(dm1_0_dm2_0S2M):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(k=0.5)
        self.logparas = []
        self.linparas = ['k']
        self.__dict__.update(default_fixed)
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k'] = (0, KMAX)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)


class S2_M2_fixed(S2_M2):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(k1=0.5,k2=0.5,dm1=10**-2,dm2=10**-2)
        self.logparas = ['dm1','dm2']
        self.linparas = ['k1','k2']
        self.__dict__.update(default_fixed)
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k1'] = (0, KMAX)
        limit['k2'] = (0, KMAX)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem1','stem2','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)

class dm1_0S2_M2_fixed(S2_M2):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(k1=0.5,k2=0.5,dm2=10**-2)
        self.logparas = ['dm2']
        self.linparas = ['k1','k2']
        self.__dict__.update(default_fixed)
        self.N0 = 1
        self.dm1=0
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k1'] = (0, KMAX)
        limit['k2'] = (0, KMAX)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem1','stem2','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)

class dm2_0S2_M2_fixed(S2_M2):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(k1=0.5,k2=0.5,dm1=10**-2)
        self.logparas = ['dm1']
        self.linparas = ['k1','k2']
        self.__dict__.update(default_fixed)
        self.N0 = 1
        self.dm2=0
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k1'] = (0, KMAX)
        limit['k2'] = (0, KMAX)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem1','stem2','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)


class dm1_dm2S2_M2_fixed(S2_M2):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(k1=0.5,k2=0.5,dm=10**-2)
        self.logparas = ['dm']
        self.linparas = ['k1','k2']
        self.__dict__.update(default_fixed)
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['k1'] = (0, KMAX)
        limit['k2'] = (0, KMAX)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem1','stem2','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)

    def calc_implicit_parameters(self, t):
        k = self.k1
        dm = self.dm

        k2 = self.k2
        dm2 = self.dm

        Nmfunc,dtNmfunc = self.get_Nm(t,self.Nm01)
        Nsfunc,dtNsfunc = self.get_Ns(t,self.Ns01)

        Nmfunc2,dtNmfunc2 = self.get_Nm2(t,self.Nm02)
        Nsfunc2,dtNsfunc2 = self.get_Ns2(t,self.Ns02)

        ps = (dm*Nmfunc + dtNmfunc)/(Nsfunc - k*Nsfunc)
        ds = (dtNsfunc - k*(dm*Nmfunc + dtNmfunc + dtNsfunc))/((-1 + k)*Nsfunc)

        ps2 = (dm2*Nmfunc2 + dtNmfunc2)/(Nsfunc2 - k2*Nsfunc2)
        ds2 = (dtNsfunc2 - k2*(dm2*Nmfunc2 + dtNmfunc2 + dtNsfunc2))/((-1 + k2)*Nsfunc2)

        iparas = {'ps1' : ps,'ds1':ds,'Ns1':Nsfunc,'Nm1':Nmfunc,'ps2' : ps2,'ds2':ds2,'Ns2':Nsfunc2,'Nm2':Nmfunc2}
        if Nmfunc <= 0:
            raise ImplicitParametersOutOfRange('Nm is negative',iparas)
        if Nsfunc <= 0:
            raise ImplicitParametersOutOfRange('Ns is negative',iparas)
        if ps < 0:
            raise ImplicitParametersOutOfRange('ps is negative',iparas)
        if ds < 0:
            raise ImplicitParametersOutOfRange('ds is negative',iparas)
        if ps > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        if ds > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ds"+"=" + str(ds) + " is outside the limits " + str(global_limit),iparas)
        if Nmfunc2 <= 0:
            raise ImplicitParametersOutOfRange('Nm2 is negative',iparas)
        if Nsfunc2 <= 0:
            raise ImplicitParametersOutOfRange('Ns2 is negative',iparas)
        if ps2 < 0:
            raise ImplicitParametersOutOfRange('ps2 is negative',iparas)
        if ds2 < 0:
            raise ImplicitParametersOutOfRange('ds2 is negative',iparas)
        if ps2 > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ps2"+"=" + str(ps) + " is outside the limits " + str(global_limit),iparas)
        if ds2 > global_limit[1]:
            raise ImplicitParametersOutOfRange("Parameter "+"ds2"+"=" + str(ds) + " is outside the limits " + str(global_limit),iparas)
        return iparas





'''
class SchrinkType2Skonst(MBt2):
    def __init__(self):
        default_parameters=dict(k=0,dm1=-2,dm2=-2, mgrow=0, grow_time=0, resttime=0,Nm2srate=-3,Nssrate=-3)
        self.logparas = ['mgrow','dm1','dm2','Nm2srate','Nssrate']
        self.linparas = ['grow_time','k','resttime']
        self.N0 = 1
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit['grow_time'] = (1, 99)
        limit['k'] = (0, KMAX)
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
        limit['k'] = (0, KMAX)

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
        limit['k'] = (0, KMAX)

        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['stem','muscle1','muscle2'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
models_list_fixed.append(dm1_0SchrinkType2Skonst_fixed)

'''




'''
---------------------------------------------------------------------------------------------------------------------------------

combines

---------------------------------------------------------------------------------------------------------------------------------
'''
class simple_grow(simple,Nm_grow):
    def __init__(self):
        simple.__init__(self)
models_list.append(simple_grow)

class simple_grow_fixed(simple_fixed,Nm_grow):
    def __init__(self):
        simple_fixed.__init__(self)
models_list_fixed.append(simple_grow_fixed)

class simple_growshrink(simple,Nm_grow):
    def __init__(self):
        simple.__init__(self)
models_list.append(simple_growshrink)

class simple_growshrink_fixed(simple_fixed,Nm_growshrink):
    def __init__(self):
        simple_fixed.__init__(self)
models_list_fixed.append(simple_growshrink_fixed)






class simpleM2_grow(simpleM2,Nm_grow,Nm2_grow):
    def __init__(self):
        simpleM2.__init__(self)
models_list.append(simpleM2_grow)

class simpleM2_grow_fixed(simpleM2_fixed,Nm_grow,Nm2_grow):
    def __init__(self):
        simpleM2_fixed.__init__(self)
models_list_fixed.append(simpleM2_grow_fixed)

class simpleM2_grow_shrink(simpleM2,Nm_grow,Nm2_growshrink):
    def __init__(self):
        simpleM2.__init__(self)
models_list.append(simpleM2_grow_shrink)

class simpleM2_grow_shrink_fixed(simpleM2_fixed,Nm_grow,Nm2_growshrink):
    def __init__(self):
        simpleM2_fixed.__init__(self)
models_list_fixed.append(simpleM2_grow_shrink_fixed)


class dm1_0simpleM2_grow(dm1_0simpleM2,Nm_grow,Nm2_grow):
    def __init__(self):
        dm1_0simpleM2.__init__(self)
models_list.append(dm1_0simpleM2_grow)

class dm1_0simpleM2_grow_fixed(dm1_0simpleM2_fixed,Nm_grow,Nm2_grow):
    def __init__(self):
        dm1_0simpleM2_fixed.__init__(self)
models_list_fixed.append(dm1_0simpleM2_grow_fixed)





class dm1_0simpleM2_grow_shrink(dm1_0simpleM2,Nm_grow,Nm2_growshrink):
    def __init__(self):
        dm1_0simpleM2.__init__(self)
models_list.append(dm1_0simpleM2_grow_shrink)

class dm1_0simpleM2_grow_shrink_fixed(dm1_0simpleM2_fixed,Nm_grow,Nm2_growshrink):
    def __init__(self):
        dm1_0simpleM2_fixed.__init__(self)
models_list_fixed.append(dm1_0simpleM2_grow_shrink_fixed)


'''
------------------------------------------------------------------------------------------------------------------
'''

class Skonst_Mgrow(SM,Nm_grow,Ns_const):
    def __init__(self):
        SM.__init__(self)
models_list.append(Skonst_Mgrow)
class Skonst_Mgrow_fixed(SM_fixed,Nm_grow,Ns_const):
    def __init__(self):
        SM_fixed.__init__(self)
models_list_fixed.append(Skonst_Mgrow_fixed)
#dm0
class dm0Skonst_Mgrow(dm0SM,Nm_grow,Ns_const):
    def __init__(self):
        dm0SM.__init__(self)
models_list.append(dm0Skonst_Mgrow)
class dm0Skonst_Mgrow_fixed(dm0SM_fixed,Nm_grow,Ns_const):
    def __init__(self):
        dm0SM_fixed.__init__(self)
models_list_fixed.append(dm0Skonst_Mgrow_fixed)
#k0
class k0Skonst_Mgrow(k0SM,Nm_grow,Ns_const):
    def __init__(self):
        k0SM.__init__(self)
models_list.append(k0Skonst_Mgrow)
class k0Skonst_Mgrow_fixed(k0SM_fixed,Nm_grow,Ns_const):
    def __init__(self):
        k0SM_fixed.__init__(self)
models_list_fixed.append(k0Skonst_Mgrow_fixed)


#alt
class Skonst_Mgrow_2(SM2,Nm_grow,Ns_const):
    def __init__(self):
        SM2.__init__(self)
models_list.append(Skonst_Mgrow_2)
class Skonst_Mgrow_2_fixed(SM2_fixed,Nm_grow,Ns_const):
    def __init__(self):
        SM2_fixed.__init__(self)
models_list_fixed.append(Skonst_Mgrow_2_fixed)
#dm0
class dm0Skonst_Mgrow_2(dm0SM2,Nm_grow,Ns_const):
    def __init__(self):
        dm0SM2.__init__(self)
models_list.append(dm0Skonst_Mgrow_2)
class dm0Skonst_Mgrow_2_fixed(dm0SM2_fixed,Nm_grow,Ns_const):
    def __init__(self):
        dm0SM2_fixed.__init__(self)
models_list_fixed.append(dm0Skonst_Mgrow_2_fixed)




#shrink
class Sshrink_Mgrowshrink(SM,Nm_growshrink,Ns_shrink):
    def __init__(self):
        SM.__init__(self)
models_list.append(Sshrink_Mgrowshrink)
class Sshrink_Mgrowshrink_fixed(SM_fixed,Nm_growshrink,Ns_shrink):
    def __init__(self):
        SM_fixed.__init__(self)
models_list_fixed.append(Sshrink_Mgrowshrink_fixed)
##k0
class k0Sshrink_Mgrowshrink(k0SM,Nm_growshrink,Ns_shrink):
    def __init__(self):
        k0SM.__init__(self)
models_list.append(k0Sshrink_Mgrowshrink)
class k0Sshrink_Mgrowshrink_fixed(k0SM_fixed,Nm_growshrink,Ns_shrink):
    def __init__(self):
        k0SM_fixed.__init__(self)
models_list_fixed.append(k0Sshrink_Mgrowshrink_fixed)



'''
------------------------------------------------------------------------------------------------------------------
'''
class Skonst_M1grow_M2grow(S2M,Nm_grow,Nm2_grow,Ns_const):
    def __init__(self):
        S2M.__init__(self)
models_list.append(Skonst_M1grow_M2grow)
class Skonst_M1grow_M2grow_fixed(S2M_fixed,Nm_grow,Nm2_grow,Ns_const):
    def __init__(self):
        S2M_fixed.__init__(self)
models_list_fixed.append(Skonst_M1grow_M2grow_fixed)
#dm1_0
class dm1_0Skonst_M1grow_M2grow(dm1_0S2M,Nm_grow,Nm2_grow,Ns_const):
    def __init__(self):
        dm1_0S2M.__init__(self)
models_list.append(dm1_0Skonst_M1grow_M2grow)
class dm1_0Skonst_M1grow_M2grow_fixed(dm1_0S2M_fixed,Nm_grow,Nm2_grow,Ns_const):
    def __init__(self):
        dm1_0S2M_fixed.__init__(self)
models_list_fixed.append(dm1_0Skonst_M1grow_M2grow_fixed)
#dm1_0 dm2_0
class dm1_0_dm2_0Skonst_M1grow_M2grow(dm1_0_dm2_0S2M,Nm_grow,Nm2_grow,Ns_const):
    def __init__(self):
        dm1_0_dm2_0S2M.__init__(self)
models_list.append(dm1_0_dm2_0Skonst_M1grow_M2grow)
class dm1_0_dm2_0Skonst_M1grow_M2grow_fixed(dm1_0_dm2_0S2M_fixed,Nm_grow,Nm2_grow,Ns_const):
    def __init__(self):
        dm1_0_dm2_0S2M_fixed.__init__(self)
models_list_fixed.append(dm1_0_dm2_0Skonst_M1grow_M2grow_fixed)


class Sshrink_M1grow_M2growshrink(S2M,Nm_grow,Nm2_growshrink,Ns_shrink):
    def __init__(self):
        S2M.__init__(self)
models_list.append(Sshrink_M1grow_M2growshrink)
class Sshrink_M1grow_M2growshrink_fixed(S2M_fixed,Nm_grow,Nm2_growshrink,Ns_shrink):
    def __init__(self):
        S2M_fixed.__init__(self)
models_list_fixed.append(Sshrink_M1grow_M2growshrink_fixed)
#dm1_0
class dm1_0Sshrink_M1grow_M2growshrink(dm1_0S2M,Nm_grow,Nm2_growshrink,Ns_shrink):
    def __init__(self):
        dm1_0S2M.__init__(self)
models_list.append(dm1_0Sshrink_M1grow_M2growshrink)
class dm1_0Sshrink_M1grow_M2growshrink_fixed(dm1_0S2M_fixed,Nm_grow,Nm2_growshrink,Ns_shrink):
    def __init__(self):
        dm1_0S2M_fixed.__init__(self)
models_list_fixed.append(dm1_0Sshrink_M1grow_M2growshrink_fixed)


#k0
class k0Skonst_M1grow_M2grow(k0S2M,Nm_grow,Nm2_grow,Ns_const):
    def __init__(self):
        k0S2M.__init__(self)
models_list.append(k0Skonst_M1grow_M2grow)
class k0Skonst_M1grow_M2grow_fixed(k0S2M_fixed,Nm_grow,Nm2_grow,Ns_const):
    def __init__(self):
        k0S2M_fixed.__init__(self)
models_list_fixed.append(k0Skonst_M1grow_M2grow_fixed)



class k0Skonst_M1grow_M2growshrink(k0S2M,Nm_grow,Nm2_growshrink,Ns_shrink):
    def __init__(self):
        k0S2M.__init__(self)
models_list.append(k0Skonst_M1grow_M2growshrink)
class k0Skonst_M1grow_M2growshrink_fixed(k0S2M_fixed,Nm_grow,Nm2_growshrink,Ns_shrink):
    def __init__(self):
        k0S2M_fixed.__init__(self)
models_list_fixed.append(k0Skonst_M1grow_M2growshrink_fixed)




class simple_growshrinkS_fixed(simple_fixed,Nm_growshrinkS):
    def __init__(self):
        simple_fixed.__init__(self)
models_list_fixed.append(simple_growshrinkS_fixed)


class simpleM2_growS_shrinkS_fixed(simpleM2_fixed,Nm_growS,Nm2_growshrinkS):
    def __init__(self):
        simpleM2_fixed.__init__(self)
models_list_fixed.append(simpleM2_growS_shrinkS_fixed)

class k0Sshrink_MgrowshrinkS_fixed(k0SM_fixed,Nm_growshrinkS,Ns_shrink):
    def __init__(self):
        k0SM_fixed.__init__(self)
models_list_fixed.append(k0Sshrink_MgrowshrinkS_fixed)

class k0Sshrink_M1growS_M2growshrinkS_fixed(k0S2M_fixed,Nm_growS,Nm2_growshrinkS,Ns_shrink):
    def __init__(self):
        k0S2M_fixed.__init__(self)
models_list_fixed.append(k0Sshrink_M1growS_M2growshrinkS_fixed)


###grow log

class simple_growlog_fixed(simple_fixed,Nm_growlog):
    def __init__(self):
        simple_fixed.__init__(self)
models_list_fixed.append(simple_growlog_fixed)

class simpleM2_growlog_fixed(simpleM2_fixed,Nm_growlog,Nm2_growlog):
    def __init__(self):
        simpleM2_fixed.__init__(self)
models_list_fixed.append(simpleM2_growlog_fixed)

class dm1_0simpleM2_growlog_fixed(dm1_0simpleM2_fixed,Nm_growlog,Nm2_growlog):
    def __init__(self):
        dm1_0simpleM2_fixed.__init__(self)
models_list_fixed.append(dm1_0simpleM2_growlog_fixed)

class dm2_0simpleM2_growlog_fixed(dm2_0simpleM2_fixed,Nm_growlog,Nm2_growlog):
    def __init__(self):
        dm2_0simpleM2_fixed.__init__(self)
models_list_fixed.append(dm2_0simpleM2_growlog_fixed)

class dm1_dm2simpleM2_growlog_fixed(dm1_dm2simpleM2_fixed,Nm_growlog,Nm2_growlog):
    def __init__(self):
        dm1_dm2simpleM2_fixed.__init__(self)
models_list_fixed.append(dm1_dm2simpleM2_growlog_fixed)


class Skonst_growlog_fixed(SM_fixed,Nm_growlog,Ns_const):
    def __init__(self):
        SM_fixed.__init__(self)
models_list_fixed.append(Skonst_growlog_fixed)


class k0Skonst_growlog_fixed(k0SM_fixed,Nm_growlog,Ns_const):
    def __init__(self):
        k0SM_fixed.__init__(self)
models_list_fixed.append(k0Skonst_growlog_fixed)

class Skonst_M1growlog_M2growlog_fixed(S2M_fixed,Nm_growlog,Nm2_growlog,Ns_const):
    def __init__(self):
        S2M_fixed.__init__(self)
models_list_fixed.append(Skonst_M1growlog_M2growlog_fixed)
class k0Skonst_M1growlog_M2growlog_fixed(k0S2M_fixed,Nm_growlog,Nm2_growlog,Ns_const):
    def __init__(self):
        k0S2M_fixed.__init__(self)
models_list_fixed.append(k0Skonst_M1growlog_M2growlog_fixed)

class Skonst_M1growlog_M2growlog_fixed(S2M_fixed,Nm_growlog,Nm2_growlog,Ns_const):
    def __init__(self):
        S2M_fixed.__init__(self)
models_list_fixed.append(Skonst_M1growlog_M2growlog_fixed)

class simple_semifixed(simple):
    def __init__(self,default_fixed=DEFAULT_FIXED):
        default_parameters=dict(T0=40,A=10,B=1)
        self.__dict__.update(default_fixed)
        self.logparas = []
        self.linparas = ['T0','A','B']
        limit  = {i:global_limit for i in default_parameters.keys()}
        limit.update(dict(T0=(20,60),A=(3,40),B=(0.01,40)))
        self.dm = 1e-10
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['muscle'],\
            default_parameters=default_parameters,\
            error={i:global_error for i in default_parameters.keys()},\
            limit=limit    )
        MB.__init__(self)
class simple_grow_old_fixed(simple_semifixed,Nm_grow_old):
    def __init__(self):
        simple_semifixed.__init__(self)
models_list_fixed.append(simple_grow_old_fixed)



class Skonst2_Mgrowlog2_fixed(S2_M2_fixed,Nm_growlog,Nm2_growlog,Ns_const,Ns2_const):
    def __init__(self):
        S2_M2_fixed.__init__(self)
models_list_fixed.append(Skonst2_Mgrowlog2_fixed)


class dm1_0Skonst2_Mgrowlog2_fixed(dm1_0S2_M2_fixed,Nm_growlog,Nm2_growlog,Ns_const,Ns2_const):
    def __init__(self):
        dm1_0S2_M2_fixed.__init__(self)
models_list_fixed.append(dm1_0Skonst2_Mgrowlog2_fixed)

class dm2_0Skonst2_Mgrowlog2_fixed(dm2_0S2_M2_fixed,Nm_growlog,Nm2_growlog,Ns_const,Ns2_const):
    def __init__(self):
        dm2_0S2_M2_fixed.__init__(self)
models_list_fixed.append(dm2_0Skonst2_Mgrowlog2_fixed)

class dm1_dm2Skonst2_Mgrowlog2_fixed(dm1_dm2S2_M2_fixed,Nm_growlog,Nm2_growlog,Ns_const,Ns2_const):
    def __init__(self):
        dm1_dm2S2_M2_fixed.__init__(self)
models_list_fixed.append(dm1_dm2Skonst2_Mgrowlog2_fixed)
