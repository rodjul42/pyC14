import numpy as np
import pandas as pd
from .base import model_base, Catm
from scipy.interpolate import UnivariateSpline,splev,interp1d
from scipy.integrate import cumulative_trapezoid
from ..tools import ImplicitParametersOutOfRange,listofdict_to_dictoflist
from .minimal import POP1, POP1q
import collections
import pkg_resources

models_list = []

global_error = 0.5
global_limit = (10**-6, 10**1)
kappa_limit = (1e-3, 10)

default_path = \
    (pkg_resources.
     resource_filename(__name__,
                       "../data/kudryavtsev_et_al_1993_table_2.xlsx")
     )

def dnatotal_linlin(t):
    a=[ 2.14903551e+01,  2.14334756e-01,  1.52591351e+00, -4.75226459e-03]
    A = a[1] + (a[2]-a[1])*t/a[0]
    B = a[2] + a[3]*(t-a[0])
    if isinstance(t, (collections.Sequence, np.ndarray)):
        return np.where(t<a[0],A,B),np.where(t<a[0],np.ones_like(t)*(a[2]-a[1])/a[0],np.ones_like(t)*a[3])
    else:
        if t<a[0]:
            return A,(a[2]-a[1])/a[0]
        else:
            return B,a[3]

def dnatotal_linconst(t):
    a=[ 2.14903551e+01,  2.14334756e-01,  1.52591351e+00, -4.75226459e-03]
    A = a[1] + (a[2]-a[1])*t/a[0]
    B = a[2]
    if isinstance(t, (collections.Sequence, np.ndarray)):
        return np.where(t<a[0],A,B),np.where(t<a[0],np.ones_like(t)*(a[2]-a[1])/a[0],np.zeros_like(t))
    else:
        if t<a[0]:
            return A,(a[2]-a[1])/a[0]
        else:
            return B,0

def dnatotal_linconst_B(t):
    a=[ 55,  2.14334756e-01,  2, -4.75226459e-03]
    A = a[1] + (a[2]-a[1])*t/a[0]
    B = a[2]
    if isinstance(t, (collections.Sequence, np.ndarray)):
        return np.where(t<a[0],A,B),np.where(t<a[0],np.ones_like(t)*(a[2]-a[1])/a[0],np.zeros_like(t))
    else:
        if t<a[0]:
            return A,(a[2]-a[1])/a[0]
        else:
            return B,0

def dnatotal_const(t):
    return 1,0

def dnatotal_spline(t):
    tck = (np.array([ 0.,  0.,  0., 30., 60., 95., 95., 95.]),
 np.array([0.16852311, 1.42300436, 1.79052905, 1.44270567, 1.20370771,
        0.        , 0.        , 0.        ]),
 2)
    if isinstance(t, (collections.Sequence, np.ndarray)):
        return splev(t, tck),splev(t, tck,1)
    else:
        return float(splev(t, tck)),float(splev(t, tck,1))


class POP1(model_base):
    populations_DNA = {'cells1':1 }
    plot_types = ['mean']
    populations = ['cells1']
    populations_plot = {'mean':populations}
    iparas = ['b2n']
    flow_in =  {'cells1':[('b2n','cells1',2)]}
    
    def __init__(self, d=0.1,delay=1,dnatotal=dnatotal_const):
        default_parameters = dict(d=d)
        limit = {i: global_limit for i in default_parameters.keys()}
        self.logparas = ['d']
        self.linparas = []
        self.dnatotal  = dnatotal

        self.Catm = Catm(delay=delay)
        model_base.__init__(self, var_names=['cells1'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)

    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)
        ipara = self.calc_implicit_parameters(t)

        b2n = ipara['b2n']
        

        M_new[self.cells1] = b2n*(self.Catm.lin(t + self.Dbirth)
                                      - M[self.cells1])
        
        return M_new.ravel()

    def calc_implicit_parameters(self, t):
        cells1,dtcells1 = self.dnatotal(t)
        b2n = self.d + dtcells1/cells1
        iparas = {'b2n':b2n,'cells1':cells1} 
        if b2n < 0:
            raise ImplicitParametersOutOfRange(
                f'b2n ({b2n}) < 0 at t={t})', iparas)
        return iparas

    def measurement_model(self, result_sim, data):
        return result_sim['cells1'] 

class POP1damage(model_base):
    populations_DNA = {'cells1':1 }
    plot_types = ['mean']
    populations = ['cells1']
    populations_plot = {'mean':populations}
    iparas = ['b2n']
    flow_in =  {'cells1':[('b2n','cells1',2)]}
    
    def __init__(self, damage=0.000313, d=0.1,delay=1,dnatotal=dnatotal_const):
        default_parameters = dict(d=d)
        limit = {i: global_limit for i in default_parameters.keys()}
        self.logparas = ['d']
        self.linparas = []
        self.dnatotal  = dnatotal
        self.damage = damage  #= dna - A*dna  dna damage rate; A amount of damage or
        self.Catm = Catm(delay=delay)
        model_base.__init__(self, var_names=['cells1'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)

    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)
        ipara = self.calc_implicit_parameters(t)

        b2n = ipara['b2n']
        

        #M_new[c2n] =(b2n + dna - A*dna)*self.Catm.lin(t + self.Dbirth) - (M[c2n]*((d + dna - A*dna)*N2n + dtN2n))/N2n
        M_new[self.cells1] =(b2n + self.damage)*self.Catm.lin(t + self.Dbirth) - (M[self.cells1]*(b2n + self.damage))   
        return M_new.ravel()

    def calc_implicit_parameters(self, t):
        cells1,dtcells1 = self.dnatotal(t)
        b2n = self.d + dtcells1/cells1
        iparas = {'b2n':b2n,'cells1':cells1} 
        if b2n < 0:
            raise ImplicitParametersOutOfRange(
                f'b2n ({b2n}) < 0 at t={t})', iparas)
        return iparas

    def measurement_model(self, result_sim, data):
        return result_sim['cells1'] 




class Hepatocyte(model_base):
    def __init__(self, path=default_path):
        self.Catm = Catm(delay=1)
        ploidy_data = pd.read_excel(path)
        ploidy_data['age'] = (ploidy_data[['age_min', 'age_max']].
                              mean(axis='columns'))
        ploidy_data /= 100.0
        ploidy_data['age'] *= 100
        '''
        self.ploidy = UnivariateSpline(ploidy_data['age'].values,
                                       ploidy_data['2C_mean'].values,
                                       ext=0, k=2)
        self.dtploidy = self.ploidy.derivative()
        self.ploidy2x2 = UnivariateSpline(ploidy_data['age'].values,
                                          ploidy_data['2Cx2_mean'].values,
                                          ext=0, k=2)
        self.dtploidy2x2 = self.ploidy2x2.derivative()
        '''
        

    def ploidy(self,t):
        return  -0.00515927*t + 0.92081785
        #return  -0.00468915*t + 0.91848584
        #return -0.00447121*t + 0.91854464
        
    def dtploidy(self,t):
        return  -0.00515927
        #return  -0.00468915
        #return -0.00447121

    def ploidy2x2(self,t):
        return  0.00168975*t + 0.0735638
        #return  0.00156924*t + 0.07462464
        #return 0.00159886*t + 0.07459858
        
    def dtploidy2x2(self,t):
        return  0.00168975
        #return  0.00156924
        #return 0.00159886


    def calc_w(self,data):
        mask_n2n4 = data.df.ploidy == '2n4n'
        mask_n2 = data.df.ploidy == '2n'
        mask_n4 = data.df.ploidy == '4n'

        mask_Cn2 = data.df.ploidy == 'C2n'
        mask_Cn4 = data.df.ploidy == 'C4n'



        wn2 = np.empty_like(data.age) * np.nan
        wn4 = np.empty_like(data.age) * np.nan
        wq = np.empty_like(data.age) * np.nan

        wn2[mask_n2n4] = (self.ploidy(data.age[mask_n2n4])
                              / (self.ploidy(data.age[mask_n2n4])
                                 + 2*(1-self.ploidy(data.age[mask_n2n4]))))
        wn2[mask_n2] = (self.ploidy(data.age[mask_n2])
                            / (self.ploidy(data.age[mask_n2])
                               + 2*self.ploidy2x2(data.age[mask_n2])))
        wn2[mask_n4] = 0

        wn2[mask_Cn2] = 1
        wn2[mask_Cn4] = 0

        wn4[mask_n2n4] = (2*(1-self.ploidy(data.age[mask_n2n4]))
                              / (self.ploidy(data.age[mask_n2n4])
                                 + 2*(1-self.ploidy(data.age[mask_n2n4]))))
        wn4[mask_n2] = (2*self.ploidy2x2(data.age[mask_n2])
                            / (self.ploidy(data.age[mask_n2])
                               + 2*self.ploidy2x2(data.age[mask_n2])))
        wn4[mask_n4] =  1
        
        wn4[mask_Cn2] = 0
        wn4[mask_Cn4] = 1

   
        return wn2,wn4

    def measurement_model(self, result_sim, data):
        try:
            f = self.f
        except AttributeError:
            f = 1.0

        wn2,wn4 = self.calc_w(data)
        average = (result_sim['c2n']*wn2*f
                   + result_sim['c4n']*wn4*f
                   + result_sim['q']*(1-f)
                   )

        return average





class Rm(Hepatocyte):
    """ Liver model,
        r2 and r4 are implicitely time dependent
    """
    populations_DNA = {'N2n':1, 'N4n':2}
    plot_types = ['mean','2n','4n']
    populations = ['N2n','N4n']
    populations_plot = {'mean':populations,'2n':['N2n'],'4n':['N4n']}
    iparas = ['b2n', 'b4n']
    flow_in =  {'N2n':[('b2n','N2n',2),('k42','N4n',4)],'N4n':[('b4n','N4n',2),('k24','N2n',1)]}
    
    def __init__(self, k24=10**-3.0, k42=10**-2.0, d2n=10**-1.0, d4n=10**-1.0,dnatotal=dnatotal_const):
        default_parameters = dict(k24=k24, k42=k42,
                                  d2n=d2n, d4n=d4n)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        limit['k42'] = kappa_limit
        limit['k24'] = kappa_limit

        self.dnatotal  = dnatotal
        Hepatocyte.__init__(self)
        model_base.__init__(self, var_names=['c2n', 'c4n', 'q'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_turnover_and_flow(self,t):
        ip = self.calc_implicit_parameters(t)
        t2n = self.d2 + self.k24 + ip['b2n']
        t4n = self.d4 + self.k42 + ip['b4n']
        N_total = 1/(2-self.ploidy(t))
        N2 = self.ploidy(t)    *N_total
        N4 = (1-self.ploidy(t))*N_total
        f2n = self.k42*N4*4 + ip['b2n']*N2*2
        f4n = self.k24*N2 +   ip['b4n']*N4*2
        return {'t2n':t2n,'t4n':t4n,'f2n':f2n,'f4n':f4n}


    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        k24 = self.k24
        k42 = self.k42
        d2n = self.d2n
        d4n = self.d4n

        ploidy = self.ploidy(t)
        dtploidy = self.dtploidy(t)
        DNATotal,dtDNATotal = self.dnatotal(t)
        



        # populations 

        N2n = -((DNATotal*ploidy)/(-2 + ploidy))
        N4n = DNATotal*(1 + 1/(-2 + ploidy))

        # dt populations 

        dtN2n = (-((-2 + ploidy)*ploidy*dtDNATotal) + 2*DNATotal*dtploidy)/(-2 + ploidy)**2
        dtN4n = ((-2 + ploidy)*(-1 + ploidy)*dtDNATotal - DNATotal*dtploidy)/(-2 + ploidy)**2

        #implicit paras 

        b2n = ((d2n + k24)*N2n - 4*k42*N4n + dtN2n)/N2n
        b4n = (-(k24*N2n) + (d4n + k42)*N4n + dtN4n)/N4n


        iparas = {'b2n': b2n, 'b4n': b4n,'N2n':N2n,'N4n':N4n}

        if b2n < 0:
            raise ImplicitParametersOutOfRange(
                f'b2n ({b2n})is negative at t = {t}', iparas)
        if b4n < 0:
            raise ImplicitParametersOutOfRange(
                f'b4n ({b4n}) is negative at t = {t}', iparas)
        if k42 > b4n:
            raise ImplicitParametersOutOfRange(
                f'kappa42 ({k42}) > b4n ({b4n} at t={t})', iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        iparas = self.calc_implicit_parameters(t)
        
        b2n = iparas['b2n']
        b4n = iparas['b4n']

        N2n = iparas['N2n']
        N4n = iparas['N4n']


        k24 = self.k24
        k42 = self.k42


        c2n = self.c2n
        c4n = self.c4n

        M_new[c2n] =(b2n*(-M[c2n] + self.Catm.lin(t + self.Dbirth))*N2n + 2*k42*(-2*M[c2n] + M[c4n] + self.Catm.lin(t + self.Dbirth))*N4n)/N2n
        M_new[c4n] = b4n*(-M[c4n] + self.Catm.lin(t + self.Dbirth)) + (k24*(M[c2n] - 2*M[c4n] + self.Catm.lin(t + self.Dbirth))*N2n)/(2.*N4n)
        M_new[self.q] = 0


        return M_new.ravel()






class POP2p(Hepatocyte):
    populations_DNA = {'N2n':1, 'N4n':2}
    plot_types = ['mean','2n','4n']
    populations = ['N2n','N4n']
    populations_plot = {'mean':populations,'2n':['N2n'],'4n':['N4n']}
    iparas = ['b2n', 'k24']
    flow_in =  {'N2n':[('b2n','N2n',2),('k42','N4n',4)],'N4n':[('b4n','N4n',2),('k24','N2n',1)]}
    
    def __init__(self, b4n=10**-1.0, k42=10**-2,d2n=10**-1.0, d4n=10**-1.0,dnatotal=dnatotal_const):
        default_parameters = dict(b4n=b4n, k42=k42,
                                  d2n=d2n, d4n=d4n)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}

        self.dnatotal  = dnatotal
        Hepatocyte.__init__(self)
        model_base.__init__(self, var_names=['c2n', 'c4n'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)


    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        b4n = self.b4n
        k42 = self.k42
        d2n = self.d2n
        d4n = self.d4n

        ploidy = self.ploidy(t)
        dtploidy = self.dtploidy(t)
        DNATotal,dtDNATotal = self.dnatotal(t)
        



        # populations 

        N2n = -((DNATotal*ploidy)/(-2 + ploidy))
        N4n = DNATotal*(1 + 1/(-2 + ploidy))

        # dt populations 

        dtN2n = (-((-2 + ploidy)*ploidy*dtDNATotal) + 2*DNATotal*dtploidy)/(-2 + ploidy)**2
        dtN4n = ((-2 + ploidy)*(-1 + ploidy)*dtDNATotal - DNATotal*dtploidy)/(-2 + ploidy)**2

        #implicit paras 
        b2n = (d2n*N2n + (-b4n + d4n - 3*k42)*N4n + dtN2n + dtN4n)/N2n
        k24 = ((-b4n + d4n + k42)*N4n + dtN4n)/N2n




        iparas = {'b2n': b2n, 'k24': k24,'N2n':N2n,'N4n':N4n}

        if b2n < 0:
            raise ImplicitParametersOutOfRange(
                f'b2n ({b2n})is negative at t = {t}', iparas)
        if k24 < 0:
            raise ImplicitParametersOutOfRange(
                f'k24 ({k24}) is negative at t = {t}', iparas)
        return iparas

    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        iparas = self.calc_implicit_parameters(t)
        
        b2n = iparas['b2n']
        k24 = iparas['k24']

        N2n = iparas['N2n']
        N4n = iparas['N4n']


        b4n = self.b4n
        k42 = self.k42

        c2n = self.c2n
        c4n = self.c4n

        M_new[c2n] =b2n*(-M[c2n] + self.Catm.lin(t + self.Dbirth)) + (2*k42*(-2*M[c2n] + M[c4n] + self.Catm.lin(t + self.Dbirth))*N4n)/N2n
        M_new[c4n] =b4n*(-M[c4n] + self.Catm.lin(t + self.Dbirth)) + (k24*(M[c2n] - 2*M[c4n] + self.Catm.lin(t + self.Dbirth))*N2n)/(2.*N4n)        

        return M_new.ravel()

    def measurement_model(self, result_sim, data):

        wn2,wn4 = self.calc_w(data)
        average = (result_sim['c2n']*wn2
                   + result_sim['c4n']*wn4 )

        return average




class POP2p_stem(Hepatocyte): 
    populations_DNA = {'N2n':1, 'N4n':2}
    plot_types = ['mean','2n','4n']
    populations = ['N2n','N4n']
    populations_plot = {'mean':populations,'2n':['N2n'],'4n':['N4n']}
    iparas = ['i2n', 'k24']
    flow_in =  {'N2n':[('i2n',None,1),('b2n','N2n',2),('k42','N4n',4)],'N4n':[('k24','N2n',1),('b4n','N4n',2)]}
    def __init__(self,dnatotal=dnatotal_const  ):
        default_parameters = dict(sigma=0.2,d2n=0.01,d4n=0.01,i2n=0.01,k24=0.001,k42=0.001)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        self.logparas = {'d2n','d4n','i2n','k42','k24'}
        self.linparas = {'sigma'}
        limit['sigma'] = (0,0.3)
        for i in self.linparas:
            limit[i] = (0,1)

        self.dnatotal = dnatotal

        Hepatocyte.__init__(self)
        model_base.__init__(self, var_names=['c2n','c4n'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)



    def calc_implicit_parameters(self, t):
        # make explicit parameters local
    
        d2n=self.d2n
        d4n = self.d4n
        i2n = self.i2n
        k42 = self.k42
        k24 = self.k24
        ploidy = self.ploidy(t)
        dtploidy = self.dtploidy(t)
        DNATotal,dtDNATotal = self.dnatotal(t)

        # populations

        N2n = -((DNATotal*ploidy)/(-2 + ploidy))
        N4n = DNATotal*(1 + 1/(-2 + ploidy))

        # dt populations 

        dtN2n = (-((-2 + ploidy)*ploidy*dtDNATotal) + 2*DNATotal*dtploidy)/(-2 + ploidy)**2
        dtN4n = ((-2 + ploidy)*(-1 + ploidy)*dtDNATotal - DNATotal*dtploidy)/(-2 + ploidy)**2

        #implicit paras 

        b2n = (-i2n + (d2n + k24)*N2n - 4*k42*N4n + dtN2n)/N2n
        b4n = (-(k24*N2n) + (d4n + k42)*N4n + dtN4n)/N4n
 
        iparas = {'b2n':b2n,'b4n':b4n,'N2n':N2n,'N4n':N4n}

        if b2n < 0:
            raise ImplicitParametersOutOfRange(
                f'b2n ({b2n})is negative at t = {t}', iparas)
        if b4n < 0:
            raise ImplicitParametersOutOfRange(
                f'b4n ({b4n})is nnegative at t = {t}', iparas)
        return iparas


    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        iparas = self.calc_implicit_parameters(t)
        N2n = iparas['N2n']
        N4n = iparas['N4n']
        b4n = iparas['b4n']
        b2n = iparas['b2n']
        
        i2n = self.i2n
        k42 = self.k42
        k24 = self.k24

        c2n = self.c2n
        c4n = self.c4n

        # c14 equations 

        M_new[c2n] =(-((M[c2n] - self.Catm.lin(t + self.Dbirth))*(i2n + b2n*N2n)) + 2*k42*(-2*M[c2n] + M[c4n] + self.Catm.lin(t + self.Dbirth))*N4n)/N2n
        M_new[c4n] =b4n*(-M[c4n] + self.Catm.lin(t + self.Dbirth)) + (k24*(M[c2n] - 2*M[c4n] + self.Catm.lin(t + self.Dbirth))*N2n)/(2.*N4n)

        return M_new.ravel()

    def measurement_model(self, result_sim, data):
        wn2,wn4 = self.calc_w(data)

        average = (result_sim['c2n']*wn2
                   + result_sim['c4n']*wn4
                   )

        return average




class POP3p_4nq(Hepatocyte):
    populations_DNA = {'N2n':1, 'N4n':2,'N4nq':2}
    plot_types = ['mean','2n','4n']
    populations = ['N2n','N4n','N4nq']
    populations_plot = {'mean':populations,'2n':['N2n'],'4n':['N4n','N4nq']}
    iparas = ['b2n', 'b4n','k24q']
    flow_in =  {'N2n':[('b2n','N2n',2),('k42','N4n',4)],'N4n':[('b4n','N4n',2),('k24','N2n',1)],'N4nq':[('k24q','N2n',1)]}
    def __init__(self, sigma=0.2, d2n=0.2, d4n=0.02,k24=0.00,k42=0.001,f=0.5,dnatotal=dnatotal_const ):
        default_parameters = dict(sigma=sigma,d2n=d2n,d4n=d4n,k42=k42,k24=k24,f=f)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        self.logparas = {'d2n','d4n','k24','k42'}
        self.linparas = {'sigma','f'}
        limit['sigma'] = (0,0.3)
        for i in self.linparas:
            limit[i] = (0,1)

        self.dnatotal = dnatotal
        Hepatocyte.__init__(self)
        model_base.__init__(self, var_names=['c2n','c4n','c4nq'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_turnover_and_flow(self,t):
        ip = self.calc_implicit_parameters(t)
        t2n = self.d2n + self.k24 + ip['b2n'] + ip['k24q']
        t4n = self.d4n + self.k42 + ip['b4n']
        N_total = 1/(2-self.ploidy(t))
        N2 = self.ploidy(t)    *N_total
        N4a = self.f*(1-self.ploidy(t))*N_total
        f2n = self.k42*N4a*4 + ip['b2n']*N2*2
        f4n = self.k24*N2 +   ip['b4n']*N4a*2
        return {'t2n':t2n,'t4n':t4n,'f2n':f2n,'f4n':f4n}

    def calc_implicit_parameters(self, t):
        d2n=self.d2n
        d4n = self.d4n
        k24 = self.k24
        k42 = self.k42
        f = self.f
        ploidy = self.ploidy(t)
        dtploidy = self.dtploidy(t)
        DNATotal,dtDNATotal = self.dnatotal(t)



        # populations 

        N2n = -((DNATotal*ploidy)/(-2 + ploidy))
        N4n = (f*DNATotal*(-1 + ploidy))/(-2 + ploidy)
        N4nq = -(((-1 + f)*DNATotal*(-1 + ploidy))/(-2 + ploidy))

        # dt populations 

        dtN2n = (-((-2 + ploidy)*ploidy*dtDNATotal) + 2*DNATotal*dtploidy)/(-2 + ploidy)**2
        dtN4n = (f*((-2 + ploidy)*(-1 + ploidy)*dtDNATotal - DNATotal*dtploidy))/(-2 + ploidy)**2
        dtN4nq = ((-1 + f)*(-((-2 + ploidy)*(-1 + ploidy)*dtDNATotal) + DNATotal*dtploidy))/(-2 + ploidy)**2

        #implicit paras 

  
        b2n = ((d2n + k24)*N2n - 4*k42*N4n + dtN2n + dtN4nq)/N2n
        b4n = (-(k24*N2n) + d4n*N4n + dtN4n)/N4n
        k24q = dtN4nq/N2n

        iparas = {'b2n':b2n,'b4n':b4n,'k24q':k24q,'N2n':N2n,'N4n':N4n,'N4nq':N4nq}

        if b2n < 0:
            raise ImplicitParametersOutOfRange(
                f'b2n ({b2n})is negative at t = {t}', iparas)
        if b4n < 0:
            raise ImplicitParametersOutOfRange(
                f'b4n ({b4n})is negative at t = {t}', iparas)
        if k24q < 0:
            raise ImplicitParametersOutOfRange(
                f'k24q ({k24q})is nnegative at t = {t}', iparas)
        return iparas


    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        iparas = self.calc_implicit_parameters(t)
        N2n = iparas['N2n']
        N4n = iparas['N4n']
        N4nq = iparas['N4nq']
        b2n = iparas['b2n']
        b4n = iparas['b4n']
        k24q = iparas['k24q']

        f = self.f
        k42 = self.k42
        k24 = self.k24

        c2n = self.c2n
        c4n = self.c4n
        c4nq = self.c4nq
        # c14 equations 

        M_new[c2n] =(b2n*(-M[c2n] + self.Catm.lin(t + self.Dbirth))*N2n + 2*k42*(-2*M[c2n] + M[c4n] + self.Catm.lin(t + self.Dbirth))*N4n)/N2n
        M_new[c4n] =b4n*(-M[c4n] + self.Catm.lin(t + self.Dbirth)) + (k24*(M[c2n] - 2*M[c4n] + self.Catm.lin(t + self.Dbirth))*N2n)/(2.*N4n)
        M_new[c4nq] =(k24q*(M[c2n] - 2*M[c4nq] + self.Catm.lin(t + self.Dbirth))*N2n)/(2.*N4nq)

        return M_new.ravel()

    def measurement_model(self, result_sim, data):
        f = self.f
        wn2,wn4t = self.calc_w(data)
        wn4 = f * wn4t
        wn4q = (1-f)*wn4t

        
        average = (result_sim['c2n']*wn2
                   + result_sim['c4n']*wn4
                   + result_sim['c4nq']*wn4q
                   )
        
        return average


class POP3p_4nq_alt(POP3p_4nq):
    populations_DNA = {'N2n':1, 'N4n':2,'N4nq':2}
    plot_types = ['mean','2n','4n']
    populations = ['N2n','N4n','N4nq']
    populations_plot = {'mean':populations,'2n':['N2n'],'4n':['N4n','N4nq']}
    iparas = ['b2n', 'b4n','k44q']
    flow_in =  {'N2n':[('b2n','N2n',2),('k42','N4n',4)],'N4n':[('b4n','N4n',2),('k24','N2n',1)],'N4nq':[('k44q','N4n',1)]}
    def __init__(self, sigma=0.2, d2n=0.2, d4n=0.02,k24=0.00,k42=0.001,f=0.5, dnatotal=dnatotal_const ):
        default_parameters = dict(sigma=sigma,d2n=d2n,d4n=d4n,k42=k42,k24=k24,f=f)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        self.logparas = {'d2n','d4n','k24','k42'}
        self.linparas = {'sigma','f'}
        limit['sigma'] = (0,0.3)
        self.dnatotal = dnatotal
        for i in self.linparas:
            limit[i] = (0,1)


        Hepatocyte.__init__(self)
        model_base.__init__(self, var_names=['c2n','c4n','c4nq'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_turnover_and_flow(self,t):
        ip = self.calc_implicit_parameters(t)
        t2n = self.d2n + self.k24 + ip['b2n'] + ip['k24q']
        t4n = self.d4n + self.k42 + ip['b4n']
        N_total = 1/(2-self.ploidy(t))
        N2 = self.ploidy(t)    *N_total
        N4a = self.f*(1-self.ploidy(t))*N_total
        f2n = self.k42*N4a*4 + ip['b2n']*N2*2
        f4n = self.k24*N2 +   ip['b4n']*N4a*2
        return {'t2n':t2n,'t4n':t4n,'f2n':f2n,'f4n':f4n}

    def calc_implicit_parameters(self, t):
        d2n=self.d2n
        d4n = self.d4n
        k24 = self.k24
        k42 = self.k42
        f = self.f
        ploidy = self.ploidy(t)
        dtploidy = self.dtploidy(t)
        DNATotal,dtDNATotal = self.dnatotal(t)

        # populations 

        N2n = -((DNATotal*ploidy)/(-2 + ploidy))
        N4n = (f*DNATotal*(-1 + ploidy))/(-2 + ploidy)
        N4nq = -(((-1 + f)*DNATotal*(-1 + ploidy))/(-2 + ploidy))

        # dt populations 

        dtN2n = (-((-2 + ploidy)*ploidy*dtDNATotal) + 2*DNATotal*dtploidy)/(-2 + ploidy)**2
        dtN4n = (f*((-2 + ploidy)*(-1 + ploidy)*dtDNATotal - DNATotal*dtploidy))/(-2 + ploidy)**2
        dtN4nq = ((-1 + f)*(-((-2 + ploidy)*(-1 + ploidy)*dtDNATotal) + DNATotal*dtploidy))/(-2 + ploidy)**2

        #implicit paras 

  
        b2n = ((d2n + k24)*N2n - 4*k42*N4n + dtN2n)/N2n
        b4n = (-(k24*N2n) + (d4n + k42)*N4n + dtN4n + dtN4nq)/N4n
        k44q = dtN4nq/N4n

        iparas = {'b2n':b2n,'b4n':b4n,'k44q':k44q,'N2n':N2n,'N4n':N4n,'N4nq':N4nq}

        if b2n < 0:
            raise ImplicitParametersOutOfRange(
                f'b2n ({b2n})is negative at t = {t}', iparas)
        if b4n < 0:
            raise ImplicitParametersOutOfRange(
                f'b4n ({b4n})is negative at t = {t}', iparas)
        if k44q < 0:
            raise ImplicitParametersOutOfRange(
                f'k44q ({k44q})is nnegative at t = {t}', iparas)
        return iparas


    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        iparas = self.calc_implicit_parameters(t)
        N2n = iparas['N2n']
        N4n = iparas['N4n']
        N4nq = iparas['N4nq']
        b2n = iparas['b2n']
        b4n = iparas['b4n']
        k44q = iparas['k44q']

        f = self.f
        k42 = self.k42
        k24 = self.k24

        c2n = self.c2n
        c4n = self.c4n
        c4nq = self.c4nq
        # c14 equations 

        M_new[c2n] =(b2n*(-M[c2n] + self.Catm.lin(t + self.Dbirth))*N2n + 2*k42*(-2*M[c2n] + M[c4n] + self.Catm.lin(t + self.Dbirth))*N4n)/N2n
        M_new[c4n] =b4n*(-M[c4n] + self.Catm.lin(t + self.Dbirth)) + (k24*(M[c2n] - 2*M[c4n] + self.Catm.lin(t + self.Dbirth))*N2n)/(2.*N4n)
        M_new[c4nq] =(k44q*(M[c4n] - M[c4nq])*N4n)/N4nq

        return M_new.ravel()





class POP3p_4nq_new(Hepatocyte):
    populations_DNA = {'N2n':1, 'N4n':2,'N4nq':2}
    plot_types = ['mean','2n','4n']
    populations = ['N2n','N4n','N4nq']
    populations_plot = {'mean':populations,'2n':['N2n'],'4n':['N4n','N4nq']}
    iparas = ['b2n', 'b4n']
    flow_in =  {'N2n':[('b2n','N2n',2),('k42','N4n',4)],'N4n':[('b4n','N4n',2),('k24','N2n',1)],'N4nq':[('q4','N4n',1)]}
    def __init__(self, sigma=0.2, d2n=0.2, d4n=0.02,k24=0.00,k42=0.001,q4=0.005, dnatotal=dnatotal_const ):
        default_parameters = dict(sigma=sigma,d2n=d2n,d4n=d4n,k42=k42,k24=k24,q4=q4)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        self.logparas = {'d2n','d4n','k24','k42','q4'}
        self.linparas = {'sigma'}
        limit['sigma'] = (0,0.3)
        self.dnatotal = dnatotal
        for i in self.linparas:
            limit[i] = (0,1)


        Hepatocyte.__init__(self)
        model_base.__init__(self, var_names=['c2n','c4n','c4nq'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_initial_parameters(self):
        dx=0.1
        ti = np.arange(0,100+dx/2,dx)
        ploidy = self.ploidy(ti)
        dtploidy = self.dtploidy(ti)
        DNATotal,dtDNATotal = self.dnatotal(ti)
        q4 = self.q4
    
        N4nT = DNATotal*(1 + 1/(-2 + ploidy))

        # dt populations 

        dtN4nT = ((-2 + ploidy)*(-1 + ploidy)*dtDNATotal - DNATotal*dtploidy)/(-2 + ploidy)**2
        
        resint = cumulative_trapezoid(np.exp(q4*ti)*dtN4nT,dx=dx,initial=0)
        N4n = np.exp(-q4*ti)*(N4nT[0] + resint)
        N4nq = N4nT - N4n
  
        dtN4n = -N4nT[0]*q4*np.exp(-q4*ti)-q4*np.exp(-q4*ti)*resint+dtN4nT
        dtN4nq = dtN4nT - dtN4n
    
    
        self.N4n = interp1d(ti, N4n)
        self.N4nq= interp1d(ti, N4nq)
        self.dtN4n=  interp1d(ti, dtN4n)
        self.dtN4nq=interp1d(ti, dtN4nq)

    def calc_implicit_parameters(self, t):
        d2n=self.d2n
        d4n = self.d4n
        k24 = self.k24
        k42 = self.k42
        ploidy = self.ploidy(t)
        dtploidy = self.dtploidy(t)
        DNATotal,dtDNATotal = self.dnatotal(t)

        # populations 

        N2n = -((DNATotal*ploidy)/(-2 + ploidy))
        N4n = float(self.N4n(t))
        N4nq = float(self.N4nq(t))

        # dt populations 

        dtN2n = (-((-2 + ploidy)*ploidy*dtDNATotal) + 2*DNATotal*dtploidy)/(-2 + ploidy)**2
        dtN4n = float(self.dtN4n(t))
        dtN4nq = float(self.dtN4nq(t))
        #implicit paras 

  
        b2n = ((d2n + k24)*N2n - 4*k42*N4n + dtN2n)/N2n
        b4n = (-(k24*N2n) + (d4n + k42)*N4n + dtN4n + dtN4nq)/N4n
        
        if (N4nq<1e-6):
            N4nq=1e-6

        iparas = {'b2n':b2n,'b4n':b4n,'N2n':N2n,'N4n':N4n,'N4nq':N4nq}
        
        if N4n < 0:
            raise ImplicitParametersOutOfRange(
                f'N4n ({N4n})is negative at t = {t}', iparas)
        if b2n < 0:
            raise ImplicitParametersOutOfRange(
                f'b2n ({b2n})is negative at t = {t}', iparas)
        if b4n < 0:
            raise ImplicitParametersOutOfRange(
                f'b4n ({b4n})is negative at t = {t}', iparas)
        return iparas


    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        iparas = self.calc_implicit_parameters(t)
        N2n = iparas['N2n']
        N4n = iparas['N4n']
        N4nq = iparas['N4nq']
        b2n = iparas['b2n']
        b4n = iparas['b4n']
        k44q = self.q4

        k42 = self.k42
        k24 = self.k24

        c2n = self.c2n
        c4n = self.c4n
        c4nq = self.c4nq
        # c14 equations 
        M_new[c2n] =(b2n*(-M[c2n] + self.Catm.lin(t + self.Dbirth))*N2n + 2*k42*(-2*M[c2n] + M[c4n] + self.Catm.lin(t + self.Dbirth))*N4n)/N2n
        M_new[c4n] =b4n*(-M[c4n] + self.Catm.lin(t + self.Dbirth)) + (k24*(M[c2n] - 2*M[c4n] + self.Catm.lin(t + self.Dbirth))*N2n)/(2.*N4n)
        M_new[c4nq] =(k44q*(M[c4n] - M[c4nq])*N4n)/N4nq

        return M_new.ravel()


    def measurement_model(self, result_sim, data):
        N4n = self.N4n(data.age)
        N4nq = self.N4nq(data.age)
        f = N4n/(N4n+N4nq)
        wn2,wn4t = self.calc_w(data)
        wn4 = f * wn4t
        wn4q = (1-f)*wn4t
        
        average = (result_sim['c2n']*wn2
                   + result_sim['c4n']*wn4
                   + result_sim['c4nq']*wn4q
                   )
        
        return average




class POP3p_2nb(Hepatocyte):
    populations_DNA = {'N2n':1, 'N4n':2,'N2nb':1}
    plot_types = ['mean','2n','4n']
    populations = ['N2n','N4n','N2nb']
    populations_plot = {'mean':populations,'2n':['N2n','N2nb'],'4n':['N4n']}
    iparas = ['b2n', 'b4n','b2nb']
    flow_in =  {'N2n':[('b2n','N2n',2),('k42','N4n',4)],
        'N4n':[('b4n','N4n',2),('k24','N2n',1),('k2b4','N2nb',1)],
        'N2nb':[('b2nb','N2nb',2),('k42','N4n',4)]}
    def __init__(self, sigma=0.2, d2n=0.2, d4n=0.02,k24=0.00,k42=0.001,f=0.5,d2nb=0.01, DNATotal = lambda x:1 ):
        default_parameters = dict(sigma=sigma,d2n=d2n,d4n=d4n,k42=k42,k24=k24,d2nb=d2nb,f=f)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        self.logparas = {'d2n','d4n','k24','k42','d2nb'}
        self.linparas = {'sigma','f'}
        limit['sigma'] = (0,0.3)
        for i in self.linparas:
            limit[i] = (0,1)

        self.DNATotal = DNATotal
        Hepatocyte.__init__(self)
        model_base.__init__(self, var_names=['c2n','c2nb','c4n'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_turnover_and_flow(self,t):
        ip = self.calc_implicit_parameters(t)
        k2b4 = ip['b2nb'] * self.k24 / ip['b2n']
        t2na = self.d2n +  self.k24 + ip['b2n']
        t2nb = self.d2nb + k2b4     + ip['b2nb']
        t4n = self.d4n + 2*self.k42 + ip['b4n']
        N_total = 1/(2-self.ploidy(t))
        N2a = self.f * self.ploidy(t)    *N_total
        N2b = (1-self.f)*self.ploidy(t)    *N_total
        N4 = (1-self.ploidy(t))*N_total
        f2na = self.k42*N4*4 + ip['b2n']*N2a*2
        f2nb = self.k42*N4*4 + ip['b2n']*N2b*2
        f4n = self.k24*N2a + k2b4*N2b +   ip['b4n']*N4*2
        return {'t2n':t2na*self.f + t2nb*(1+self.f),'t4n':t4n,'t2na':t2na,'t2nb':t2nb,'f2n':f2na + f2nb,'f4n':f4n,'f2na':f2na,'f2na':f2na}

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        ploidy = self.ploidy(t)
        dtploidy = self.dtploidy(t)
    
        d2n=self.d2n
        d4n = self.d4n
        d2nb = self.d2nb
        k24 =self.k24
        k42 = self.k42
        f=self.f
        

        # populations 

        #N2n = -((f*ploidy)/(-2 + ploidy))
        #N4n = 1 + 1/(-2 + ploidy)
        #N2nb = ((-1 + f)*ploidy)/(-2 + ploidy)
        N2n = -((f*self.DNATotal*ploidy)/(-2 + ploidy))
        N4n = self.DNATotal*(1 + 1/(-2 + ploidy))
        N2nb = ((-1 + f)*self.DNATotal*ploidy)/(-2 + ploidy)
        

        # dt populations 

        dtN2n = (2*f*dtploidy)/(-2 + ploidy)**2
        dtN4n = -(dtploidy/(-2 + ploidy)**2)
        dtN2nb = (-2*(-1 + f)*dtploidy)/(-2 + ploidy)**2

        #implicit paras 

        b2n = ((d2n + k24)*N2n - 4*k42*N4n + dtN2n)/N2n
        b4n = -((d2n*k24*N2n**2 + (4*k42*N4n - dtN2n)*((d4n + 2*k42)*N4n + dtN4n) + N2n*(k24*(d2nb*N2nb - 8*k42*N4n + dtN2n + dtN2nb) - d2n*((d4n + 2*k42)*N4n + dtN4n)))/(N4n*(d2n*N2n - 4*k42*N4n + dtN2n)))
        b2nb = (((d2n + k24)*N2n - 4*k42*N4n + dtN2n)*(d2nb*N2nb - 4*k42*N4n + dtN2nb))/(N2nb*(d2n*N2n - 4*k42*N4n + dtN2n))
    
        k2b4 = b2nb * k24 / b2n
    
        iparas = {'b2n':b2n,'b4n':b4n,'b2nb':b2nb,'N2n':N2n,'N2nb':N2nb,'N4n':N4n,'k2b4':k2b4}

        if b2n < 0:
            raise ImplicitParametersOutOfRange(
                f'b2n ({b2n})is negative at t = {t}', iparas)
        if b4n < 0:
            raise ImplicitParametersOutOfRange(
                f'b4n ({b4n})is negative at t = {t}', iparas)
        if b2nb < 0:
            raise ImplicitParametersOutOfRange(
                f'b2nb ({b2nb})is negative at t = {t}', iparas)
        return iparas


    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        iparas = self.calc_implicit_parameters(t)
        N2n = iparas['N2n']
        N4n = iparas['N4n']
        N2nb = iparas['N2nb']
        b2n = iparas['b2n']
        b4n = iparas['b4n']
        b2nb = iparas['b2nb']

        f = self.f
        k42 = self.k42
        k24 = self.k24

        c2n = self.c2n
        c4n = self.c4n
        c2nb = self.c2nb
        # c14 equations 

        M_new[c2n] =(b2n*(-M[c2n] + self.Catm.lin(t + self.Dbirth))*N2n + 2*k42*(-2*M[c2n] + M[c4n] + self.Catm.lin(t + self.Dbirth))*N4n)/N2n
        M_new[c4n] =(b2n*k24*(M[c2n] - 2*M[c4n] + self.Catm.lin(t + self.Dbirth))*N2n + b2nb*k24*(M[c2nb] - 2*M[c4n] + self.Catm.lin(t + self.Dbirth))*N2nb + 2*b2n*b4n*(-M[c4n] + self.Catm.lin(t + self.Dbirth))*N4n)/(2.*b2n*N4n)
        M_new[c2nb] =(b2nb*(-M[c2nb] + self.Catm.lin(t + self.Dbirth))*N2nb + 2*k42*(-2*M[c2nb] + M[c4n] + self.Catm.lin(t + self.Dbirth))*N4n)/N2nb




        return M_new.ravel()

    def measurement_model(self, result_sim, data):
        f = self.f
        wn2t,wn4 = self.calc_w(data)
        wn2 = f*wn2t
        wn2b = (1-f)*wn2t
       

        
        average = (result_sim['c2n']*wn2
                  + result_sim['c2nb']*wn2b
                   + result_sim['c4n']*wn4
                   )
        
        return average




class POP3p_2nb_int(POP3p_2nb):
    populations_DNA = {'N2n':1, 'N4n':2,'N2nb':1}
    plot_types = ['mean','2n','4n']
    populations = ['N2n','N4n','N2nb']
    populations_plot = {'mean':populations,'2n':['N2n','N2nb'],'4n':['N4n']}
    iparas = ['b2n', 'b4n','b2nb']
    flow_in =  {'N2n':[('b2n','N2n',2),('k42','N4n',4),('gBA','N2nb',1)],
            'N4n':[('b4n','N4n',2),('k24','N2n',1)],
            'N2nb':[('b2nb','N2nb',2),('gAB','N2n',1)]}
    def __init__(self, sigma=0.2, d2n=0.2, d4n=0.02,k24=0.00,k42=0.001,f=0.5,d2nb=0.01,gAB=0.1,gBA=0.1,dnatotal = dnatotal_const):
        default_parameters = dict(sigma=sigma,d2n=d2n,d4n=d4n,k42=k42,k24=k24,d2nb=d2nb,f=f,gAB=gAB,gBA=gBA)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        self.logparas = {'d2n','d4n','k24','k42','d2nb','gAB','gBA'}
        self.linparas = {'sigma','f'}
        limit['sigma'] = (0,0.3)
        for i in self.linparas:
            limit[i] = (0,1)

        self.dnatotal = dnatotal
        Hepatocyte.__init__(self)
        model_base.__init__(self, var_names=['c2n','c2nb','c4n'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)


    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        ploidy = self.ploidy(t)
        dtploidy = self.dtploidy(t)
        DNATotal,dtDNATotal = self.dnatotal(t)
        d2n=self.d2n
        d4n = self.d4n
        d2nb = self.d2nb
        k24 =self.k24
        k42 = self.k42
        gAB = self.gAB
        gBA = self.gBA
        f=self.f
        

        # populations 

        N2n = -((f*DNATotal*ploidy)/(-2 + ploidy))
        N4n = DNATotal*(1 + 1/(-2 + ploidy))
        N2nb = ((-1 + f)*DNATotal*ploidy)/(-2 + ploidy)

        # dt populations 

        dtN2n = (-(f*(-2 + ploidy)*ploidy*dtDNATotal) + 2*f*DNATotal*dtploidy)/(-2 + ploidy)**2
        dtN4n = ((-2 + ploidy)*(-1 + ploidy)*dtDNATotal - DNATotal*dtploidy)/(-2 + ploidy)**2
        dtN2nb = ((-1 + f)*((-2 + ploidy)*ploidy*dtDNATotal - 2*DNATotal*dtploidy))/(-2 + ploidy)**2



        #implicit paras 
        
        b2n = ((d2n + gAB + k24)*N2n - gBA*N2nb - 4*k42*N4n + dtN2n)/N2n
        b4n = (-(k24*N2n) + (d4n + k42)*N4n + dtN4n)/N4n
        b2nb = (-(gAB*N2n) + (d2nb + gBA)*N2nb + dtN2nb)/N2nb
            
        iparas = {'b2n':b2n,'b4n':b4n,'b2nb':b2nb,'N2n':N2n,'N2nb':N2nb,'N4n':N4n}

        if b2n < 0:
            raise ImplicitParametersOutOfRange(
                f'b2n ({b2n})is negative at t = {t}', iparas)
        if b4n < 0:
            raise ImplicitParametersOutOfRange(
                f'b4n ({b4n})is negative at t = {t}', iparas)
        if b2nb < 0:
            raise ImplicitParametersOutOfRange(
                f'b2nb ({b2nb})is negative at t = {t}', iparas)
        return iparas


    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        iparas = self.calc_implicit_parameters(t)
        N2n = iparas['N2n']
        N4n = iparas['N4n']
        N2nb = iparas['N2nb']
        b2n = iparas['b2n']
        b4n = iparas['b4n']
        b2nb = iparas['b2nb']

        f = self.f
        k42 = self.k42
        k24 = self.k24
        gAB = self.gAB
        gBA = self.gBA
        
        c2n = self.c2n
        c4n = self.c4n
        c2nb = self.c2nb
        # c14 equations 

        M_new[c2n] =(b2n*self.Catm.lin(t + self.Dbirth)*N2n + gBA*M[c2nb]*N2nb + 2*k42*(M[c4n] + self.Catm.lin(t + self.Dbirth))*N4n - M[c2n]*(b2n*N2n + gBA*N2nb + 4*k42*N4n))/N2n
        M_new[c4n] =b4n*(-M[c4n] + self.Catm.lin(t + self.Dbirth)) + (k24*(M[c2n] - 2*M[c4n] + self.Catm.lin(t + self.Dbirth))*N2n)/(2.*N4n)
        M_new[c2nb] =b2nb*(-M[c2nb] + self.Catm.lin(t + self.Dbirth)) + (gAB*(M[c2n] - M[c2nb])*N2n)/N2nb

        return M_new.ravel()


class POP4p_conv(Hepatocyte):
    populations_DNA = {'N2n':1, 'N4n':2,'N2nb':1,'N4nb':2}
    plot_types = ['mean','2n','4n']
    populations = ['N2n','N4n','N2nb','N4nb']
    populations_plot = {'mean':populations,'2n':['N2n','N2nb'],'4n':['N4n','N4nb']}
    iparas = ['b2n', 'b4n','alpha','f']
    flow_in =  {'N2n':[('b2n','N2n',2),('k42','N4n',4)],'N4n':[('b4n','N4n',2),('k24','N2n',1)],
        'N2nb':[('alpha','N2n',1)], 'N4nb':[('alpha','N4n',1)] }    
    def __init__(self, sigma=0.2, d2n=0.2, d4n=0.01,k24=0.00,k42=0.001,g=0.5, dnatotal=dnatotal_const ):
        default_parameters = dict(sigma=sigma,d2n=d2n,d4n=d4n,k42=k42,k24=k24,g=g)
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        self.logparas = {'d2n','d4n','k24','k42'}
        self.linparas = {'sigma','g',}
        limit['sigma'] = (0,0.3)
        for i in self.linparas:
            limit[i] = (0,1)
        
        self.dnatotal = dnatotal


        Hepatocyte.__init__(self)
        model_base.__init__(self, var_names=['c2n','c4n','c2nb','c4nb'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_turnover_and_flow(self,t):
        ip = self.calc_implicit_parameters(t)
        t2n = self.d2n + self.k24 + ip['b2n'] + ip['alpha']
        t4n = self.d4n + self.k42 + ip['b4n'] + ip['alpha']
        N_total = 1/(2-self.ploidy(t))
        N2a = ip['f'] * self.ploidy(t)    *N_total
        N2b = (1-ip['f'])*self.ploidy(t)    *N_total
        N4a = self.g*(1-self.ploidy(t))*N_total
        N4b = (1-self.g)*(1-self.ploidy(t))*N_total
        f2na = self.k42*N4a*4 + ip['b2n']*N2a*2
        f2nb = ip['alpha']*N2a
        f4na = self.k24*N2a    + ip['b4n']*N4a*2
        f4nb = ip['alpha']*N4a
        return {'t2n':t2n,'t4n':t4n,'f2n':f2na,'f4n':f4na,'f2nb':f2nb,'f4nb':f4nb}

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        ploidy = self.ploidy(t)
        dtploidy = self.dtploidy(t)
        DNATotal,dtDNATotal = self.dnatotal(t)

        g = self.g
        d4n = self.d4n

        d2n = self.d2n
        k24 = self.k24
        k42 = self.k42

        # populations 

        N2ntot = -((DNATotal*ploidy)/(-2 + ploidy))
        N4ntot = DNATotal*(1 + 1/(-2 + ploidy))

        # dt populations 

        dtN2ntot = (-((-2 + ploidy)*ploidy*dtDNATotal) + 2*DNATotal*dtploidy)/(-2 + ploidy)**2
        dtN4ntot = ((-2 + ploidy)*(-1 + ploidy)*dtDNATotal - DNATotal*dtploidy)/(-2 + ploidy)**2

        #implicit paras 

        b2n = k24 + ((d2n*N2ntot + dtN2ntot)*(d2n*N2ntot - 4*g*k42*N4ntot + dtN2ntot))/(N2ntot*(d2n*N2ntot + d4n*(-1 + g)*N4ntot + dtN2ntot + (-1 + g)*dtN4ntot))
        b4n = (-(d2n**2*k24*N2ntot**4) - (-1 + g)*g*N4ntot*dtN2ntot**2*(d4n*N4ntot + dtN4ntot) + d2n*N2ntot**3*((-2*d4n*(-1 + g)*k24 + d2n*g*(d4n + k42))*N4ntot - 2*k24*dtN2ntot + (d2n*g - 2*(-1 + g)*k24)*dtN4ntot) + g*N2ntot*dtN2ntot*(d4n*(-1 + g)*(-2*d2n + d4n + k42)*N4ntot**2 + dtN4ntot*(dtN2ntot + (-1 + g)*dtN4ntot) + N4ntot*((d4n + k42)*dtN2ntot + (-1 + g)*(-2*d2n + 2*d4n + k42)*dtN4ntot)) + N2ntot**2*(d4n*(-1 + g)*(-(d2n**2*g) - d4n*(-1 + g)*k24 + d2n*g*(d4n + k42))*N4ntot**2 - k24*dtN2ntot**2 + 2*(d2n*g + k24 - g*k24)*dtN2ntot*dtN4ntot + (-1 + g)*(d2n*g + k24 - g*k24)*dtN4ntot**2 + N4ntot*(2*(-(d4n*(-1 + g)*k24) + d2n*g*(d4n + k42))*dtN2ntot + (-1 + g)*(-(d2n**2*g) - 2*d4n*(-1 + g)*k24 + d2n*g*(2*d4n + k42))*dtN4ntot)))/(g*N2ntot*N4ntot*(d2n*N2ntot + dtN2ntot)*(d2n*N2ntot + d4n*(-1 + g)*N4ntot + dtN2ntot + (-1 + g)*dtN4ntot))
        alpha = -(((-1 + g)*(d2n*N2ntot + dtN2ntot)*(d4n*N4ntot + dtN4ntot))/(N2ntot*(d2n*N2ntot + d4n*(-1 + g)*N4ntot + dtN2ntot + (-1 + g)*dtN4ntot)))
        f = (d2n*N2ntot + d4n*(-1 + g)*N4ntot + dtN2ntot + (-1 + g)*dtN4ntot)/(d2n*N2ntot + dtN2ntot)

        # populations 

        N2n = f*N2ntot
        N4n = g*N4ntot
        N2nb = -((-1 + f)*N2ntot)
        N4nb = -((-1 + g)*N4ntot)

        # dt populations 

        dtN2n = f*dtN2ntot
        dtN4n = g*dtN4ntot
        dtN2nb = -((-1 + f)*dtN2ntot)
        dtN4nb = -((-1 + g)*dtN4ntot)



        iparas = {'b2n':b2n,'b4n':b4n,'alpha':alpha,'f':f,'N2n':N2n,'N4n':N4n,'N2nb':N2nb,'N4nb':N4nb}

        if b2n < 0:
            raise ImplicitParametersOutOfRange(
                f'b2n ({b2n})is negative at t = {t}', iparas)
        if b4n < 0:
            raise ImplicitParametersOutOfRange(
                f'b4n ({b4n})is negative at at t = {t}', iparas)
        if alpha < 0:
            raise ImplicitParametersOutOfRange(
                f'alpha ({alpha})is negative at at t = {t}', iparas)
        if f < 0 or f>1:
            raise ImplicitParametersOutOfRange(
                f'f ({f})is negative at at t = {t}', iparas)
        return iparas


    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        iparas = self.calc_implicit_parameters(t)
        N2n = iparas['N2n']
        N2nb = iparas['N2nb']
        N4n = iparas['N4n']
        N4nb = iparas['N4nb']
        b2n = iparas['b2n']
        b4n = iparas['b4n']
        alpha = iparas['alpha']
        f = iparas['f']

        k42 = self.k42
        k24 = self.k24

        c2n = self.c2n
        c4n = self.c4n
        c2nb = self.c2nb
        c4nb = self.c4nb


        # c14 equations 
        M_new[c2n] =(b2n*(-M[c2n] + self.Catm.lin(t + self.Dbirth))*N2n + 2*k42*(-2*M[c2n] + M[c4n] + self.Catm.lin(t + self.Dbirth))*N4n)/N2n
        M_new[c4n] =b4n*(-M[c4n] + self.Catm.lin(t + self.Dbirth)) + (k24*(M[c2n] - 2*M[c4n] + self.Catm.lin(t + self.Dbirth))*N2n)/(2.*N4n)
        M_new[c2nb] =(alpha*(M[c2n] - M[c2nb])*N2n)/N2nb
        M_new[c4nb] =(alpha*(M[c2n] - M[c4nb])*N2n)/N4nb
            
        return M_new.ravel()


    def measurement_model(self, result_sim, data):
        
  
        iparas = [self.calc_implicit_parameters(a) for a in data.age]
        f = np.array(listofdict_to_dictoflist(iparas)['f'])
        g = self.g

        wn2,wn4 = self.calc_w(data)
        wn2b = (1-f)*wn2
        wn4b = (1-g)*wn4
        
        average = (result_sim['c2n']*wn2
                  + result_sim['c2nb']*wn2b
                  + result_sim['c4n']*wn4
                  + result_sim['c4nb']*wn4b
                   )
        
        return average




class POP3p_2x2n_NP(Hepatocyte):
    """ Liver model,
        r2 and r4 are implicitely time dependent
    """

    populations_DNA = {'N2n':1, 'N2x2n':2 ,'N4n':2}
    plot_types = ['mean','2n','pn']
    populations = ['N2n','N4n','N2x2n']
    populations_plot = {'mean':populations,'2n':['N2n'],'pn':['N4n','N2x2n']}
    iparas = ['g2nb2n','k2x2nb4n','k4nb4n','k4nb2x2n','k2x2nb2x2n']
    flow_in =  {'N2n':[('g2nb2n','N2n',2),('k2nb2x2n','N2x2n',4),('k2nb4n','N4n',4)],
            'N2x2n':[('g2x2nb2n','N2n',1),('k2x2nb2x2n','N2x2n',2),('k2x2nb4n','N4n',2)],
            'N4n':[('k4nb2x2n','N2x2n',2),('k4nb4n','N4n',2)]}
    def __init__(self, sigma=0.2, d2n=0.2, d2x2n=0.02, d4n=0.02,k2nb4n=0.1,g2x2nb2n=0.01,k2nb2x2n=0.01,dnatotal=dnatotal_const ):
        default_parameters = dict(sigma=sigma,d2n=d2n, d2x2n=d2x2n,d4n=d4n,k2nb4n=k2nb4n,g2x2nb2n=g2x2nb2n,k2nb2x2n=k2nb2x2n )
        error = {parameter_name: global_error for parameter_name in
                 default_parameters.keys()}
        limit = {parameter_name: global_limit for parameter_name in
                 default_parameters.keys()}
        self.logparas = {'d2n','d2x2n','d4n','k2nb4n','g2x2nb2n','k2nb2x2n'}  
        self.linparas = {'sigma'}
        limit['sigma'] = (0,0.3)
        for i in self.linparas:
            limit[i] = (0,1)

        self.dnatotal = dnatotal
            
        Hepatocyte.__init__(self)
        model_base.__init__(self, var_names=['c2n', 'c2x2n', 'c4n'],
                            default_parameters=default_parameters,
                            error=error,
                            limit=limit)

    def calc_initial_parameters(self):
        return 

    def calc_turnover_and_flow(self,t):
        ip = self.calc_implicit_parameters(t)

        k2x2nb2x2n = (self.k2nb2x2n*ip['k2x2nb4n'])/self.k2nb4n
        k4nb2x2n = (self.k2nb2x2n*ip['k4nb4n'])/self.k2nb4n

        t2n  = self.d2n + self.g2x2nb2n + ip['g2nb2n']
        t2x2n= self.d2x2n + self.k2nb2x2n + k4nb2x2n + k2x2nb2x2n
        t4n = self.d4n + self.k2nb4n + ip['k2x2nb4n'] + ip['k4nb4n']
        N_total = 1/(2-self.ploidy(t))
        N2 =   self.ploidy(t)    *N_total
        N2x2 = self.ploidy2x2(t)*N_total
        N4 = (1 - N2 - 2*N2x2)/2.0

        f2n =  (self.k2nb2x2n*N2x2 +  self.k2nb4n*N4)*4 + ip['g2nb2n']*N2*2
        f2x2n = self.g2x2nb2n*N2 + (k2x2nb2x2n*N2x2 + ip['k2x2nb4n']*N4)*2
        f4n =   (k4nb2x2n*N2x2 + ip['k4nb4n']*N4)*2
        return {'t2n':t2n,'t4n':t4n,'t2x2n':t2x2n,'f2n':f2n,'f4n':f4n,'f2x2n':f2x2n}

    def calc_implicit_parameters(self, t):
        # make explicit parameters local
        ploidy = self.ploidy(t)
        dtploidy = self.dtploidy(t)
        ploidy2x2 = self.ploidy2x2(t)
        dtploidy2x2 = self.dtploidy2x2(t)        
        DNATotal,dtDNATotal = self.dnatotal(t)

        d2n=self.d2n
        d2x2n=self.d2x2n
        d4n = self.d4n

        k2nb4n = self.k2nb4n
        g2x2nb2n = self.g2x2nb2n
        k2nb2x2n = self.k2nb2x2n


        # populations 

        N2n = -((DNATotal*ploidy)/(-2 + ploidy))
        N4n = (DNATotal*(-1 + ploidy + ploidy2x2))/(-2 + ploidy)
        N2x2n = -((DNATotal*ploidy2x2)/(-2 + ploidy))

        # dt populations 

        dtN2n = (-((-2 + ploidy)*ploidy*dtDNATotal) + 2*DNATotal*dtploidy)/(-2 + ploidy)**2
        dtN4n = ((-2 + ploidy)*(-1 + ploidy + ploidy2x2)*dtDNATotal + DNATotal*(-((1 + ploidy2x2)*dtploidy) + (-2 + ploidy)*dtploidy2x2))/(-2 + ploidy)**2
        dtN2x2n = (ploidy2x2*(-((-2 + ploidy)*dtDNATotal) + DNATotal*dtploidy) - DNATotal*(-2 + ploidy)*dtploidy2x2)/(-2 + ploidy)**2
        #implicit paras 

        g2nb2n = ((d2n + g2x2nb2n)*N2n - 4*k2nb2x2n*N2x2n - 4*k2nb4n*N4n + dtN2n)/N2n
        k2x2nb4n = (k2nb4n*(2*k2nb2x2n*(d2x2n + k2nb2x2n)*N2x2n**2 - g2x2nb2n*N2n*(2*k2nb2x2n*N2x2n + k2nb4n*N4n) + k2nb4n*N4n*dtN2x2n + N2x2n*((d4n*k2nb2x2n + d2x2n*k2nb4n + 2*k2nb2x2n*k2nb4n)*N4n + k2nb2x2n*(2*dtN2x2n + dtN4n))))/(2.*(k2nb2x2n*N2x2n + k2nb4n*N4n)**2)       
        k4nb4n = (k2nb4n*N4n*(d4n*k2nb2x2n*N2x2n + k2nb4n*(-(g2x2nb2n*N2n) + (d2x2n + 2*k2nb2x2n)*N2x2n + 2*(d4n + k2nb4n)*N4n + dtN2x2n)) + k2nb4n*(k2nb2x2n*N2x2n + 2*k2nb4n*N4n)*dtN4n)/(2.*(k2nb2x2n*N2x2n + k2nb4n*N4n)**2)
        
        k2x2nb2x2n = k2nb2x2n * k2x2nb4n / k2nb4n
        k4nb2x2n = k2nb2x2n * k4nb4n / k2nb4n

        iparas = { 'g2nb2n':g2nb2n,'k2x2nb4n':k2x2nb4n,'k4nb4n':k4nb4n,'k4nb2x2n':k4nb2x2n,'k2x2nb2x2n':k2x2nb2x2n,
                'N4n':N4n,'N2n':N2n,'N2x2n':N2x2n }

        if g2nb2n < 0:
            raise ImplicitParametersOutOfRange(
                f'g2nb2n ({g2nb2n})is negative at t = {t}', iparas)
        if k2x2nb4n < 0:
            raise ImplicitParametersOutOfRange(
                f'k2x2nb4n ({k2x2nb4n})is not in [0,1] at t = {t}', iparas)
        if k4nb4n < 0 :
            raise ImplicitParametersOutOfRange(
                f'k4nb4n ({k4nb4n})is negative at t = {t}', iparas)
        return iparas


    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)

        iparas = self.calc_implicit_parameters(t)
        N4n = iparas['N4n']
        N2n = iparas['N2n']
        N2x2n = iparas['N2x2n']

        g2nb2n = iparas['g2nb2n']
        k2x2nb4n = iparas['k2x2nb4n']
        k4nb4n = iparas['k4nb4n']

        d2n=self.d2n
        d2x2n=self.d2x2n
        d4n = self.d4n

        k2nb4n = self.k2nb4n
        g2x2nb2n = self.g2x2nb2n
        k2nb2x2n = self.k2nb2x2n

        k2x2nb2x2n = (k2nb2x2n*k2x2nb4n)/k2nb4n
        k4nb2x2n = (k2nb2x2n*k4nb4n)/k2nb4n

        c2n = self.c2n
        c2x2n = self.c2x2n
        c4n = self.c4n
        # c14 equations 

        M_new[c2n] =(g2nb2n*self.Catm.lin(t + self.Dbirth)*N2n + 2*k2nb2x2n*M[c2x2n]*N2x2n + 2*k2nb2x2n*self.Catm.lin(t + self.Dbirth)*N2x2n + 2*k2nb4n*(M[c4n] + self.Catm.lin(t + self.Dbirth))*N4n - M[c2n]*(g2nb2n*N2n + 4*k2nb2x2n*N2x2n + 4*k2nb4n*N4n))/N2n
        M_new[c2x2n] =(g2x2nb2n*k2nb4n*M[c2n]*N2n + self.Catm.lin(t + self.Dbirth)*(g2x2nb2n*k2nb4n*N2n + 2*k2nb2x2n*k2x2nb4n*N2x2n) + 2*k2nb4n*k2x2nb4n*(M[c4n] + self.Catm.lin(t + self.Dbirth))*N4n - 2*M[c2x2n]*(g2x2nb2n*k2nb4n*N2n + k2nb2x2n*k2x2nb4n*N2x2n + 2*k2nb4n*k2x2nb4n*N4n))/(2.*k2nb4n*N2x2n)
        M_new[c4n] =k4nb4n*(-M[c4n] + self.Catm.lin(t + self.Dbirth) + (k2nb2x2n*(M[c2x2n] - 2*M[c4n] + self.Catm.lin(t + self.Dbirth))*N2x2n)/(k2nb4n*N4n))

        return M_new.ravel()

    def measurement_model(self, result_sim, data):
        mask_n2n4 = data.df.ploidy == '2n4n'
        mask_n2 = data.df.ploidy == '2n'
        mask_n4 = data.df.ploidy == '4n'

        mask_Cn2 = data.df.ploidy == 'C2n'
        mask_Cn4 = data.df.ploidy == 'C4n'
        

        wn2 = np.empty_like(data.age) * np.nan
        wn2x2 = np.empty_like(data.age) * np.nan
        wn4 = np.empty_like(data.age) * np.nan



        N_total = 1/(2-self.ploidy(data.age[mask_n2n4]))
        wn2[mask_n2n4] =  self.ploidy(data.age[mask_n2n4])*N_total
        wn2x2[mask_n2n4] = 2*self.ploidy2x2(data.age[mask_n2n4])*N_total
        wn4[mask_n2n4] = 2*(1-self.ploidy2x2(data.age[mask_n2n4])-self.ploidy(data.age[mask_n2n4]))*N_total
        
        tmp = 1/ (self.ploidy(data.age[mask_n2]) + 2*self.ploidy2x2(data.age[mask_n2]))
        wn2[mask_n2] = self.ploidy(data.age[mask_n2]) * tmp
        wn2x2[mask_n2] = 2*self.ploidy2x2(data.age[mask_n2]) * tmp
        wn4[mask_n2] =  0
       
        wn2[mask_n4] = 0
        wn2x2[mask_n4] = 0 
        wn4[mask_n4] = 1

        wn2[mask_Cn2] = 1
        wn2x2[mask_Cn2] = 0 
        wn4[mask_Cn2] = 0

        tmp = 1/ (1-self.ploidy(data.age[mask_Cn4])) 
        wn2[mask_Cn4] = 0
        wn2x2[mask_Cn4] =  self.ploidy2x2(data.age[mask_Cn4]) * tmp
        wn4[mask_Cn4] = (1-self.ploidy2x2(data.age[mask_Cn4])-self.ploidy(data.age[mask_Cn4])) * tmp
        
        average = (result_sim['c2n']*wn2 + result_sim['c2x2n']*wn2x2  + result_sim['c4n']*wn4     )

        return average






models_list = [POP1,Rm,POP2p_stem,POP3p_2nb_int,POP4p_conv,POP3p_2x2n_NP,POP2p,POP3p_4nq_new]

