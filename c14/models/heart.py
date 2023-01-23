import numpy as np
import pandas as pd
from .base import model_base, Catm
from ..tools import ImplicitParametersOutOfRange,FailedToCalulatePopulation,NonFiniteValuesinIntegrate,listofdict_to_dictoflist,listofdict_to_dictofarray




import logging
logger =  logging.getLogger(__name__)

global_error = 0.5
global_limit = (1e-6, 1)
N_MIN =0
TH_NULL = 1e-4

N4_0 =  0.04
N8_0 = 0.005

PMONOS = 0.8
PMONOSD = 0.65
PMONO_0 = 0.8
DEATH_H = 0.00548#0.00473
DEATH_N = 0.000898#0.00555#0.00161



def conv_h(data):
    data['age'] = data['Dcoll'] - data['Dbirth']
    data['d14C'] =data['d14c+']
    data['age20'] = 20
    data['measurment_types'] = data['14C_Type']
    data.set_index('index',inplace=True)
    return data
    
def conv_d(data):
    data['age'] = data['Dcoll'] - data['Dbirth']
    data['agediag'] = data['Ddiag'] - data['Dbirth']
    data['d14C'] =data['d14c+']
    data['age20'] = 20
    data['agediag_interpolated'] = np.isnan(data['agediag'])
    data['agediag'] = np.where(np.isnan(data['agediag']),data['age']*0.85, data['agediag'] )
    data['agediag_m10'] = np.where( (data['age']-data['agediag'])<10 ,data['age']-10, data['agediag'] )
    data['agediag_p5'] = data['agediag'] - 5
    data['measurment_types'] = data['14C_Type']
    try:
        data['agelvad'] = data['age'] - data['dlvad']
    except KeyError:
        pass
    data.set_index('index',inplace=True)

    return data
    


class POP8P(model_base):
    populations_DNA = {'N2n':1, 'N4n':2, 'N8n':4, 'N16n':8, 'N2x2n':2, 'N2x4n':4, 'N2x8n':8, 'N2x16n':16}
    plot_types = ['mean','2n','4n','pn']
    
    populations_2n = ['N2n', 'N2x2n']
    populations_4n = ['N4n', 'N2x4n']
    populations_pn = ['N4n', 'N8n', 'N16n', 'N2x4n', 'N2x8n', 'N2x16n']
    populations = ['N2n','N4n','N8n','N16n','N2x2n','N2x4n','N2x8n','N2x16n']
    populations_plot = {'mean':populations,'2n':populations_2n,'4n':populations_4n,'pn':populations_pn}
    iparas = ['b2n', 'k2n4n', 'k4n8n', 'k8n16n', 'k2n2x2n', 'k2x2n2x4n', 'k2x4n2x8n', 'k2x8n2x16n']
    flow_in =  {'N2n':[('b2n','N2n',2)],'N4n':[('k2n4n','N2n',1)],'N8n':[('k4n8n','N4n',1)],'N16n':[('k8n16n','N8n',1)],
           'N2x2n':[('k2n2x2n','N2n',1)],'N2x4n':[('k2x2n2x4n','N2x2n',1)],'N2x8n':[('k2x4n2x8n','N2x4n',1)],'N2x16n':[('k2x8n2x16n','N2x8n',1)]}
    def __init__(self,p2nS,p4nS,p8nS,age0,age,state='h',skip_failed = False):
        self.skip_failed = skip_failed
        self.p2nS = np.array(p2nS)
        self.p4nS = np.array(p4nS)
        self.p8nS = np.where(1-self.p2nS-self.p4nS-np.array(p8nS)<1e-10,
                             1-self.p2nS-self.p4nS,
                             np.array(p8nS))

        
        
        self.n4_0 = np.where(self.p4nS > 0.04, np.ones_like(self.p2nS)*0.04,   self.p4nS * 0.01  ) 
        self.n8_0 = np.where(self.p8nS > 0.005, np.ones_like(self.p2nS)*0.005, self.p8nS * 0.01  ) 
        self.n2_0 = 1 - self.n4_0 - self.n8_0
        
         
        self.age = np.array(age)
        self.age0 = age0
        
        if state == 'h':
            self.r2x4nS = 0.45
            self.r2x8nS = 0.03
            self.r2x16nS = 0.00
            self.pMonoS = 0.80
            self.pMono_0 = 1 - 0.01*(1-self.pMonoS)
            #r2x8nS_threashold  = (-1.0101010101010102*self.n8_0 + 1.0101010101010102*self.p8nS)/(1. + 1.*self.pMono_0 - 2.*self.pMonoS)
            #self.r2x8nS = np.where( r2x8nS_threashold > 0.2, 0.2*np.ones_like(self.p8nS),r2x8nS_threashold )#0.2
            #r2x2nS_th = self.p2nS*(1 -self.pMonoS) / (1 - self.pMonoS )
            #self.r2x16nS = (1  - self.p2nS - self.p4nS - self.p8nS)*(1 - self.pMonoS) / (1 - self.pMonoS)
            #r8 = 1 - self.r2x4nS - self.r2x16nS - r2x2nS_th
            #self.r2x8nS = r8
        elif state == 'd':
            self.r2x4nS  = 0.46
            self.r2x8nS  = 0.37
            self.r2x16nS = 0.05
            self.pMonoS = 0.69
            self.pMono_0 = 1 - 0.01*(1-self.pMonoS)
            
            #r2x2nS_th = self.p2nS*(1 -self.pMonoS) / (1 - self.pMonoS )
            #self.r2x4nS = 0.2
            #self.r2x16nS = (1  - self.p2nS - self.p4nS - self.p8nS)*(1 - self.pMonoS) / (1 - self.pMonoS)
            #r8 = 1 - self.r2x4nS - self.r2x16nS - r2x2nS_th
            #self.r2x8nS = r8
        else:
            raise NotImplementedError
            
        self.r2x4n_0 =  0.01*self.r2x4nS
        self.r2x8n_0 =  0.01*self.r2x8nS
        self.r2x16n_0 = 0.01*self.r2x16nS


        
        

        default_parameters = {'d':0.01,'sigma':0.1}
        self.logparas = ['d']
        self.linparas = ['sigma']
        limit = {i: global_limit for i in default_parameters.keys()}
        limit['sigma'] = (0,0.5)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['c2n','c4n','c8n','c16n','c2x2n','c2x4n','c2x8n','c2x16n'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)
   

    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)
        
        iparas = self.calc_implicit_parameters(t)
        
        d = self.d
        c2n = self.c2n
        c4n = self.c4n
        c8n = self.c8n
        c16n = self.c16n
        c2x2n = self.c2x2n
        c2x4n = self.c2x4n
        c2x8n = self.c2x8n
        c2x16n = self.c2x16n
        
        b2n = iparas['b2n']
        k2n4n = iparas['k2n4n']
        k4n8n = iparas['k4n8n']
        k8n16n = iparas['k8n16n']
        k2n2x2n = iparas['k2n2x2n']
        k2x2n2x4n = iparas['k2x2n2x4n']
        k2x4n2x8n = iparas['k2x4n2x8n']
        k2x8n2x16n = iparas['k2x8n2x16n']

        N2n = iparas['N2n']
        N4n = iparas['N4n']
        N8n = iparas['N8n']
        N16n = iparas['N16n']
        N2x2n = iparas['N2x2n']
        N2x4n = iparas['N2x4n']
        N2x8n = iparas['N2x8n']
        N2x16n = iparas['N2x16n']

        Catm_t = self.Catm.lin(t + self.Dbirth)
        M_new[c2n] = b2n*(-M[c2n] + Catm_t)
        M_new[c4n] = (k2n4n*(M[c2n] - 2*M[c4n] +Catm_t)*N2n)/(2.*N4n)
        M_new[c8n] = (k4n8n*(M[c4n] - 2*M[c8n] + Catm_t)*N4n)/(2.*N8n)
        tmp = (k8n16n*(-2*M[c16n] + M[c8n] + Catm_t)*N8n)/(2.*N16n)
        M_new[c16n] = np.where(N16n!=TH_NULL,tmp,M_new[c8n])
        M_new[c2x2n] = (k2n2x2n*(M[c2n] - 2*M[c2x2n] + Catm_t)*N2n)/(2.*N2x2n)
        M_new[c2x4n] = (k2x2n2x4n*(M[c2x2n] - 2*M[c2x4n] + Catm_t)*N2x2n)/(2.*N2x4n)
        M_new[c2x8n] = (k2x4n2x8n*(M[c2x4n] - 2*M[c2x8n] + Catm_t)*N2x4n)/(2.*N2x8n)
        tmp2 = (k2x8n2x16n*(-2*M[c2x16n] + M[c2x8n] + Catm_t)*N2x8n)/(2.*N2x16n)
        M_new[c2x16n] = np.where(N2x16n!=TH_NULL,tmp2,M_new[c2x8n])   

        #mask_age = t<=(self.age+1e-5)
        #M_new = M_new*mask_age
        #if self.skip_failed:
        #    self.mask = np.logical_and(self.mask,iparas['mask'])
            
        return M_new.ravel()

    def calc_initial_parameters(self):
        self.mask = np.ones_like(self.age, dtype=bool)
    
    def calc_implicit_parameters(self, t):
        
        d = self.d

        Ntotal = 1
        dtNtotal = 0

        pMono,dtpMono = self.calc_pMono(t)        
        p2n,dtp2n = self.calc_p2n(t)
        p4n,dtp4n = self.calc_p4n(t)
        p8n,dtp8n = self.calc_p8n(t)

        
        r2x4n,dtr2x4n = self.calc_r2x4n(t)
        r2x8n,dtr2x8n = self.calc_r2x8n(t)
        r2x16n,dtr2x16n = self.calc_r2x16n(t)


        N2n = Ntotal*(p2n - (-1 + pMono)*(-1 + r2x16n + r2x4n + r2x8n))
        N4n = Ntotal*(p4n + (-1 + pMono)*r2x4n)
        N8n = Ntotal*(p8n + (-1 + pMono)*r2x8n)
        N16nb = -(Ntotal*(-1 + p2n + p4n + p8n + r2x16n - pMono*r2x16n))
        N2x2n = np.ones_like(p2n)*Ntotal*(-1 + pMono)*(-1 + r2x16n + r2x4n + r2x8n)
        N2x4n = np.ones_like(p2n)*-(Ntotal*(-1 + pMono)*r2x4n)
        N2x8n = np.ones_like(p2n)*-(Ntotal*(-1 + pMono)*r2x8n)
        N2x16nb = np.ones_like(p2n)*-(Ntotal*(-1 + pMono)*r2x16n)
        
        N16n   = np.where(N16nb  <TH_NULL,TH_NULL*np.ones_like(N16nb),  N16nb)
        N2x16n = np.where(N2x16nb<TH_NULL,TH_NULL*np.ones_like(N2x16nb),N2x16nb)
        

        dtN2n = (p2n - (-1 + pMono)*(-1 + r2x16n + r2x4n + r2x8n))*dtNtotal + Ntotal*(dtp2n - (-1 + r2x16n + r2x4n + r2x8n)*dtpMono - (-1 + pMono)*(dtr2x16n + dtr2x4n + dtr2x8n))
        dtN4n = p4n*dtNtotal + r2x4n*((-1 + pMono)*dtNtotal + Ntotal*dtpMono) + Ntotal*(dtp4n + (-1 + pMono)*dtr2x4n)
        dtN8n = p8n*dtNtotal + r2x8n*((-1 + pMono)*dtNtotal + Ntotal*dtpMono) + Ntotal*(dtp8n + (-1 + pMono)*dtr2x8n)
        dtN16n = -((-1 + p2n + p4n + p8n + r2x16n - pMono*r2x16n)*dtNtotal) - Ntotal*(dtp2n + dtp4n + dtp8n - r2x16n*dtpMono - (-1 + pMono)*dtr2x16n)
        dtN2x2n = (-1 + pMono)*(-1 + r2x16n + r2x4n + r2x8n)*dtNtotal + Ntotal*((-1 + r2x16n + r2x4n + r2x8n)*dtpMono + (-1 + pMono)*(dtr2x16n + dtr2x4n + dtr2x8n))
        dtN2x4n = r2x4n*(-((-1 + pMono)*dtNtotal) - Ntotal*dtpMono) - Ntotal*(-1 + pMono)*dtr2x4n
        dtN2x8n = r2x8n*(-((-1 + pMono)*dtNtotal) - Ntotal*dtpMono) - Ntotal*(-1 + pMono)*dtr2x8n
        dtN2x16n = r2x16n*(-((-1 + pMono)*dtNtotal) - Ntotal*dtpMono) - Ntotal*(-1 + pMono)*dtr2x16n

        
        
        b2n = (d*(N16n + N2n + N2x16n + N2x2n + N2x4n + N2x8n + N4n + N8n) + dtN16n + dtN2n + dtN2x16n + dtN2x2n + dtN2x4n + dtN2x8n + dtN4n + dtN8n)/N2n
        k2n4n = (d*(N16n + N4n + N8n) + dtN16n + dtN4n + dtN8n)/N2n
        k4n8n = (d*(N16n + N8n) + dtN16n + dtN8n)/N4n
        k8n16n = (d*N16n + dtN16n)/N8n
        k2n2x2n = (d*(N2x16n + N2x2n + N2x4n + N2x8n) + dtN2x16n + dtN2x2n + dtN2x4n + dtN2x8n)/N2n
        k2x2n2x4n = (d*(N2x16n + N2x4n + N2x8n) + dtN2x16n + dtN2x4n + dtN2x8n)/N2x2n
        k2x4n2x8n = (d*(N2x16n + N2x8n) + dtN2x16n + dtN2x8n)/N2x4n
        k2x8n2x16n = (d*N2x16n + dtN2x16n)/N2x8n


        
        k8n16n = np.where(N16n==TH_NULL,np.zeros_like(k8n16n),k8n16n)
        k2x8n2x16n = np.where(N2x16n==TH_NULL,np.zeros_like(k2x8n2x16n),k2x8n2x16n)


        iparas ={'b2n':b2n,'k2n4n': k2n4n,'k4n8n': k4n8n,'k8n16n': k8n16n,'k2n2x2n': k2n2x2n,'k2x2n2x4n': k2x2n2x4n,'k2x4n2x8n': k2x4n2x8n,'k2x8n2x16n': k2x8n2x16n,
           'N2n':N2n,'N4n':N4n,'N8n':N8n,'N16n':N16n,'N2x2n':N2x2n,'N2x4n':N2x4n,'N2x8n':N2x8n,'N2x16n':N2x16n }

        if ( np.logical_and(N2n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2n ({N2n})is negative at t = {t}', iparas)
        if ( np.logical_and(N4n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N4n ({N4n})is negative at t = {t}', iparas)
        if ( np.logical_and(N8n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N8n ({N8n})is negative at t = {t}', iparas)
        if ( np.logical_and(N16nb<-TH_NULL,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N16nb ({N16nb})is negative at t = {t}', iparas)
        if ( np.logical_and(N2x2n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2x2n ({N2x2n})is negative at t = {t}', iparas)
        if ( np.logical_and(N2x4n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2x4n ({N2x4n})is negative at t = {t}', iparas)
        if ( np.logical_and(N2x8n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2x8n ({N2x8n})is negative at t = {t}', iparas)
        if ( np.logical_and(N2x16nb<-TH_NULL,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2x16nb ({N2x16nb})is negative at t = {t}', iparas)

            
        if ( np.logical_and(b2n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'b2n ({b2n})is negative at t = {t}', iparas)
        if (np.logical_and(k2n4n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2n4n ({k2n4n})is negative at t = {t}', iparas)
        if (np.logical_and(k4n8n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k4n8n ({k4n8n})is negative at t = {t}', iparas)
        if (np.logical_and(k8n16n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k8n16n ({k8n16n})is negative at t = {t}', iparas)    
        if (np.logical_and(k2n2x2n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2n2x2n ({k2n2x2n})is negative at t = {t}', iparas)
        if (np.logical_and(k2x2n2x4n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2x2n2x4n ({k2x2n2x4n})is negative at t = {t}', iparas)
        if (np.logical_and(k2x4n2x8n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2x4n2x8n ({k2x4n2x8n})is negative at t = {t}', iparas)
        if (np.logical_and(k2x8n2x16n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2x8n2x16n ({k2x8n2x16n})is negative at t = {t}', iparas)
        



        return iparas
    

    def calc_p2n(self,t):
        tt = np.ones_like(self.p2nS)*t
        p2n = np.where(tt<self.age0,self.n2_0-tt/self.age0*(self.n2_0-self.p2nS),self.p2nS)
        dtp2n = np.where(tt<self.age0,-1/self.age0*(self.n2_0-self.p2nS),0)
        return p2n,dtp2n

    def calc_p4n(self,t):
        tt = np.ones_like(self.p4nS)*t
        p4n = np.where(tt<self.age0,self.n4_0+tt/self.age0*(self.p4nS-self.n4_0),self.p4nS)
        dtp4n = np.where(tt<self.age0,1/self.age0*(self.p4nS-self.n4_0),0)
        return p4n,dtp4n

    def calc_p8n(self,t):
        tt = np.ones_like(self.p8nS)*t
        p8n = np.where(tt<self.age0,self.n8_0+tt/self.age0*(self.p8nS-self.n8_0),self.p8nS)
        dtp8n = np.where(tt<self.age0,1/self.age0*(self.p8nS-self.n8_0),0)
        return p8n,dtp8n

    
    def calc_pMono(self,t):
        tt = np.ones_like(self.p8nS)*t
        pMono = np.where( tt<self.age0,
                        self.pMono_0 - t/self.age0*(self.pMono_0 - self.pMonoS),
                        self.pMonoS)
        dtpMono = np.where( tt<self.age0,
                         -1/self.age0*(self.pMono_0 - self.pMonoS),
                           0)
        return pMono,dtpMono
        
    def calc_r2x4n(self,t):
        tt = np.ones_like(self.p8nS)*t
        r2x4n = np.where( tt<self.age0,
                         self.r2x4n_0 + t/self.age0*(self.r2x4nS - self.r2x4n_0),
                         self.r2x4nS)
        dtr2x4n= np.where( tt<self.age0,
                          1/self.age0*(self.r2x4nS - self.r2x4n_0 ),
                          0)
        return r2x4n,dtr2x4n
                         
        
    def calc_r2x8n(self,t):
        tt = np.ones_like(self.p8nS)*t
        r2x8n =  np.where( tt<self.age0,
                      self.r2x8n_0 + t/self.age0*(self.r2x8nS - self.r2x8n_0 ),
                      self.r2x8nS)
        dtr2x8n=  np.where( tt<self.age0,
                      1/self.age0*(self.r2x8nS - self.r2x8n_0 ),
                      0)
        return r2x8n,dtr2x8n
                          
    def calc_r2x16n(self,t):
        tt = np.ones_like(self.r2x16nS)*t
        r2x16n = np.where(tt<self.age0,self.r2x16n_0-tt/self.age0*(self.r2x16n_0-self.r2x16nS),self.r2x16nS)
        dtr2x16n = np.where(tt<self.age0,-1/self.age0*(self.r2x16n_0-self.r2x16nS),0)
        return r2x16n,dtr2x16n
                          
        
        
    def measurement_model(self, result_sim, data):
        if '14C_Type' not in data.df.columns:
            P = listofdict_to_dictofarray([self.calc_implicit_parameters(a) for a in data.age])
            res = (np.diag(P['N2n'])*result_sim['c2n']+2*np.diag(P['N4n'])*result_sim['c4n'] \
            + 4* np.diag(P['N8n'])*result_sim['c8n']+ 8* np.diag(P['N16n'])*result_sim['c16n'] \
            +2*np.diag(P['N2x2n'])*result_sim['c2x2n']+4 * np.diag(P['N2x4n'])*result_sim['c2x4n'] \
            + 8 * np.diag(P['N2x8n'])*result_sim['c2x8n']+ 16 * np.diag(P['N2x16n'])*result_sim['c2x16n']) \
            /(np.diag(P['N2n'])+2*np.diag(P['N4n'])+4*np.diag(P['N8n'])+ 8* np.diag(P['N16n']) \
            +2*np.diag(P['N2x2n'])+4*np.diag(P['N2x4n'])+8*np.diag(P['N2x8n'])+16*np.diag(P['N2x16n'])  )
            return res
        result = []
        for i,(ind,row) in enumerate(data.df.iterrows()):
            result_sim_i = result_sim.loc[ind]
            P = self.calc_implicit_parameters(row['age']) 
            IP = {n:v[i] for n,v in P.items()}
            if row['14C_Type']=='mean':
                res = (IP['N2n']*result_sim_i['c2n']+2*IP['N4n']*result_sim_i['c4n'] \
            + 4* IP['N8n']*result_sim_i['c8n']+ 8* IP['N16n']*result_sim_i['c16n'] \
            +2*IP['N2x2n']*result_sim_i['c2x2n']+4 * IP['N2x4n']*result_sim_i['c2x4n'] \
            + 8 * IP['N2x8n']*result_sim_i['c2x8n']+ 16 * IP['N2x16n']*result_sim_i['c2x16n']) \
            /(IP['N2n']+2*IP['N4n']+4*IP['N8n']+ 8* IP['N16n'] \
            +2*IP['N2x2n']+4*IP['N2x4n']+8*IP['N2x8n']+16*IP['N2x16n']  )
                result.append( res )
            elif row['14C_Type']=='2n':
                res = (IP['N2n']*result_sim_i['c2n'] + 2*IP['N2x2n']*result_sim_i['c2x2n']) \
                    / ( IP['N2n'] + 2*IP['N2x2n'])
                result.append( res )
            elif row['14C_Type']=='4n':
                res = (IP['N4n']*result_sim_i['c4n'] + 2*IP['N2x4n']*result_sim_i['c2x4n']) \
                    / ( IP['N4n'] + 2*IP['N2x4n'])
                result.append( res )
            elif row['14C_Type']=='pn':
                res = (2*IP['N4n']*result_sim_i['c4n'] \
            + 4* IP['N8n']*result_sim_i['c8n']+ 8* IP['N16n']*result_sim_i['c16n'] \
            + 4 * IP['N2x4n']*result_sim_i['c2x4n'] \
            + 8 * IP['N2x8n']*result_sim_i['c2x8n']+ 16 * IP['N2x16n']*result_sim_i['c2x16n']) \
            /(2*IP['N4n']+4*IP['N8n']+ 8* IP['N16n'] \
            +4*IP['N2x4n']+8*IP['N2x8n']+16*IP['N2x16n']  )
                result.append( res )
            else:
                print('type unkown',row['14C_Type'])
        return pd.Series(result,index=result_sim.index,dtype=float)



class POP8P_h_exp(POP8P):
    def __init__(self,data,age0):
        self.p2nS = np.array(data['2n'])
        self.p4nS = np.array(data['4n'])
        self.p8nS = np.where(1-self.p2nS-self.p4nS-np.array(data['8n'])<1e-10,
                             1-self.p2nS-self.p4nS,
                             np.array(data['8n']))
        self.age = np.array(data['age'])
        
        self.pMonoS = 0.8
    

        self.n4_0 = 0.04
        if (self.p4nS < 0.04).any():
            print ('error in n4 at t=0')
        self.n8_0 = 0.005
        if (self.p8nS < 0.005).any():
            print ('error in n8 at t=0')
        self.n2_0 = 1 - self.n4_0 - self.n8_0
        
        
        
        
        


        self.t0_2n = 11.632384709730932
        self.Ni_2n = ( self.p2nS - self.n2_0*np.exp(-self.age/self.t0_2n) ) / ( 1-np.exp(-self.age/self.t0_2n) )
        
        self.t0_4n = self.t0_2n#10.474682311920247
        self.Ni_4n =  ( self.p4nS - self.n4_0*np.exp(-self.age/self.t0_4n) ) / ( 1-np.exp(-self.age/self.t0_4n) )
        
        self.t0_8n = self.t0_2n#19.501536691850745
        self.Ni_8n =  ( self.p8nS - self.n8_0*np.exp(-self.age/self.t0_8n) ) / ( 1-np.exp(-self.age/self.t0_8n) )
        
        

        default_parameters = {'d':0.01,'sigma':0.1}
        self.logparas = ['d']
        self.linparas = ['sigma']
        limit = {i: global_limit for i in default_parameters.keys()}
        limit['sigma'] = (0,0.5)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['c2n','c4n','c8n','c16n','c2x2n','c2x4n','c2x8n','c2x16n'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)
        
    def calc_p2n(self,t):
        p2n =    (self.n2_0-self.Ni_2n)*np.exp(-t/self.t0_2n)+self.Ni_2n
        dtp2n = -1*(self.n2_0-self.Ni_2n)*np.exp(-t/self.t0_2n)/self.t0_2n
        return p2n,dtp2n

    def calc_p4n(self,t):
        p4n =    (self.n4_0-self.Ni_4n)*np.exp(-t/self.t0_4n)+self.Ni_4n
        dtp4n = -1*(self.n4_0-self.Ni_4n)*np.exp(-t/self.t0_4n)/self.t0_4n
             
        return p4n,dtp4n

    def calc_p8n(self,t):
        p8n =    (self.n8_0-self.Ni_8n)*np.exp(-t/self.t0_8n)+self.Ni_8n
        dtp8n = -1*(self.n8_0-self.Ni_8n)*np.exp(-t/self.t0_8n)/self.t0_8n
        
        return p8n,dtp8n
    
    
    def calc_pMono(self,t):
        
        pMono = self.pMonoS
        dtpMono = 0
        
        return pMono,dtpMono    
    
    
            
    
    def calc_implicit_parameters(self, t):
        
        d = self.d

        Ntotal = 1
        dtNtotal = 0

        pMono,dtpMono = self.calc_pMono(t)
        p2n,dtp2n = self.calc_p2n(t)
        p4n,dtp4n = self.calc_p4n(t)
        p8n,dtp8n = self.calc_p8n(t)

        N2n = Ntotal*p2n*pMono
        N4n = Ntotal*p4n*pMono
        N8n = Ntotal*p8n*pMono
        N16nb = -(Ntotal*(-1 + p2n + p4n + p8n)*pMono)
        N2x2n = -(Ntotal*p2n*(-1 + pMono))
        N2x4n = -(Ntotal*p4n*(-1 + pMono))
        N2x8n = -(Ntotal*p8n*(-1 + pMono))
        N2x16nb = Ntotal*(-1 + p2n + p4n + p8n)*(-1 + pMono)
        
        N16n   = np.where(N16nb  <TH_NULL,TH_NULL*np.ones_like(N16nb),  N16nb)
        N2x16n = np.where(N2x16nb<TH_NULL,TH_NULL*np.ones_like(N2x16nb),N2x16nb)
        

        dtN2n = Ntotal*pMono*dtp2n + p2n*(pMono*dtNtotal + Ntotal*dtpMono)
        dtN4n = Ntotal*pMono*dtp4n + p4n*(pMono*dtNtotal + Ntotal*dtpMono)
        dtN8n = Ntotal*pMono*dtp8n + p8n*(pMono*dtNtotal + Ntotal*dtpMono)
        dtN16n = -(pMono*((-1 + p2n + p4n + p8n)*dtNtotal + Ntotal*(dtp2n + dtp4n + dtp8n))) - Ntotal*(-1 + p2n + p4n + p8n)*dtpMono
        dtN2x2n = -((-1 + pMono)*(p2n*dtNtotal + Ntotal*dtp2n)) - Ntotal*p2n*dtpMono
        dtN2x4n = -((-1 + pMono)*(p4n*dtNtotal + Ntotal*dtp4n)) - Ntotal*p4n*dtpMono
        dtN2x8n = -((-1 + pMono)*(p8n*dtNtotal + Ntotal*dtp8n)) - Ntotal*p8n*dtpMono
        dtN2x16n = (-1 + p2n + p4n + p8n)*(-1 + pMono)*dtNtotal + Ntotal*((-1 + pMono)*(dtp2n + dtp4n + dtp8n) + (-1 + p2n + p4n + p8n)*dtpMono)

        
        
        b2n = (d*(N16n + N2n + N2x16n + N2x2n + N2x4n + N2x8n + N4n + N8n) + dtN16n + dtN2n + dtN2x16n + dtN2x2n + dtN2x4n + dtN2x8n + dtN4n + dtN8n)/N2n
        k2n4n = (d*(N16n + N4n + N8n) + dtN16n + dtN4n + dtN8n)/N2n
        k4n8n = (d*(N16n + N8n) + dtN16n + dtN8n)/N4n
        k8n16n = (d*N16n + dtN16n)/N8n
        k2n2x2n = (d*(N2x16n + N2x2n + N2x4n + N2x8n) + dtN2x16n + dtN2x2n + dtN2x4n + dtN2x8n)/N2n
        k2x2n2x4n = (d*(N2x16n + N2x4n + N2x8n) + dtN2x16n + dtN2x4n + dtN2x8n)/N2x2n
        k2x4n2x8n = (d*(N2x16n + N2x8n) + dtN2x16n + dtN2x8n)/N2x4n
        k2x8n2x16n = (d*N2x16n + dtN2x16n)/N2x8n


        
        
        k8n16n = np.where(N16n==TH_NULL,np.zeros_like(k8n16n),k8n16n)
        k2x8n2x16n = np.where(N2x16n==TH_NULL,np.zeros_like(k2x8n2x16n),k2x8n2x16n)


        iparas ={'b2n':b2n,'k2n4n': k2n4n,'k4n8n': k4n8n,'k8n16n': k8n16n,'k2n2x2n': k2n2x2n,'k2x2n2x4n': k2x2n2x4n,'k2x4n2x8n': k2x4n2x8n,'k2x8n2x16n': k2x8n2x16n,
           'N2n':N2n,'N4n':N4n,'N8n':N8n,'N16n':N16n,'N2x2n':N2x2n,'N2x4n':N2x4n,'N2x8n':N2x8n,'N2x16n':N2x16n }

        if ( np.logical_and(N2n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2n ({N2n})is negative at t = {t}', iparas)
        if ( np.logical_and(N4n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N4n ({N4n})is negative at t = {t}', iparas)
        if ( np.logical_and(N8n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N8n ({N8n})is negative at t = {t}', iparas)
        if ( np.logical_and(N16nb<-1e10,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N16nb ({N16nb})is negative at t = {t}', iparas)
        if ( np.logical_and(N2x2n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2x2n ({N2x2n})is negative at t = {t}', iparas)
        if ( np.logical_and(N2x4n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2x4n ({N2x4n})is negative at t = {t}', iparas)
        if ( np.logical_and(N2x8n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2x8n ({N2x8n})is negative at t = {t}', iparas)
        if ( np.logical_and(N2x16nb<-1e10,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2x16nb ({N2x16nb})is negative at t = {t}', iparas)


        if ( np.logical_and(b2n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'b2n ({b2n})is negative at t = {t}', iparas)
        if (np.logical_and(k2n4n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2n4n ({k2n4n})is negative at t = {t}', iparas)
        if (np.logical_and(k4n8n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k4n8n ({k4n8n})is negative at t = {t}', iparas)
        if (np.logical_and(k8n16n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k8n16n ({k8n16n})is negative at t = {t}', iparas)    
        if (np.logical_and(k2n2x2n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2n2x2n ({k2n2x2n})is negative at t = {t}', iparas)
        if (np.logical_and(k2x2n2x4n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2x2n2x4n ({k2x2n2x4n})is negative at t = {t}', iparas)
        if (np.logical_and(k2x4n2x8n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2x4n2x8n ({k2x4n2x8n})is negative at t = {t}', iparas)
        if (np.logical_and(k2x8n2x16n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2x8n2x16n ({k2x8n2x16n})is negative at t = {t}', iparas)




        return iparas          
    



class POP1damage(model_base):
    populations_DNA = {'cells1':1 }
    plot_types = ['mean']
    populations = ['cells1']
    populations_plot = {'mean':populations}
    iparas = ['b2n']
    flow_in =  {'cells1':[('b2n','cells1',2)]}

    def __init__(self,data, damage=0.000313, d=0.1,delay=1):

        self.p2nS = np.array(data['2n'])
        self.p4nS = np.array(data['4n'])
        self.p8nS = np.where(1-self.p2nS-self.p4nS-np.array(data['8n'])<1e-10,
                             1-self.p2nS-self.p4nS,
                             np.array(data['8n']))
        self.age = np.array(data['age'])
        
        self.pMonoS = 0.8

        self.n4_0 = 0.04
        if (self.p4nS < 0.04).any():
            print ('error in n4 at t=0')
        self.n8_0 = 0.005
        if (self.p8nS < 0.005).any():
            print ('error in n8 at t=0')
        self.n2_0 = 1 - self.n4_0 - self.n8_0

        self.t0_2n = 11.632384709730932
        self.Ni_2n = ( self.p2nS - self.n2_0*np.exp(-self.age/self.t0_2n) ) / ( 1-np.exp(-self.age/self.t0_2n) )
        
        self.t0_4n = self.t0_2n#10.474682311920247
        self.Ni_4n =  ( self.p4nS - self.n4_0*np.exp(-self.age/self.t0_4n) ) / ( 1-np.exp(-self.age/self.t0_4n) )
        
        self.t0_8n = self.t0_2n#19.501536691850745
        self.Ni_8n =  ( self.p8nS - self.n8_0*np.exp(-self.age/self.t0_8n) ) / ( 1-np.exp(-self.age/self.t0_8n) )
        
        

 

        default_parameters = dict(d=d,sigma=0.02)
        limit = {i: global_limit for i in default_parameters.keys()}
        limit['sigma'] = (0,0.5)
        self.logparas = ['d']
        self.linparas = ['sigma']
        self.damage = damage  #= dna - A*dna  dna damage rate; A amount of damage or
        self.Catm = Catm(delay=delay)
        model_base.__init__(self, var_names=['cells1'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)


    def calc_p2n(self,t):
        p2n =    (self.n2_0-self.Ni_2n)*np.exp(-t/self.t0_2n)+self.Ni_2n
        dtp2n = -1*(self.n2_0-self.Ni_2n)*np.exp(-t/self.t0_2n)/self.t0_2n
        return p2n,dtp2n

    def calc_p4n(self,t):
        p4n =    (self.n4_0-self.Ni_4n)*np.exp(-t/self.t0_4n)+self.Ni_4n
        dtp4n = -1*(self.n4_0-self.Ni_4n)*np.exp(-t/self.t0_4n)/self.t0_4n
             
        return p4n,dtp4n

    def calc_p8n(self,t):
        p8n =    (self.n8_0-self.Ni_8n)*np.exp(-t/self.t0_8n)+self.Ni_8n
        dtp8n = -1*(self.n8_0-self.Ni_8n)*np.exp(-t/self.t0_8n)/self.t0_8n
        
        return p8n,dtp8n
    
    
    def calc_pMono(self,t):
        
        pMono = self.pMonoS
        dtpMono = 0
        
        return pMono,dtpMono    
    
    
            
    
    def dnatotal(self,t):
        Ntotal = 1
        dtNtotal = 0

        pMono,dtpMono = self.calc_pMono(t)
        p2n,dtp2n = self.calc_p2n(t)
        p4n,dtp4n = self.calc_p4n(t)
        p8n,dtp8n = self.calc_p8n(t)

        N2n = Ntotal*p2n*pMono
        N4n = Ntotal*p4n*pMono
        N8n = Ntotal*p8n*pMono
        N16nb = -(Ntotal*(-1 + p2n + p4n + p8n)*pMono)
        N2x2n = -(Ntotal*p2n*(-1 + pMono))
        N2x4n = -(Ntotal*p4n*(-1 + pMono))
        N2x8n = -(Ntotal*p8n*(-1 + pMono))
        N2x16nb = Ntotal*(-1 + p2n + p4n + p8n)*(-1 + pMono)
        
        N16n   = np.where(N16nb  <TH_NULL,TH_NULL*np.ones_like(N16nb),  N16nb)
        N2x16n = np.where(N2x16nb<TH_NULL,TH_NULL*np.ones_like(N2x16nb),N2x16nb)
        

        dtN2n = Ntotal*pMono*dtp2n + p2n*(pMono*dtNtotal + Ntotal*dtpMono)
        dtN4n = Ntotal*pMono*dtp4n + p4n*(pMono*dtNtotal + Ntotal*dtpMono)
        dtN8n = Ntotal*pMono*dtp8n + p8n*(pMono*dtNtotal + Ntotal*dtpMono)
        dtN16n = -(pMono*((-1 + p2n + p4n + p8n)*dtNtotal + Ntotal*(dtp2n + dtp4n + dtp8n))) - Ntotal*(-1 + p2n + p4n + p8n)*dtpMono
        dtN2x2n = -((-1 + pMono)*(p2n*dtNtotal + Ntotal*dtp2n)) - Ntotal*p2n*dtpMono
        dtN2x4n = -((-1 + pMono)*(p4n*dtNtotal + Ntotal*dtp4n)) - Ntotal*p4n*dtpMono
        dtN2x8n = -((-1 + pMono)*(p8n*dtNtotal + Ntotal*dtp8n)) - Ntotal*p8n*dtpMono
        dtN2x16n = (-1 + p2n + p4n + p8n)*(-1 + pMono)*dtNtotal + Ntotal*((-1 + pMono)*(dtp2n + dtp4n + dtp8n) + (-1 + p2n + p4n + p8n)*dtpMono)
        return N2n+2*N4n+4*N8n+8*N16n+2*N2x2n+4*N2x4n+8*N2x8n+16*N2x16n,dtN2n+2*dtN4n+4*dtN8n+8*dtN16n+2*dtN2x2n+4*dtN2x4n+8*dtN2x8n+16*dtN2x16n


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
        if (b2n<0).any():
            raise ImplicitParametersOutOfRange(
                f'b2n ({b2n}) < 0 at t={t})', iparas)
        return iparas

    def measurement_model(self, result_sim, data):
        return result_sim['cells1']





 
class POP8P_h_exp_exp(POP8P_h_exp):


    def calc_implicit_parameters(self, t):
        self.d = (self.d0-self.dinf) * np.exp(-t*self.dtau) + self.dinf
        return POP8P_h_exp.calc_implicit_parameters(self, t)

    def __init__(self,data,age0):
        self.p2nS = np.array(data['2n'])
        self.p4nS = np.array(data['4n'])
        self.p8nS = np.where(1-self.p2nS-self.p4nS-np.array(data['8n'])<1e-10,
                             1-self.p2nS-self.p4nS,
                             np.array(data['8n']))
        self.age = np.array(data['age'])
        
        self.pMonoS = 0.8

        self.n4_0 = 0.04
        if (self.p4nS < 0.04).any():
            print ('error in n4 at t=0')
        self.n8_0 = 0.005
        if (self.p8nS < 0.005).any():
            print ('error in n8 at t=0')
        self.n2_0 = 1 - self.n4_0 - self.n8_0
        
        self.t0_2n = 11.632384709730932
        self.Ni_2n = ( self.p2nS - self.n2_0*np.exp(-self.age/self.t0_2n) ) / ( 1-np.exp(-self.age/self.t0_2n) )
        
        self.t0_4n = self.t0_2n#10.474682311920247
        self.Ni_4n =  ( self.p4nS - self.n4_0*np.exp(-self.age/self.t0_4n) ) / ( 1-np.exp(-self.age/self.t0_4n) )
        
        self.t0_8n = self.t0_2n#19.501536691850745
        self.Ni_8n =  ( self.p8nS - self.n8_0*np.exp(-self.age/self.t0_8n) ) / ( 1-np.exp(-self.age/self.t0_8n) )
        
        

        default_parameters = {'d0':0.01,'dinf':0.0001,'dtau':0.1,'sigma':0.1}
        self.logparas = ['d0','dinf','dtau']
        self.linparas = ['sigma']
        limit = {i: global_limit for i in default_parameters.keys()}
        limit['sigma'] = (0,0.5)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['c2n','c4n','c8n','c16n','c2x2n','c2x4n','c2x8n','c2x16n'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)

 

class POP8P_d_exp(POP8P_h_exp):
    def __init__(self,data,age0):
        self.p2nSD = np.array(data['2n'])
        self.p4nSD = np.array(data['4n'])
        self.p8nSD = np.where(1-self.p2nSD-self.p4nSD-np.array(data['8n'])<1e-10,
                             1-self.p2nSD-self.p4nSD,
                             np.array(data['8n']))
        self.age = np.array(data['age'])
        self.age_d = np.array(data[age0])

        self.p2nS = 0.33
        self.p4nS = 0.6
        self.p8nS = 1 - self.p2nS - self.p4nS
        
        self.n4_0 = N4_0
        self.n8_0 = N8_0 
        self.n2_0 = 1 - self.n4_0 - self.n8_0

        
        self.pMonoS = PMONOS 
        self.pMonoSD = PMONOSD
        self.pMono_0 = PMONO_0
        
              
        
    
        self.t0_2n = 11.632384709730932
        self.Ni_2n = ( self.p2nS - self.n2_0*np.exp(-self.age_d/self.t0_2n) ) / ( 1-np.exp(-self.age_d/self.t0_2n) )
        
        self.t0_4n = self.t0_2n#10.474682311920247
        self.Ni_4n =  ( self.p4nS - self.n4_0*np.exp(-self.age_d/self.t0_4n) ) / ( 1-np.exp(-self.age_d/self.t0_4n) )
        
        self.t0_8n = self.t0_2n#19.501536691850745
        self.Ni_8n =  ( self.p8nS - self.n8_0*np.exp(-self.age_d/self.t0_8n) ) / ( 1-np.exp(-self.age_d/self.t0_8n) )
        
        

        default_parameters = {'d':0.01,'sigma':0.1}
        self.logparas = ['d']
        self.linparas = ['sigma']
        limit = {i: global_limit for i in default_parameters.keys()}
        limit['sigma'] = (0,0.5)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['c2n','c4n','c8n','c16n','c2x2n','c2x4n','c2x8n','c2x16n'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)
        
    def calc_p2n(self,t):
        p2n =    (self.n2_0-self.Ni_2n)*np.exp(-t/self.t0_2n)+self.Ni_2n
        p2n = np.where(t<self.age_d,p2n, self.p2nS - (t-self.age_d)/(self.age - self.age_d) * (self.p2nS - self.p2nSD) )
        
        dtp2n = -1*(self.n2_0-self.Ni_2n)*np.exp(-t/self.t0_2n)/self.t0_2n
        dtp2n = np.where(t<self.age_d,dtp2n, -1/(self.age - self.age_d) * (self.p2nS - self.p2nSD) )
        
        return p2n,dtp2n

    def calc_p4n(self,t):
        p4n =    (self.n4_0-self.Ni_4n)*np.exp(-t/self.t0_4n)+self.Ni_4n
        p4n = np.where(t<self.age_d,p4n, self.p4nS - (t-self.age_d)/(self.age - self.age_d) * (self.p4nS - self.p4nSD) )
        dtp4n = -1*(self.n4_0-self.Ni_4n)*np.exp(-t/self.t0_4n)/self.t0_4n
        dtp4n = np.where(t<self.age_d,dtp4n, -1/(self.age - self.age_d) * (self.p4nS - self.p4nSD) )
             
        return p4n,dtp4n

    def calc_p8n(self,t):
        p8n =    (self.n8_0-self.Ni_8n)*np.exp(-t/self.t0_8n)+self.Ni_8n
        p8n = np.where(t<self.age_d,p8n, self.p8nS - (t-self.age_d)/(self.age - self.age_d) * (self.p8nS - self.p8nSD) )
        dtp8n = -1*(self.n8_0-self.Ni_8n)*np.exp(-t/self.t0_8n)/self.t0_8n
        dtp8n = np.where(t<self.age_d,dtp8n, -1/(self.age - self.age_d) * (self.p8nS - self.p8nSD) )
        
        return p8n,dtp8n
    
    def calc_pMono(self,t):
        pMono = np.where( t<self.age_d, self.pMonoS , self.pMonoS -  (t-self.age_d)/(self.age - self.age_d)* (self.pMonoS - self.pMonoSD) )
        
        dtpMono = np.where( t<self.age_d, 0,  -1/(self.age - self.age_d)* (self.pMonoS - self.pMonoSD))
        return pMono,dtpMono
 
    
class POP8P_d_exp_Hcalc(POP8P_d_exp):
    def __init__(self,data,age0):
        self.p2nSD = np.array(data['2n'])
        self.p4nSD = np.array(data['4n'])
        self.p8nSD = np.where(1-self.p2nSD-self.p4nSD-np.array(data['8n'])<1e-10,
                             1-self.p2nSD-self.p4nSD,
                             np.array(data['8n']))
        self.age = np.array(data['age'])
        self.age_d = np.array(data[age0])

        
        self.p2nS = np.array(data['h2n'])
        self.p4nS = np.array(data['h4n'])
        self.p8nS = np.array(data['h8n'])


        self.n4_0 = N4_0
        self.n8_0 = N8_0 
        self.n2_0 = 1 - self.n4_0 - self.n8_0

        
        self.pMonoS = PMONOS 
        self.pMonoSD = PMONOSD
        self.pMono_0 = PMONO_0
        
    
        self.t0_2n = 11.632384709730932
        self.Ni_2n = ( self.p2nS - self.n2_0*np.exp(-self.age_d/self.t0_2n) ) / ( 1-np.exp(-self.age_d/self.t0_2n) )
        
        self.t0_4n = self.t0_2n#10.474682311920247
        self.Ni_4n =  ( self.p4nS - self.n4_0*np.exp(-self.age_d/self.t0_4n) ) / ( 1-np.exp(-self.age_d/self.t0_4n) )
        
        self.t0_8n = self.t0_2n#19.501536691850745
        self.Ni_8n =  ( self.p8nS - self.n8_0*np.exp(-self.age_d/self.t0_8n) ) / ( 1-np.exp(-self.age_d/self.t0_8n) )
        
        

        default_parameters = {'d':0.01,'sigma':0.1}
        self.logparas = ['d']
        self.linparas = ['sigma']
        limit = {i: global_limit for i in default_parameters.keys()}
        limit['sigma'] = (0,0.5)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['c2n','c4n','c8n','c16n','c2x2n','c2x4n','c2x8n','c2x16n'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)


class POP8P_H2_d_exp_Hcalc(POP8P_d_exp ):
    def __init__(self,data,age0):
        self.p2nSD = np.array(data['2n'])
        self.p4nSD = np.array(data['4n'])
        self.p8nSD = np.where(1-self.p2nSD-self.p4nSD-np.array(data['8n'])<1e-10,
                             1-self.p2nSD-self.p4nSD,
                             np.array(data['8n']))
        self.age = np.array(data['age'])
        self.age_d = np.array(data[age0])

        
        self.p2nS = np.array(data['h2n'])
        self.p4nS = np.array(data['h4n'])
        self.p8nS = np.array(data['h8n'])
        
        self.n4_0 = N4_0
        self.n8_0 = N8_0 
        self.n2_0 = 1 - self.n4_0 - self.n8_0

        
        self.pMonoS = PMONOS 
        self.pMonoSD = PMONOSD
        self.pMono_0 = PMONO_0
        
        
        self.da = DEATH_H 
    
        self.t0_2n = 11.632384709730932
        self.Ni_2n = ( self.p2nS - self.n2_0*np.exp(-self.age_d/self.t0_2n) ) / ( 1-np.exp(-self.age_d/self.t0_2n) )
        
        self.t0_4n = self.t0_2n#10.474682311920247
        self.Ni_4n =  ( self.p4nS - self.n4_0*np.exp(-self.age_d/self.t0_4n) ) / ( 1-np.exp(-self.age_d/self.t0_4n) )
        
        self.t0_8n = self.t0_2n#19.501536691850745
        self.Ni_8n =  ( self.p8nS - self.n8_0*np.exp(-self.age_d/self.t0_8n) ) / ( 1-np.exp(-self.age_d/self.t0_8n) )
        
        

        default_parameters = {'db':0.01,'sigma':0.1}
        self.logparas = ['db']
        self.linparas = ['sigma']
        limit = {i: global_limit for i in default_parameters.keys()}
        limit['sigma'] = (0,0.5)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['c2n','c4n','c8n','c16n','c2x2n','c2x4n','c2x8n','c2x16n'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)

    def calc_prior(self):
        return 1


    def calc_implicit_parameters(self, t):
        self.d = np.where( t > self.age_d , self.db, self.da )
        return super().calc_implicit_parameters(t)




class POP8P_D_d_exp_Hcalc(POP8P_d_exp):
    iparas = ['b2n', 'k2n4n', 'k4n8n', 'k8n16n', 'k2n2x2n', 'k2x2n2x4n', 'k2x4n2x8n', 'k2x8n2x16n']
    flow_in =  {'N2n':[('b2n','N2n',2)],'N4n':[('b4n','N4n',2),('k2n4n','N2n',1)],
    'N8n':[('b8n','N8n',2),('k4n8n','N4n',1)],'N16n':[('b16n','N16n',2),('k8n16n','N8n',1)],
    'N2x2n':[('b2x2n','N2x2n',2),('k2n2x2n','N2n',1)],
    'N2x4n':[('b2x4n','N2x4n',2),('k2x2n2x4n','N2x2n',1),('k4n2x4n','N4n',1)],
    'N2x8n':[('b2x8n','N2x8n',2),('k2x4n2x8n','N2x4n',1),('k8n2x8n','N8n',1)],
    'N2x16n':[('b2x16n','N2x16n',2),('k2x8n2x16n','N2x8n',1),('k16n2x16n','N16n',1)]}
    def __init__(self,data,age0):
        self.p2nSD = np.array(data['2n'])
        self.p4nSD = np.array(data['4n'])
        self.p8nSD = np.where(1-self.p2nSD-self.p4nSD-np.array(data['8n'])<1e-10,
                             1-self.p2nSD-self.p4nSD,
                             np.array(data['8n']))
        self.age = np.array(data['age'])
        self.age_d = np.array(data[age0])

        
        self.p2nS = np.array(data['h2n'])
        self.p4nS = np.array(data['h4n'])
        self.p8nS = np.array(data['h8n'])
        
        self.n4_0 = N4_0
        self.n8_0 = N8_0 
        self.n2_0 = 1 - self.n4_0 - self.n8_0

        
        self.pMonoS = PMONOS 
        self.pMonoSD = PMONOSD
        self.pMono_0 = PMONO_0
        
        
    
        self.t0_2n = 11.632384709730932
        self.Ni_2n = ( self.p2nS - self.n2_0*np.exp(-self.age_d/self.t0_2n) ) / ( 1-np.exp(-self.age_d/self.t0_2n) )
        
        self.t0_4n = self.t0_2n#10.474682311920247
        self.Ni_4n =  ( self.p4nS - self.n4_0*np.exp(-self.age_d/self.t0_4n) ) / ( 1-np.exp(-self.age_d/self.t0_4n) )
        
        self.t0_8n = self.t0_2n#19.501536691850745
        self.Ni_8n =  ( self.p8nS - self.n8_0*np.exp(-self.age_d/self.t0_8n) ) / ( 1-np.exp(-self.age_d/self.t0_8n) )
        
        

        default_parameters = {'d':0.01,'k4n2x4n':0.01,'k8n2x8n':0.01,'k16n2x16n':0.01,'b4n':0.01,
                    'b8n':0.01,'b16n':0.01,'b2x2n':0.01,'b2x4n':0.01,'b2x8n':0.01,'b2x16n':0.01,'sigma':0.1}
        self.logparas = ['d','k4n2x4n','k8n2x8n','k16n2x16n','b4n','b8n','b16n','b2x2n','b2x4n','b2x8n','b2x16n']
        self.linparas = ['sigma']
        limit = {i: global_limit for i in default_parameters.keys()}
        limit['sigma'] = (0,0.5)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['c2n','c4n','c8n','c16n','c2x2n','c2x4n','c2x8n','c2x16n'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)



    def calc_implicit_parameters(self, t):
        d = self.d
        k4n2x4n = self.k4n2x4n
        k8n2x8n = self.k8n2x8n
        k16n2x16n = self.k16n2x16n
        b4n = self.b4n
        b8n = self.b8n
        b16n = self.b16n
        b2x2n = self.b2x2n
        b2x4n = self.b2x4n
        b2x8n = self.b2x8n
        b2x16n = self.b2x16n

        Ntotal = 1
        dtNtotal = 0

        pMono,dtpMono = self.calc_pMono(t)
        p2n,dtp2n = self.calc_p2n(t)
        p4n,dtp4n = self.calc_p4n(t)
        p8n,dtp8n = self.calc_p8n(t)

        N2n = Ntotal*p2n*pMono
        N4n = Ntotal*p4n*pMono
        N8n = Ntotal*p8n*pMono
        N16nb = -(Ntotal*(-1 + p2n + p4n + p8n)*pMono)
        N2x2n = -(Ntotal*p2n*(-1 + pMono))
        N2x4n = -(Ntotal*p4n*(-1 + pMono))
        N2x8n = -(Ntotal*p8n*(-1 + pMono))
        N2x16nb = Ntotal*(-1 + p2n + p4n + p8n)*(-1 + pMono)
        
        N16n   = np.where(N16nb  <1e-10,1e-10*np.ones_like(N16nb),  N16nb)
        N2x16n = np.where(N2x16nb<1e-10,1e-10*np.ones_like(N2x16nb),N2x16nb)
        

        dtN2n = Ntotal*pMono*dtp2n + p2n*(pMono*dtNtotal + Ntotal*dtpMono)
        dtN4n = Ntotal*pMono*dtp4n + p4n*(pMono*dtNtotal + Ntotal*dtpMono)
        dtN8n = Ntotal*pMono*dtp8n + p8n*(pMono*dtNtotal + Ntotal*dtpMono)
        dtN16n = -(pMono*((-1 + p2n + p4n + p8n)*dtNtotal + Ntotal*(dtp2n + dtp4n + dtp8n))) - Ntotal*(-1 + p2n + p4n + p8n)*dtpMono
        dtN2x2n = -((-1 + pMono)*(p2n*dtNtotal + Ntotal*dtp2n)) - Ntotal*p2n*dtpMono
        dtN2x4n = -((-1 + pMono)*(p4n*dtNtotal + Ntotal*dtp4n)) - Ntotal*p4n*dtpMono
        dtN2x8n = -((-1 + pMono)*(p8n*dtNtotal + Ntotal*dtp8n)) - Ntotal*p8n*dtpMono
        dtN2x16n = (-1 + p2n + p4n + p8n)*(-1 + pMono)*dtNtotal + Ntotal*((-1 + pMono)*(dtp2n + dtp4n + dtp8n) + (-1 + p2n + p4n + p8n)*dtpMono)

        

        #implicit paras 

        b2n = ((-b16n + d)*N16n - b2x16n*N2x16n - b2x2n*N2x2n - b2x4n*N2x4n - b2x8n*N2x8n - b4n*N4n - b8n*N8n + d*(N2n + N2x16n + N2x2n + N2x4n + N2x8n + N4n + N8n) + dtN16n + dtN2n + dtN2x16n + dtN2x2n + dtN2x4n + dtN2x8n + dtN4n + dtN8n)/N2n
        k2n4n = ((-b16n + d + k16n2x16n)*N16n + (-b4n + d + k4n2x4n)*N4n + (-b8n + d + k8n2x8n)*N8n + dtN16n + dtN4n + dtN8n)/N2n
        k4n8n = ((-b16n + d + k16n2x16n)*N16n + (-b8n + d + k8n2x8n)*N8n + dtN16n + dtN8n)/N4n
        k8n16n = ((-b16n + d + k16n2x16n)*N16n + dtN16n)/N8n
        k2n2x2n = (-(k16n2x16n*N16n) + (-b2x16n + d)*N2x16n - b2x2n*N2x2n - b2x4n*N2x4n - b2x8n*N2x8n + d*(N2x2n + N2x4n + N2x8n) - k4n2x4n*N4n - k8n2x8n*N8n + dtN2x16n + dtN2x2n + dtN2x4n + dtN2x8n)/N2n
        k2x2n2x4n = (-(k16n2x16n*N16n) + (-b2x16n + d)*N2x16n + (-b2x4n + d)*N2x4n + (-b2x8n + d)*N2x8n - k4n2x4n*N4n - k8n2x8n*N8n + dtN2x16n + dtN2x4n + dtN2x8n)/N2x2n
        k2x4n2x8n = (-(k16n2x16n*N16n) + (-b2x16n + d)*N2x16n + (-b2x8n + d)*N2x8n - k8n2x8n*N8n + dtN2x16n + dtN2x8n)/N2x4n
        k2x8n2x16n = (-(k16n2x16n*N16n) + (-b2x16n + d)*N2x16n + dtN2x16n)/N2x8n
                
        
        k8n16n = np.where(N16n==1e-10,np.zeros_like(k8n16n),k8n16n)
        k2x8n2x16n = np.where(N2x16n==1e-10,np.zeros_like(k2x8n2x16n),k2x8n2x16n)


        iparas ={'b2n':b2n,'k2n4n': k2n4n,'k4n8n': k4n8n,'k8n16n': k8n16n,'k2n2x2n': k2n2x2n,'k2x2n2x4n': k2x2n2x4n,'k2x4n2x8n': k2x4n2x8n,'k2x8n2x16n': k2x8n2x16n,
           'N2n':N2n,'N4n':N4n,'N8n':N8n,'N16n':N16n,'N2x2n':N2x2n,'N2x4n':N2x4n,'N2x8n':N2x8n,'N2x16n':N2x16n }

        if ( np.logical_and(N2n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2n ({N2n})is negative at t = {t}', iparas)
        if ( np.logical_and(N4n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N4n ({N4n})is negative at t = {t}', iparas)
        if ( np.logical_and(N8n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N8n ({N8n})is negative at t = {t}', iparas)
        if ( np.logical_and(N16nb<-1e10,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N16nb ({N16nb})is negative at t = {t}', iparas)
        if ( np.logical_and(N2x2n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2x2n ({N2x2n})is negative at t = {t}', iparas)
        if ( np.logical_and(N2x4n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2x4n ({N2x4n})is negative at t = {t}', iparas)
        if ( np.logical_and(N2x8n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2x8n ({N2x8n})is negative at t = {t}', iparas)
        if ( np.logical_and(N2x16nb<-1e10,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'N2x16nb ({N2x16nb})is negative at t = {t}', iparas)


        if ( np.logical_and(b2n<0,t<=self.age) ).any():
            raise ImplicitParametersOutOfRange(f'b2n ({b2n})is negative at t = {t}', iparas)
        if (np.logical_and(k2n4n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2n4n ({k2n4n})is negative at t = {t}', iparas)
        if (np.logical_and(k4n8n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k4n8n ({k4n8n})is negative at t = {t}', iparas)
        if (np.logical_and(k8n16n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k8n16n ({k8n16n})is negative at t = {t}', iparas)    
        if (np.logical_and(k2n2x2n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2n2x2n ({k2n2x2n})is negative at t = {t}', iparas)
        if (np.logical_and(k2x2n2x4n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2x2n2x4n ({k2x2n2x4n})is negative at t = {t}', iparas)
        if (np.logical_and(k2x4n2x8n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2x4n2x8n ({k2x4n2x8n})is negative at t = {t}', iparas)
        if (np.logical_and(k2x8n2x16n<0,t<=self.age)).any():
            raise ImplicitParametersOutOfRange(f'k2x8n2x16n ({k2x8n2x16n})is negative at t = {t}', iparas)

        return iparas          

   

    def rhs(self, t, y):
        M = np.reshape(y, (self.nvars, -1))
        M_new = np.zeros_like(M)
        
        iparas = self.calc_implicit_parameters(t)
        
        d = self.d
        k4n2x4n = self.k4n2x4n
        k8n2x8n = self.k8n2x8n
        k16n2x16n = self.k16n2x16n
        b4n = self.b4n
        b8n = self.b8n
        b16n = self.b16n
        b2x2n = self.b2x2n
        b2x4n = self.b2x4n
        b2x8n = self.b2x8n
        b2x16n = self.b2x16n

        c2n = self.c2n
        c4n = self.c4n
        c8n = self.c8n
        c16n = self.c16n
        c2x2n = self.c2x2n
        c2x4n = self.c2x4n
        c2x8n = self.c2x8n
        c2x16n = self.c2x16n
        
        b2n = iparas['b2n']
        k2n4n = iparas['k2n4n']
        k4n8n = iparas['k4n8n']
        k8n16n = iparas['k8n16n']
        k2n2x2n = iparas['k2n2x2n']
        k2x2n2x4n = iparas['k2x2n2x4n']
        k2x4n2x8n = iparas['k2x4n2x8n']
        k2x8n2x16n = iparas['k2x8n2x16n']

        N2n = iparas['N2n']
        N4n = iparas['N4n']
        N8n = iparas['N8n']
        N16n = iparas['N16n']
        N2x2n = iparas['N2x2n']
        N2x4n = iparas['N2x4n']
        N2x8n = iparas['N2x8n']
        N2x16n = iparas['N2x16n']

        Catm_t = self.Catm.lin(t + self.Dbirth)

        # c14 equations 
        M_new[c2n] =b2n*(-M[c2n] + Catm_t)
        M_new[c4n] =b4n*(-M[c4n] + Catm_t) + (k2n4n*(M[c2n] - 2*M[c4n] + Catm_t)*N2n)/(2.*N4n)
        M_new[c8n] =b8n*(-M[c8n] + Catm_t) + (k4n8n*(M[c4n] - 2*M[c8n] + Catm_t)*N4n)/(2.*N8n)
        M_new[c16n] =(2*b16n*(-M[c16n] + Catm_t)*N16n + k8n16n*(-2*M[c16n] + M[c8n] + Catm_t)*N8n)/(2.*N16n)
        M_new[c2x2n] =b2x2n*(-M[c2x2n] + Catm_t) + (k2n2x2n*(M[c2n] - 2*M[c2x2n] + Catm_t)*N2n)/(2.*N2x2n)
        M_new[c2x4n] =(k2x2n2x4n*M[c2x2n]*N2x2n + Catm_t*(k2x2n2x4n*N2x2n + 2*b2x4n*N2x4n) + k4n2x4n*(M[c4n] + Catm_t)*N4n - 2*M[c2x4n]*(k2x2n2x4n*N2x2n + b2x4n*N2x4n + k4n2x4n*N4n))/(2.*N2x4n)
        M_new[c2x8n] =(k2x4n2x8n*M[c2x4n]*N2x4n + Catm_t*(k2x4n2x8n*N2x4n + 2*b2x8n*N2x8n) + k8n2x8n*(M[c8n] + Catm_t)*N8n - 2*M[c2x8n]*(k2x4n2x8n*N2x4n + b2x8n*N2x8n + k8n2x8n*N8n))/(2.*N2x8n)
        M_new[c2x16n] =(k16n2x16n*M[c16n]*N16n + Catm_t*(k16n2x16n*N16n + 2*b2x16n*N2x16n) + k2x8n2x16n*(M[c2x8n] + Catm_t)*N2x8n - 2*M[c2x16n]*(k16n2x16n*N16n + b2x16n*N2x16n + k2x8n2x16n*N2x8n))/(2.*N2x16n)

        return M_new.ravel()


class POP8P_D_h_exp(POP8P_D_d_exp_Hcalc):
    def __init__(self,data,age0):
        self.p2nS = np.array(data['2n'])
        self.p4nS = np.array(data['4n'])
        self.p8nS = np.where(1-self.p2nS-self.p4nS-np.array(data['8n'])<1e-10,
                             1-self.p2nS-self.p4nS,
                             np.array(data['8n']))
        self.age = np.array(data['age'])
        
        self.pMonoS = 0.8
    

        self.n4_0 = 0.04
        if (self.p4nS < 0.04).any():
            print ('error in n4 at t=0')
        self.n8_0 = 0.005
        if (self.p8nS < 0.005).any():
            print ('error in n8 at t=0')
        self.n2_0 = 1 - self.n4_0 - self.n8_0
            
        self.t0_2n = 11.632384709730932
        self.Ni_2n = ( self.p2nS - self.n2_0*np.exp(-self.age/self.t0_2n) ) / ( 1-np.exp(-self.age/self.t0_2n) )
        
        self.t0_4n = self.t0_2n#10.474682311920247
        self.Ni_4n =  ( self.p4nS - self.n4_0*np.exp(-self.age/self.t0_4n) ) / ( 1-np.exp(-self.age/self.t0_4n) )
        
        self.t0_8n = self.t0_2n#19.501536691850745
        self.Ni_8n =  ( self.p8nS - self.n8_0*np.exp(-self.age/self.t0_8n) ) / ( 1-np.exp(-self.age/self.t0_8n) )
        
        

        default_parameters = {'d':0.1,'k4n2x4n':0.001,'k8n2x8n':0.001,'k16n2x16n':0.001,'b4n':0.01,
                    'b8n':0.01,'b16n':0.01,'b2x2n':0.01,'b2x4n':0.01,'b2x8n':0.01,'b2x16n':0.01,'sigma':0.1}
        self.logparas = ['d','k4n2x4n','k8n2x8n','k16n2x16n','b4n','b8n','b16n','b2x2n','b2x4n','b2x8n','b2x16n']
        self.linparas = ['sigma']
        limit = {i: global_limit for i in default_parameters.keys()}
        limit['sigma'] = (0,0.5)
        self.Catm = Catm(delay=1)
        model_base.__init__(self, var_names=['c2n','c4n','c8n','c16n','c2x2n','c2x4n','c2x8n','c2x16n'],
                            default_parameters=default_parameters,
                            error={i: global_error for i in
                                   default_parameters.keys()},
                            limit=limit)

    def calc_p2n(self,t):
        p2n =    (self.n2_0-self.Ni_2n)*np.exp(-t/self.t0_2n)+self.Ni_2n
        dtp2n = -1*(self.n2_0-self.Ni_2n)*np.exp(-t/self.t0_2n)/self.t0_2n
        return p2n,dtp2n

    def calc_p4n(self,t):
        p4n =    (self.n4_0-self.Ni_4n)*np.exp(-t/self.t0_4n)+self.Ni_4n
        dtp4n = -1*(self.n4_0-self.Ni_4n)*np.exp(-t/self.t0_4n)/self.t0_4n
             
        return p4n,dtp4n

    def calc_p8n(self,t):
        p8n =    (self.n8_0-self.Ni_8n)*np.exp(-t/self.t0_8n)+self.Ni_8n
        dtp8n = -1*(self.n8_0-self.Ni_8n)*np.exp(-t/self.t0_8n)/self.t0_8n
        
        return p8n,dtp8n
    
    
    def calc_pMono(self,t):
        
        pMono = self.pMonoS
        dtpMono = 0
        
        return pMono,dtpMono    
    







models_list = [POP8P_h_exp,POP8P_D_h_exp, POP8P_d_exp_Hcalc, POP8P_D_d_exp_Hcalc,POP8P_H2_d_exp_Hcalc,POP1damage]

