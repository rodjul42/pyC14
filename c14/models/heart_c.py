import numpy as np
import pandas as pd
from .base import model_base, Catm
from scipy.interpolate import UnivariateSpline
from ..tools import ImplicitParametersOutOfRange,FailedToCalulatePopulation,NonFiniteValuesinIntegrate,listofdict_to_dictoflist,listofdict_to_dictofarray

import scipy as sp
import scipy.integrate
from scipy.stats import norm
from scipy.interpolate import interp1d


import logging
logger =  logging.getLogger(__name__)

from c14.models.heart import *


import c14.models.models_c.heart_c as  heart_c

class POP8P_h_exp_c(POP8P_h_exp):
  
    def calc_initial_parameters(self):
        self.indata = np.stack([self.age,self.Ni_2n,self.Ni_4n,self.Ni_8n])
        self.ipara = np.ones((17,len(self.age)))
        self.new_data = np.zeros(self.nvars*len(self.age))
        return 
    
    
    def rhs(self, t, y):
        Catm_t = self.Catm.lin(t + self.Dbirth)
        ret = heart_c.POP8_h_exp_rhs(self.indata,Catm_t,y,self.new_data,1.0*t,self.d,self.pMonoS,self.n2_0,self.n4_0,self.n8_0,self.t0_2n)
        if ret >0:
            raise ImplicitParametersOutOfRange(f'(ipara is negative at t = {t}', None)
        return self.new_data          
                       
  

class POP8P_d_exp_c(POP8P_d_exp):
  
    def calc_initial_parameters(self):
        self.indata = np.stack([self.age,self.Ni_2n,self.Ni_4n,self.Ni_8n,self.age_d,self.p2nSD,self.p4nSD,self.p8nSD])
        self.ipara = np.ones((17,len(self.age)))
        self.new_data = np.zeros(self.nvars*len(self.age))
        return 
    
    
    def rhs(self, t, y):
        Catm_t = self.Catm.lin(t + self.Dbirth)
        ret = heart_c.POP8_d_exp_rhs(self.indata,Catm_t,y,self.new_data,1.0*t,
            self.d,self.pMonoS,self.n2_0,self.n4_0,self.n8_0,self.t0_2n, self.p2nS, self.p4nS,self.pMonoSD )
        if ret >0:
            raise ImplicitParametersOutOfRange(f'(ipara is negative at t = {t}', None)
        return self.new_data       


class POP8P_d_exp_Hcalc_c(POP8P_d_exp_Hcalc):
  
    def calc_initial_parameters(self):
        self.indata = np.stack([self.age,self.Ni_2n,self.Ni_4n,self.Ni_8n,self.age_d,self.p2nSD,self.p4nSD,self.p8nSD,self.p2nS,self.p4nS])
        self.ipara = np.ones((17,len(self.age)))
        self.new_data = np.zeros(self.nvars*len(self.age))
        return 
    
    
    def rhs(self, t, y):
        Catm_t = self.Catm.lin(t + self.Dbirth)
        ret = heart_c.POP8P_d_exp_Hcalc_rhs(self.indata,Catm_t,y,self.new_data,1.0*t,
            self.d,self.pMonoS,self.n2_0,self.n4_0,self.n8_0,self.t0_2n, self.pMonoSD )
        if ret >0:
            raise ImplicitParametersOutOfRange(f'(ipara is negative at t = {t}', None)
        return self.new_data       


class POP8P_H2_d_exp_Hcalc_c(POP8P_H2_d_exp_Hcalc):
    def calc_initial_parameters(self):
        self.indata = np.stack([self.age,self.Ni_2n,self.Ni_4n,self.Ni_8n,self.age_d,self.p2nSD,self.p4nSD,self.p8nSD,self.p2nS,self.p4nS])
        self.ipara = np.ones((17,len(self.age)))
        self.new_data = np.zeros(self.nvars*len(self.age))
        return 

    def rhs(self, t, y):
        Catm_t = self.Catm.lin(t + self.Dbirth)
        ret = heart_c.POP8P_H2b_d_exp_Hcalc_rhs(self.indata,Catm_t,y,self.new_data,1.0*t,
            self.da,self.db,self.pMonoS,self.n2_0,self.n4_0,self.n8_0,self.t0_2n, self.pMonoSD )
        if ret >0:
            raise ImplicitParametersOutOfRange(f'(ipara is negative at t = {t}', None)
        return self.new_data


class POP8P_D_d_exp_Hcalc_c(POP8P_D_d_exp_Hcalc):
  
    def calc_initial_parameters(self):
        self.indata = np.stack([self.age,self.Ni_2n,self.Ni_4n,self.Ni_8n,self.age_d,self.p2nSD,self.p4nSD,self.p8nSD,self.p2nS,self.p4nS])
        self.ipara = np.ones((17,len(self.age)))
        self.new_data = np.zeros(self.nvars*len(self.age))
        return 
    
    
    def rhs(self, t, y):
        Catm_t = self.Catm.lin(t + self.Dbirth)
        ret = heart_c.POP8P_D_d_exp_Hcalc_rhs(self.indata,Catm_t,y,self.new_data,1.0*t,
            self.d, self.k4n2x4n, self.k8n2x8n, self.k16n2x16n,
            self.b4n, self.b8n, self.b16n, self.b2x2n, self.b2x4n, self.b2x8n, self.b2x16n,
            self.pMonoS,self.n2_0,self.n4_0,self.n8_0,self.t0_2n, self.pMonoSD )
        if ret >0:
            raise ImplicitParametersOutOfRange(f'(ipara is negative at t = {t}', None)
        return self.new_data       

class POP8P_D_h_exp_c(POP8P_D_h_exp):
  
    def calc_initial_parameters(self):
        self.indata = np.stack([self.age,self.Ni_2n,self.Ni_4n,self.Ni_8n])
        self.ipara = np.ones((17,len(self.age)))
        self.new_data = np.zeros(self.nvars*len(self.age))
        return 
    
    
    def rhs(self, t, y):
        Catm_t = self.Catm.lin(t + self.Dbirth)
        ret = heart_c.POP8P_D_h_exp_rhs(self.indata,Catm_t,y,self.new_data,1.0*t,
            self.d, self.k4n2x4n, self.k8n2x8n, self.k16n2x16n,
            self.b4n, self.b8n, self.b16n, self.b2x2n, self.b2x4n, self.b2x8n, self.b2x16n,
            self.pMonoS,self.n2_0,self.n4_0,self.n8_0,self.t0_2n )
        if ret >0:
            raise ImplicitParametersOutOfRange(f'(ipara is negative at t = {t}', None)
        return self.new_data       



models_list = [POP8P_h_exp,POP8P_h_exp_c,POP8P_D_h_exp_c,POP8P_H2_d_exp_Hcalc_c,
    POP8P_d_exp_Hcalc_c,POP8P_D_d_exp_Hcalc_c]
