import numpy as np
import pandas as pd
import collections
import arviz as az
import logging
import collections

from .models.base import exp_data,Catm
from .plotting import visualisze
from .tools import *
from .optimize import optimize

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


def get_arviz(results,burnin=1000,remove_stuck=True,age=None):
    for n,v in results['results'].items():
        print(n)
        model = results['models'][n]
        v['pe'] = find_point_estimate(v['raw'],model)
        v['azdata'],v['azdata_names'] = convert_to_arviz(v['raw'],model,burnin,remove_stuck=remove_stuck,iparas_time=age,phy_space=False)
        chsck = run_convergence_checks(v['azdata'])
        for i in chsck:
            print(i)
        print('\n')
    return

def median_sd(x):
     median = np.percentile(x, 50)
     sd = np.sqrt(np.mean((x-median)**2))
     return sd
func_dict = {
     "std": np.std,
     "median_std": median_sd,
     "percentile.1585": lambda x: np.percentile(x, 15.85),
     "median": lambda x: np.percentile(x, 50),
     "percentile.8415": lambda x: np.percentile(x, 84.15),
}



def get_getranking(results,method='BB-pseudo-BMA'):
    if len(results)<2:
        return None
    results['ranking'] = az.compare({n:v['azdata'] for n,v in results['results'].items()},ic='loo', method=method)
    return results['ranking']




def get_pointestimates(results,burnin=1000, remove_stuck=True,error_in_real=True,sigma=1):
    if sigma==1:
        hdi_prob=0.6827
        lowern = 'hdi_15.865%'
        uppern = 'hdi_84.135%'
    elif sigma==2:
        hdi_prob=0.9545
        lowern = 'hdi_2.275%'
        uppern = 'hdi_97.725%'

    idx = pd.IndexSlice
    export_data = []
    res_data = []
    try: 
        ranking = results['ranking']
    except KeyError:
        print('no ranking')
        ranking = pd.DataFrame(np.ones((len(results['results'].keys()),3)),index=results['results'].keys(),columns=['rank','loo','weight'])

    for n,v in results['results'].items():
        model = results['models'][n]
        if error_in_real:
            azdata,azdata_names = convert_to_arviz(v['raw'],model,burnin,remove_stuck=remove_stuck,phy_space=True)
        else:
            azdata = v['azdata']
            azdata_names = v['azdata_names']
        df = az.summary(azdata,round_to=8,hdi_prob=hdi_prob,stat_funcs=func_dict)[['mean', lowern, uppern, 'median']]
        df.index = azdata_names
        if error_in_real:
            pass
        else:
            for col in df.columns:
                para_real = model.transform_fit_to_physical( dict(df[col]) , mode="bayes")
                df[col] = [para_real[k] for k in df.index]
        df_new = df.rename(columns={lowern:'lower',uppern:'upper'}).unstack()
        v['median'] = df['median']
        value = [df_new.loc[('median',p)]*100 if p in model.logparas else df_new.loc[('median',p)] for p in model.parameter_names]  #toget perecent for rates 
        limits=  np.array([ [df_new.loc[('lower',p)]*100,df_new.loc[('upper',p)]*100]  if p in model.logparas else [df_new.loc[('lower',p)],df_new.loc[('upper',p)]] for p in model.parameter_names])
        limitspm=(limits.T - value).T
        valuestr=[]
        limit_intstr=[]
        limit_pmstr=[]
        for i in range(len(limitspm)):
            r = int(max(-np.floor(np.log10(limitspm[i,1])),-np.floor(np.log10(-limitspm[i,0]))))+2
            if model.parameter_names[i] in model.logparas:
                unit='%/year'
            else:
                unit=''
            valuestr.append(f"{np.round(value[i],decimals=r):.{max(r,0)}f}{unit}") 
            limit_intstr.append(f"[{np.round(limits[i,0],decimals=1+r):.{max(1+r,0)}f}{unit} - {np.round(limits[i,1],decimals=1+r):.{max(1+r,0)}f}{unit}]")
            limit_pmstr.append(f"{np.round(limitspm[i,0],decimals=1+r):.{max(1+r,0)}f}{unit};+{np.round(limitspm[i,1],decimals=1+r):.{max(1+r,0)}f}{unit}")
        s = pd.DataFrame()
        s['Value'] =  valuestr
        s['Confidence Interval'] = limit_intstr
        s['Limits rel'] = limit_pmstr
        s['Scenario'] = n
        s['Parameter'] =  [p  for p in model.parameter_names]
        #s['order'] = orr[n]
        s['dummy'] = 0
        for t in ['rank','loo','weight']:
            s[t] = ranking.loc[n,t]
        #s['Scenario'] = [mrename[i] for i in s['Scenario']]
        s['loo'] = [f"{np.round(i,decimals=2)}" for i in s['loo']]
        s['Weight'] = [f"{np.round(i*100,decimals=0):.0f}%" for i in s['weight']]
        #export_data.append(s.set_index(['order','rank','Scenario','loo','Weight','dummy']).sort_index())
        export_data.append(s.set_index(['rank','Scenario','loo','Weight','dummy']).sort_index())
        
        
        res_data.append(pd.Series(df.unstack(level=1 ),name=n) )
    export_data= pd.concat(export_data).sort_index()
    #export_data = export_data.droplevel('rank').droplevel('order')
    #export_data = export_data.droplevel('rank')
    #export_data2= export_data.droplevel('dummy')
    results['point_est'] = pd.DataFrame(res_data)
    return export_data


def calc_c14_all(results,PE=False):
    data = results['data']
    edata = exp_data(results['data'])
    for ind,model in results['models'].items():
        print(ind)
        if PE:
            para = model.transform_fit_to_physical( results['results'][ind]['pe'], mode="bayes")
        else:
            para = {i:results['point_est'].loc[ind].loc[('median',i)] for i in model.parameter_names}
        model.set_parameters_phy(para,mode='bayes')
        visd = visualisze(model,edata, step_size=0.1,Dbirth_sim=edata.Dbirth,age=1)
        visd.odeint()      
        results['results'][ind]['c14'] = visd.solpd_data
        
        ttmp = [(ind,visd.solpd_data.index[visd.solpd_data.index.get_loc(row['age'],method='nearest')]) for ind,row in results['data'].iterrows()]
        results_as_optimizer = pd.concat([visd.solpd_data.loc[i[1],(model.var_names,i[0])] for i in ttmp]).unstack(level=0).reindex(data.index)
        
        mdata= model.measurement_model(results_as_optimizer, edata)
        results['results'][ind]['c14_M'] =  mdata      
        
        
        ipsd=[]
        T = np.array(visd.solpd_data.index)
        for i in T:
            ip = model.calc_implicit_parameters(i)
            ipsd.append(ip)
        

        for tt in model.plot_types:
            data_s = []
            data_m = []
            for a_i,(index,row) in enumerate(data.iterrows()):
                if isinstance(ipsd[0][model.populations[0]], (collections.Sequence, np.ndarray)): 
                    ipid = listofdict_to_dictofarray_f(ipsd,a_i)    
                else:
                    ipid = listofdict_to_dictofarray(ipsd)
                t = T
                c = np.zeros_like(t)
                n = np.zeros_like(t)
                for pop in model.populations_plot[tt]:
                    tmp = ipid[pop]*model.populations_DNA[pop]
                    n += tmp
                    c += tmp* visd.solpd_data.loc[:,(f'c{pop[1:]}',index)]
                tmp = pd.Series(c/n,index=t,name=index)
                data_s.append(tmp)
                data_m.append(tmp.iloc[tmp.index.get_loc(row['age'],method='nearest')])
                    
                    
            results['results'][ind]['c14s_'+tt] =  pd.concat(data_s,axis=1)
            results['results'][ind]['c14m_'+tt] =  pd.Series(data_m,index=data.index)
        

        '''
        c14s_t = []
        c14s_2n= []
        c14s_4n= []
        c14s_pn= []
        
        c14m_t = []
        c14m_2n= []
        c14m_4n= []
        c14m_pn= []
        for a_i,(index,row) in enumerate(data.iterrows()):
            ipid = listofdict_to_dictofarray_f(ipsd,a_i)
            t = T
            c = np.zeros_like(t)
            n = np.zeros_like(t)
            for pop in model.populations:
                tmp = ipid[pop]*model.populations_DNA[pop]
                n += tmp
                c += tmp* visd.solpd_data.loc[:,(f'c{pop[1:]}',index)]
            tmp = pd.Series(c/n,index=t,name=index)
            c14s_t.append(tmp)
            c14m_t.append(tmp.iloc[int(np.floor(10*row['age']))])
            
            c = np.zeros_like(t)
            n = np.zeros_like(t)
            for pop in model.populations_2n:
                tmp = ipid[pop]*model.populations_DNA[pop]
                n += tmp
                c += tmp* visd.solpd_data.loc[:,(f'c{pop[1:]}',index)]
            tmp = pd.Series(c/n,index=t,name=index)
            c14s_2n.append(tmp)
            c14m_2n.append(tmp.iloc[int(np.floor(10*row['age']))])

            
            c = np.zeros_like(t)
            n = np.zeros_like(t)
            for pop in model.populations_4n:
                tmp = ipid[pop]*model.populations_DNA[pop]
                n += tmp
                c += tmp* visd.solpd_data.loc[:,(f'c{pop[1:]}',index)]
            tmp = pd.Series(c/n,index=t,name=index)
            c14s_4n.append(tmp)
            c14m_4n.append(tmp.iloc[int(np.floor(10*row['age']))])

            
            c = np.zeros_like(t)
            n = np.zeros_like(t)
            for pop in model.populations_pn:
                tmp = ipid[pop]*model.populations_DNA[pop]
                n += tmp
                c += tmp* visd.solpd_data.loc[:,(f'c{pop[1:]}',index)]
            tmp = pd.Series(c/n,index=t,name=index)
            c14s_pn.append(tmp)
            c14m_pn.append(tmp.iloc[int(np.floor(10*row['age']))])

            
        
        results['results'][ind]['c14s_t'] = pd.concat(c14s_t,axis=1)
        results['results'][ind]['c14s_2n'] = pd.concat(c14s_2n,axis=1)
        results['results'][ind]['c14s_4n'] = pd.concat(c14s_4n,axis=1)
        results['results'][ind]['c14s_pn'] = pd.concat(c14s_pn,axis=1)
        
        results['results'][ind]['c14m_t'] =  pd.Series(c14m_t,index=data.index)
        results['results'][ind]['c14m_2n'] = pd.Series(c14m_2n,index=data.index)
        results['results'][ind]['c14m_4n'] = pd.Series(c14m_4n,index=data.index)
        results['results'][ind]['c14m_pn'] = pd.Series(c14m_pn,index=data.index)
        '''
    return 

