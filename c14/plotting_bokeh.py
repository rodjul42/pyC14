from datetime import date
from operator import mod
from re import S
from bokeh import models
from bokeh.models.annotations import Tooltip
import numpy as np
from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row,gridplot
from bokeh.models import Div
from bokeh.plotting import figure,ColumnDataSource
import bokeh.palettes 
from .tools import listofdict_to_dictofarray_f
from .models.base import  Catm

TOOLTIPS = [
    ("", "$name")
]
TOOLTIPS2 = [
    ("", "$name"),
    ("", "@ind")
]
opts = dict(min_border=0,tools="tap,pan,wheel_zoom,box_zoom,reset")


def gen_figs(names,ncol,opts=opts,tooltips=TOOLTIPS):
    n_plots = len( names )

    plt_shape = (ncol,int(np.ceil(n_plots/ncol)))
    sub_figs= []
    for i,name in enumerate(names):
        p=figure(title=name,tooltips=tooltips,**opts)
        p.xaxis[0].axis_label = 'age'
        sub_figs.append( p)
    sub_figs += [None]*(plt_shape[0]*plt_shape[1] - n_plots)
    n=0
    sub_figs_g = []
    for i in range(plt_shape[1]):
        tmp=[]
        for j in range(plt_shape[0]):
            tmp.append(sub_figs[n])
            n = n + 1
        sub_figs_g.append(tmp)
    return sub_figs,sub_figs_g,plt_shape


def get_plot_types(results):
    plot_types = []
    for i,(model_name,model) in enumerate(results['models'].items()):
            plot_types = plot_types + model.plot_types
    plot_types = np.unique(plot_types)
              
    
    return plot_types,{i:ii for ii,i in enumerate(plot_types)}

def calc_activity(model):
    activity = {i:{} for i in model.populations}
    for popA,flows in model.flow_in.items():
        for rate,pop,factor in flows:
            if pop is not None:
                try:
                    activity[pop][rate] += factor*model.populations_DNA[popA]
                except KeyError:
                    activity[pop][rate] = factor*model.populations_DNA[popA]
    
    activity_final = {i:[] for i in model.populations}
    for pop,v in activity.items():
        for rate_name in v.keys():
            v[rate_name] -= model.populations_DNA[pop]
            if v[rate_name]>0:
                activity_final[pop].append(rate_name)
    return  activity, activity_final

def calc_c14intake(model):
    activity = {i:{} for i in model.populations}
    for popA,flows in model.flow_in.items():
        for rate,pop,factor in flows:
            if pop is not None:
                try:
                    activity[popA][rate] += factor*model.populations_DNA[popA] - model.populations_DNA[pop]
                except KeyError:
                    activity[popA][rate] = factor*model.populations_DNA[popA] - model.populations_DNA[pop]
    

    return  activity

def plot_pop_b(model_name,results,ncol=4,plot_width=350,plot_height=200,hheight=10):
    PLT_name = Div(text="""<h1>Populations</>""", height=hheight, sizing_mode="stretch_width")
    
    model = results['models'][model_name]
    parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
    model.set_parameters_phy(parameter_phy,mode='bayes')
    results_model = results['results'][model_name]
    data = results['data']

    T = results_model['c14s_' + model.plot_types[0] ].index   
    ipsd=[]
    for i in T:
        ip = model.calc_implicit_parameters(i)
        ipsd.append(ip)

    sub_figs,sub_figs_g,plt_shape = gen_figs(model.populations,ncol=ncol)
    
    
    for a_i,(d_ind,d_row) in enumerate(data.iterrows()):
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        t = T[T<=d_row['age']]
        for i,ind2 in enumerate(model.populations):
            source = ColumnDataSource(data = dict(
                    x = t,
                    y = ipid[ind2][:len(t)]),name=str(d_ind))
            sub_figs[i].line('x','y',source=source,
                             name=f"index {d_ind}  age {d_row['age']:.1f}",tags=[str(d_ind)+'_line'],
                             color=bokeh.palettes.Category20_20[a_i%20])
    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height, sizing_mode="stretch_width")

def plot_ip_b(model_name,results,ncol=4,plot_width=350,plot_height=200,hheight=10):
    PLT_name = Div(text="""<h1>Implicit Parameter</>""", height=hheight, sizing_mode="stretch_width")
    
    model = results['models'][model_name]
    parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
    model.set_parameters_phy(parameter_phy,mode='bayes')
    results_model = results['results'][model_name]
    data = results['data']    

    T = results_model['c14s_' + model.plot_types[0] ].index   
    ipsd=[]
    for i in T:
        ip = model.calc_implicit_parameters(i)
        ipsd.append(ip)

    sub_figs,sub_figs_g,plt_shape = gen_figs(model.iparas,ncol=ncol)

    for a_i,(d_ind,d_row) in enumerate(data.iterrows()):
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        t = T[T<=d_row['age']]
        for i,ind2 in enumerate(model.iparas):
            source = ColumnDataSource(data = dict(
                    x = t,
                    y = ipid[ind2][:len(t)]),name=str(d_ind)) #ipid['N2n'][:len(t)]*
            sub_figs[i].line('x','y',source=source,
                             name=f"index {d_ind}  age {d_row['age']:.1f}",tags=[str(d_ind)+'_line'],
                             color=bokeh.palettes.Category20_20[a_i%20])

    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height, sizing_mode="stretch_width")


def plot_flow_b(model_name,results,ncol=4,plot_width=350,plot_height=200,hheight=10):
    PLT_name = Div(text="""<h1>Flow Parameter</>""", height=hheight, sizing_mode="stretch_width")
    
    model = results['models'][model_name]
    parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
    model.set_parameters_phy(parameter_phy,mode='bayes')
    results_model = results['results'][model_name]
    data = results['data']    

    T = results_model['c14s_' + model.plot_types[0] ].index   
    ipsd=[]
    for i in T:
        ip = model.calc_implicit_parameters(i)
        ipsd.append(ip)


    sub_figs,sub_figs_g,plt_shape = gen_figs(model.flow_in.keys(),ncol=ncol)

    for a_i,(d_ind,d_row) in enumerate(data.iterrows()):
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        ipid.update({n:np.array([v]) for n,v in parameter_phy.items()})
        t = T[T<=d_row['age']]
        for i,(ind2,flows) in enumerate(model.flow_in.items()):
            y = np.zeros_like(t)
            for rate,pop,factor in flows:
                y += ipid[rate][:len(t)] * factor * ipid[pop][:len(t)]
            
            source = ColumnDataSource(data = dict(
                    x = t,
                    y = y),name=str(data.index[a_i]))
            sub_figs[i].line('x','y',source=source,
                             name=f"index {d_ind}  age {d_row['age']:.1f}",tags=[str(d_ind)+'_line'],
                             color=bokeh.palettes.Category20_20[a_i%20])

    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height, sizing_mode="stretch_width")



def plot_act_b(model_name,results,ncol=4,plot_width=350,plot_height=200,hheight=10):
    PLT_name = Div(text="""<h1>Activity</>""", height=hheight, sizing_mode="stretch_width")
    
    model = results['models'][model_name]
    parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
    model.set_parameters_phy(parameter_phy,mode='bayes')
    results_model = results['results'][model_name]
    data = results['data']    

    T = results_model['c14s_' + model.plot_types[0] ].index   
    ipsd=[]
    for i in T:
        ip = model.calc_implicit_parameters(i)
        ipsd.append(ip)

    sub_figs,sub_figs_g,plt_shape = gen_figs(model.flow_in.keys(),ncol=ncol)
    
    _,activity = calc_activity(model)

    for a_i,(d_ind,d_row) in enumerate(data.iterrows()):
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        ipid.update({n:np.array([v]) for n,v in parameter_phy.items()})
        t = T[T<=d_row['age']]
        for i,(ind2,act) in enumerate(activity.items()):
            y = np.zeros_like(t)
            for rate in act:
                y += ipid[rate][:len(t)]
            source = ColumnDataSource(data = dict(
                    x = t,
                    y = y),name=str(data.index[a_i]))
            sub_figs[i].line('x','y',source=source,
                             name=f"index {d_ind}  age {d_row['age']:.1f}",tags=[str(d_ind)+'_line'],
                             color=bokeh.palettes.Category20_20[a_i%20])

    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height,sizing_mode="stretch_width")






def plot_Tact_b(model_name,results,ipsd=None,hheight=10,plot_height=200):
    PLT_name = Div(text="""<h1>Overall Activity</>""", height=hheight, sizing_mode="stretch_width")
    
    model = results['models'][model_name]
    parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
    model.set_parameters_phy(parameter_phy,mode='bayes')
    results_model = results['results'][model_name]
    data = results['data']    

    T = results_model['c14s_' + model.plot_types[0] ].index   
    ipsd=[]
    for i in T:
        ip = model.calc_implicit_parameters(i)
        ipsd.append(ip)
    fig = figure(title='Overal activity',plot_height=plot_height, sizing_mode="stretch_width")
    fig.xaxis[0].axis_label = 'age'

    
    _,activity = calc_activity(model)

    
    for a_i,(d_ind,d_row) in enumerate(data.iterrows()):
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        ipid.update({n:np.array([v]) for n,v in parameter_phy.items()})
        t = T[T<=d_row['age']]
        nT = np.zeros_like(t)
        yT = np.zeros_like(t)
        for i,(ind2,act) in enumerate(activity.items()):
            y = np.zeros_like(t)
            for rate in act:
                y += ipid[rate][:len(t)]
            yT += y * ipid[ind2][:len(t)]
            nT +=ipid[ind2][:len(t)] 
        datai = data.iloc[a_i]
        source = ColumnDataSource(data = dict(
                    x = t,
                    y = yT/nT),name=str(data.index[a_i]))
        fig.line('x','y',source=source,
                             name=f"index {d_ind}  age {d_row['age']:.1f}",tags=[str(d_ind)+'_line'],
                             color=bokeh.palettes.Category20_20[a_i%20])
    return PLT_name,fig

def plot_Tdna_b(model_name,results,ipsd=None,hheight=10,plot_height=200):
    PLT_name = Div(text="""<h1>Total DNA</>""", height=hheight, sizing_mode="stretch_width")
    
    model = results['models'][model_name]
    parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
    model.set_parameters_phy(parameter_phy,mode='bayes')
    results_model = results['results'][model_name]
    data = results['data']    

    T = results_model['c14s_' + model.plot_types[0] ].index   
    ipsd=[]
    for i in T:
        ip = model.calc_implicit_parameters(i)
        ipsd.append(ip)
    fig = figure(title='Total DNA',plot_height=plot_height, sizing_mode="stretch_width")
    fig.xaxis[0].axis_label = 'age'

    

    
    for a_i,(d_ind,d_row) in enumerate(data.iterrows()):
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        ipid.update({n:np.array([v]) for n,v in parameter_phy.items()})
        t = T[T<=d_row['age']]
        nT = np.zeros_like(t)
        yT = np.zeros_like(t)
        for i,pop in enumerate(model.populations):
            yT += ipid[pop][:len(t)] * model.populations_DNA[pop]
        datai = data.iloc[a_i]
        source = ColumnDataSource(data = dict(
                    x = t,
                    y = yT),name=str(data.index[a_i]))
        fig.line('x','y',source=source,
                             name=f"index {d_ind}  age {d_row['age']:.1f}",tags=[str(d_ind)+'_line'],
                             color=bokeh.palettes.Category20_20[a_i%20])
    return PLT_name,fig


def plot_C14intake_b(model_name,results,ipsd=None,ncol=4,plot_width=350,plot_height=200,hheight=10):
    PLT_name = Div(text="""<h1>C14 intake</>""", height=hheight, sizing_mode="stretch_width")
    
    model = results['models'][model_name]
    parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
    model.set_parameters_phy(parameter_phy,mode='bayes')
    results_model = results['results'][model_name]
    data = results['data']    

    T = results_model['c14s_' + model.plot_types[0] ].index   
    ipsd=[]
    for i in T:
        ip = model.calc_implicit_parameters(i)
        ipsd.append(ip)


    sub_figs,sub_figs_g,plt_shape = gen_figs(model.flow_in.keys(),ncol=ncol)
    
    activity = calc_c14intake(model)

    
    for a_i,(d_ind,d_row) in enumerate(data.iterrows()):
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        ipid.update({n:np.array([v]) for n,v in parameter_phy.items()})
        t = T[T<=d_row['age']]
        for i,(ind2,act) in enumerate(activity.items()):
            y = np.zeros_like(t)
            for rate,c14_intake in act.items():
                y += ipid[rate][:len(t)]*c14_intake
            datai = data.iloc[a_i]
            source = ColumnDataSource(data = dict(
                    x = t,
                    y = y),name=str(data.index[a_i]))
            sub_figs[i].line('x','y',source=source,
                             name=f"index {d_ind}  age {d_row['age']:.1f}",tags=[str(d_ind)+'_line'],
                             color=bokeh.palettes.Category20_20[a_i%20])

    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height*plt_shape[1],sizing_mode="stretch_width")






def plot_TC14intake_b(model_name,results,hheight=10,plot_height=200):
    PLT_name = Div(text="""<h1>Total C14 intake</>""", height=hheight, sizing_mode="stretch_width")

    model = results['models'][model_name]
    parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
    model.set_parameters_phy(parameter_phy,mode='bayes')
    results_model = results['results'][model_name]
    data = results['data']    

    T = results_model['c14s_' + model.plot_types[0] ].index   
    ipsd=[]
    for i in T:
        ip = model.calc_implicit_parameters(i)
        ipsd.append(ip)

    fig = figure(title='Total C14 intake',plot_height=plot_height, sizing_mode="stretch_width")
    fig.xaxis[0].axis_label = 'age'

    
    activity = calc_c14intake(model)

    
    for a_i,(d_ind,d_row) in enumerate(data.iterrows()):
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        ipid.update({n:np.array([v]) for n,v in parameter_phy.items()})
        t = T[T<=d_row['age']]
        nT = np.zeros_like(t)
        yT = np.zeros_like(t)
        for i,(ind2,act) in enumerate(activity.items()):
            y = np.zeros_like(t)
            for rate,c14_intake in act.items():
                y += ipid[rate][:len(t)]*c14_intake
            yT += y * ipid[ind2][:len(t)]
            nT +=ipid[ind2][:len(t)] 
        datai = data.iloc[a_i]
        source = ColumnDataSource(data = dict(
                    x = t,
                    y = yT/nT),name=str(data.index[a_i]))
        fig.line('x','y',source=source,
                             name=f"index {d_ind}  age {d_row['age']:.1f}",tags=[str(d_ind)+'_line'],
                             color=bokeh.palettes.Category20_20[a_i%20])
    return PLT_name,fig





def plot_C14sim_b(model_name,results,plot_data=False,ncol=4,plot_width=350,plot_height=400,hheight=10):
    PLT_name = Div(text="""<h1>C14 time</>""", height=hheight, sizing_mode="stretch_width")
    
    model = results['models'][model_name]
    parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
    model.set_parameters_phy(parameter_phy,mode='bayes')
    results_model = results['results'][model_name]
    
    sub_figs,sub_figs_g,plt_shape = gen_figs(model.plot_types,ncol=ncol)
    catmx = np.linspace(1940,2020,500)
    carmy = Catm().lin(catmx)
    for fig in sub_figs:
        if fig is not None:
            fig.line(x=catmx,y=carmy ,name=f"c14atm",tags=['c14atm'], color='black',line_dash='dotted')
    #for a_i,a in enumerate(m.age):
    for a_i,(index,row) in enumerate(results['data'].iterrows()):
        T = results_model['c14s_' + model.plot_types[0] ].index
        t = T[T<=row['age']]
      
        for ii,tt in enumerate(model.plot_types):
            source = ColumnDataSource(data = dict( x = row['Dbirth']+t, y = results_model['c14s_'+tt][index]), name=str(index))
            sub_figs[ii].line('x','y',source=source, name=f"index {index}  age {row['age']:.1f}",tags=[str(index)+'_line'], 
                color=bokeh.palettes.Category20_20[a_i%20])



    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height,sizing_mode="stretch_width")




#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
################  MODELS ##
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################


def plot_M_pop_b(subject,results,ipsd=None,ncol=4,plot_width=350,plot_height=200,hheight=10):
    PLT_name = Div(text="""<h1>Populations</>""", height=hheight, sizing_mode="stretch_width")

    
    '''
    leng = len(allpops)
    if leng>1:
        for i in range(leng-1):
            if np.array([a!=b for a,b in zip(allpops[i],allpops[-1])]).any():
                raise NotImplementedError('populations  differ for different models: not working yet')
    '''

    allpops = list(set().union(*[set(model.populations) for model_name,model in results['models'].items()]))
    pop_map_plot = {pop:i for i,pop in enumerate(allpops)}
    sub_figs,sub_figs_g,plt_shape = gen_figs(allpops,ncol=ncol)
    

    for m_i,(model_name,model) in enumerate(results['models'].items()):
        age = results['data'].loc[subject,'age']
        T = results['results'][model_name]['c14s_' + model.plot_types[0] ].index    
        parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
        model.set_parameters_phy(parameter_phy,mode='bayes')


        
        ipsd=[]
        for i in T:
            ip = model.calc_implicit_parameters(i)
            ipsd.append(ip)
        a_i =  [j for j,x in enumerate(results['data'].index) if x == subject][0]  
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        t = T[T<=age]
        for i,ind2 in enumerate(model.populations):
            datai = results['data'].loc[subject]
            source = ColumnDataSource(data = dict(
                    x = t,
                    y = ipid[ind2][:len(t)]),name=str(model_name))
            sub_figs[pop_map_plot[ind2]].line('x','y',source=source,
                             name=f"model {model_name} index {subject}  age {age:.1f}",tags=[str(model_name)+'_line'],
                             color=bokeh.palettes.Category20_20[m_i%20])

    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height, sizing_mode="stretch_width")

def plot_M_ip_b(subject,results,ipsd=None,ncol=4,plot_width=350,plot_height=200,hheight=10):
    PLT_name = Div(text="""<h1>Implicit Parameter</>""", height=hheight, sizing_mode="stretch_width")

    '''
    alliparas = [model.iparas for model_name,model in results['models'].items()]
    leng = len(alliparas)
    if leng>1:
        for i in range(leng-1):
            if np.array([a!=b for a,b in zip(alliparas[i],alliparas[-1])]).any():
                #raise NotImplementedError('iparas differ for different models: not working yet')
                pass
    '''

    alliparas = list(set().union(*[set(model.iparas) for model_name,model in results['models'].items()]))
    ipara_map_plot = {pop:i for i,pop in enumerate(alliparas)}
    sub_figs,sub_figs_g,plt_shape = gen_figs(alliparas,ncol=ncol)

    for m_i,(model_name,model) in enumerate(results['models'].items()):
        age = results['data'].loc[subject,'age']
        T = results['results'][model_name]['c14s_' + model.plot_types[0]].index    
        parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
        model.set_parameters_phy(parameter_phy,mode='bayes')
        ipsd=[]
        for i in T:
            ip = model.calc_implicit_parameters(i)
            ipsd.append(ip)

        a_i =  [j for j,x in enumerate(results['data'].index) if x == subject][0]  
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        t = T[T<=age]
        for i,ind2 in enumerate(model.iparas):
            datai = results['data'].loc[subject]
            source = ColumnDataSource(data = dict(
                    x = t,
                    y = ipid[ind2][:len(t)]),name=str(model_name))
            sub_figs[ipara_map_plot[ind2]].line('x','y',source=source,
                             name=f"model {model_name} index {subject}  age {age:.1f}",tags=[str(model_name)+'_line'],
                             color=bokeh.palettes.Category20_20[m_i%20])

    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height, sizing_mode="stretch_width")


def plot_M_flow_b(subject,results,ncol=4,plot_width=350,plot_height=200,hheight=10):
    PLT_name = Div(text="""<h1>Flow Parameter</>""", height=hheight, sizing_mode="stretch_width")

    '''
    allflow_in = [list(model.flow_in.keys()) for model_name,model in results['models'].items()]
    leng = len(allflow_in)
    if leng>1:
        for i in range(leng-1):
            if np.array([a!=b for a,b in zip(allflow_in[i],allflow_in[-1])]).any():
                raise NotImplementedError('flow_in differ for different models: not working yet')
    '''

    allflows = list(set().union(*[set(model.flow_in.keys()) for model_name,model in results['models'].items()]))
    flow_map_plot = {pop:i for i,pop in enumerate(allflows)}
    sub_figs,sub_figs_g,plt_shape = gen_figs(allflows,ncol=ncol)


    for m_i,(model_name,model) in enumerate(results['models'].items()):
        age = results['data'].loc[subject,'age']
        T = results['results'][model_name]['c14s_' + model.plot_types[0]].index    
        parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
        model.set_parameters_phy(parameter_phy,mode='bayes')

        ipsd=[]
        for i in T:
            ip = model.calc_implicit_parameters(i)
            ipsd.append(ip)
            
        a_i =  [j for j,x in enumerate(results['data'].index) if x == subject][0]  
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        ipid.update({n:np.array([v]) for n,v in parameter_phy.items()})
        t = T[T<=age]
        for i,(ind2,flows) in enumerate(model.flow_in.items()):
            y = np.zeros_like(t)
            for rate,pop,factor in flows:
                try:
                    if pop is None:
                        y += ipid[rate][:len(t)] * factor
                    else:
                        y += ipid[rate][:len(t)] * factor * ipid[pop][:len(t)]
                except Exception as e:
                    raise Exception([e,Exception(f'in class {model_name}')])
            datai = results['data'].loc[subject]
            source = ColumnDataSource(data = dict(
                    x = t,
                    y = y),name=str(model_name))
            sub_figs[flow_map_plot[ind2] ].line('x','y',source=source,
                             name=f"model {model_name} index {subject}  age {age:.1f}",tags=[str(model_name)+'_line'],
                             color=bokeh.palettes.Category20_20[m_i%20])

    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height, sizing_mode="stretch_width")



def plot_M_act_b(subject,results,ncol=4,plot_width=350,plot_height=200,hheight=10):
    PLT_name = Div(text="""<h1>Activity</>""", height=hheight, sizing_mode="stretch_width")

    '''
    allflow_in = [list(model.flow_in.keys()) for model_name,model in results['models'].items()]
    leng = len(allflow_in)
    if leng>1:
        for i in range(leng-1):
            if np.array([a!=b for a,b in zip(allflow_in[i],allflow_in[-1])]).any():
                raise NotImplementedError('flow_in differ for different models: not working yet')
    '''

    allflows = list(set().union(*[set(model.flow_in.keys()) for model_name,model in results['models'].items()]))
    flow_map_plot = {pop:i for i,pop in enumerate(allflows)}
    sub_figs,sub_figs_g,plt_shape = gen_figs(allflows,ncol=ncol)

    for m_i,(model_name,model) in enumerate(results['models'].items()):
        age = results['data'].loc[subject,'age']
        T = results['results'][model_name]['c14s_' + model.plot_types[0]].index    
        parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
        model.set_parameters_phy(parameter_phy,mode='bayes')

        ipsd=[]
        for i in T:
            ip = model.calc_implicit_parameters(i)
            ipsd.append(ip)

        _,activity = calc_activity(model)


    
        a_i =  [j for j,x in enumerate(results['data'].index) if x == subject][0]  
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        ipid.update({n:np.array([v]) for n,v in parameter_phy.items()})
        t = T[T<=age]
        for i,(ind2,act) in enumerate(activity.items()):
            y = np.zeros_like(t)
            for rate in act:
                y += ipid[rate][:len(t)]
            datai = results['data'].loc[subject]
            source = ColumnDataSource(data = dict(
                    x = t,
                    y = y),name=str(model_name))
            sub_figs[flow_map_plot[ind2]].line('x','y',source=source,
                             name=f"model {model_name} index {subject}  age {age:.1f}",tags=[str(model_name)+'_line'],
                             color=bokeh.palettes.Category20_20[m_i%20])

    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height, sizing_mode="stretch_width")







def plot_M_Tact_b(subject,results,hheight=10,plot_height=200):
    PLT_name = Div(text="""<h1>Overall Activity</>""", height=hheight, sizing_mode="stretch_width")

    fig = figure(title='Overal activity',plot_height=plot_height, sizing_mode="stretch_width")
    fig.xaxis[0].axis_label = 'age'

    for m_i,(model_name,model) in enumerate(results['models'].items()):
        age = results['data'].loc[subject,'age']
        T = results['results'][model_name]['c14s_' + model.plot_types[0]].index    
        parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
        model.set_parameters_phy(parameter_phy,mode='bayes')

        ipsd=[]
        for i in T:
            ip = model.calc_implicit_parameters(i)
            ipsd.append(ip)
        
        _,activity = calc_activity(model)
        

        a_i =  [j for j,x in enumerate(results['data'].index) if x == subject][0]  
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        ipid.update({n:np.array([v]) for n,v in parameter_phy.items()})
        t = T[T<=age]
        nT = np.zeros_like(t)
        yT = np.zeros_like(t)
        for i,(ind2,act) in enumerate(activity.items()):
            y = np.zeros_like(t)
            for rate in act:
                y += ipid[rate][:len(t)]
            yT += y * ipid[ind2][:len(t)]
            nT +=ipid[ind2][:len(t)] 
        datai = results['data'].loc[subject]
        source = ColumnDataSource(data = dict(
                    x = t,
                    y = yT/nT),name=str(model_name))
        fig.line('x','y',source=source,
                             name=f"model {model_name} index {subject}  age {age:.1f}",tags=[str(model_name)+'_line'],
                             color=bokeh.palettes.Category20_20[m_i%20])
    return PLT_name,fig




def plot_M_Tdna_b(subject,results,hheight=10,plot_height=200):
    PLT_name = Div(text="""<h1>Total DNA</>""", height=hheight, sizing_mode="stretch_width")

    fig = figure(title='Total DNA',plot_height=plot_height, sizing_mode="stretch_width")
    fig.xaxis[0].axis_label = 'age'

    for m_i,(model_name,model) in enumerate(results['models'].items()):
        age = results['data'].loc[subject,'age']
        T = results['results'][model_name]['c14s_' + model.plot_types[0]].index    
        parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
        model.set_parameters_phy(parameter_phy,mode='bayes')

        ipsd=[]
        for i in T:
            ip = model.calc_implicit_parameters(i)
            ipsd.append(ip)
        

        a_i =  [j for j,x in enumerate(results['data'].index) if x == subject][0]  
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        ipid.update({n:np.array([v]) for n,v in parameter_phy.items()})
        t = T[T<=age]
        
        yT = np.zeros_like(t)
        for i,pop in enumerate(model.populations):
            yT += ipid[pop][:len(t)]*model.populations_DNA[pop]
        datai = results['data'].loc[subject]
        source = ColumnDataSource(data = dict(
                    x = t,
                    y = yT),name=str(model_name))
        fig.line('x','y',source=source,
                             name=f"model {model_name} index {subject}  age {age:.1f}",tags=[str(model_name)+'_line'],
                             color=bokeh.palettes.Category20_20[m_i%20])
    return PLT_name,fig


def plot_M_C14intake_b(subject,results,ncol=4,plot_width=350,plot_height=200,hheight=10):
    PLT_name = Div(text="""<h1>C14 intake</>""", height=hheight, sizing_mode="stretch_width")



    allpops = list(set().union(*[set(model.populations) for model_name,model in results['models'].items()]))
    pop_map_plot = {pop:i for i,pop in enumerate(allpops)}
    sub_figs,sub_figs_g,plt_shape = gen_figs(allpops,ncol=ncol)
    
    #sub_figs,sub_figs_g,plt_shape = gen_figs(list(results['models'].items())[0][1].populations,ncol=ncol)

    for m_i,(model_name,model) in enumerate(results['models'].items()):
        age = results['data'].loc[subject,'age']
        T = results['results'][model_name]['c14s_'  + model.plot_types[0]].index    
        parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
        model.set_parameters_phy(parameter_phy,mode='bayes')
        
        ipsd=[]
        for i in T:
            ip = model.calc_implicit_parameters(i)
            ipsd.append(ip)

        
        activity = calc_c14intake(model)

    
        a_i =  [j for j,x in enumerate(results['data'].index) if x == subject][0]  
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        ipid.update({n:np.array([v]) for n,v in parameter_phy.items()})
        t = T[T<=results['data'].loc[subject,'age']]
        for i,(ind2,act) in enumerate(activity.items()):
            y = np.zeros_like(t)
            for rate,c14_intake in act.items():
                y += ipid[rate][:len(t)]*c14_intake
            source = ColumnDataSource(data = dict(
                    x = t,
                    y = y),name=str(model_name))
            sub_figs[pop_map_plot[ind2]].line('x','y',source=source,
                             name=f"model {model_name} index {subject}  age {age:.1f}",tags=[str(model_name)+'_line'],
                             color=bokeh.palettes.Category20_20[m_i%20])

    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height, sizing_mode="stretch_width")






def plot_M_TC14intake_b(subject,results,hheight=10,plot_height=200):
    PLT_name = Div(text="""<h1>Total C14 intake</>""", height=hheight, sizing_mode="stretch_width")
    
    fig = figure(title='Total C14 intake',plot_height=plot_height, sizing_mode="stretch_width")
    fig.xaxis[0].axis_label = 'age'
    
    for i_m,(model_name,model) in enumerate(results['models'].items()):
        age = results['data'].loc[subject,'age']
        T = results['results'][model_name]['c14s_' + model.plot_types[0]].index    
        parameter_phy =  {i:results['point_est'].loc[model_name].loc[('median',i)] for i in model.parameter_names}
        model.set_parameters_phy(parameter_phy,mode='bayes')
        
        ipsd=[]
        for i in T:
            ip = model.calc_implicit_parameters(i)
            ipsd.append(ip)

        
        activity = calc_c14intake(model)

    
        #for a_i,a in enumerate(m.age):
        a_i =  [j for j,x in enumerate(results['data'].index) if x == subject][0]  
        ipid = listofdict_to_dictofarray_f(ipsd,a_i)
        ipid.update({n:np.array([v]) for n,v in parameter_phy.items()})
        t = T[T<=results['data'].loc[subject,'age']]
        nT = np.zeros_like(t)
        yT = np.zeros_like(t)
        for i,(ind2,act) in enumerate(activity.items()):
            y = np.zeros_like(t)
            for rate,c14_intake in act.items():
                y += ipid[rate][:len(t)]*c14_intake
            yT += y * ipid[ind2][:len(t)]
            nT +=ipid[ind2][:len(t)] 
        source = ColumnDataSource(data = dict(
                    x = t,
                    y = yT/nT),name=str(model_name))
        fig.line('x','y',source=source,
                             name=f"model {model_name} index {subject}  age {age:.1f}",tags=[str(model_name)+'_line'],
                             color=bokeh.palettes.Category20_20[i_m%20])
    return PLT_name,fig


def plot_M_C14sim_b(subject,results,ncol=4,plot_width=350,plot_height=400,hheight=10):
    PLT_name = Div(text="""<h1>C14 time</>""", height=hheight, sizing_mode="stretch_width")  

       
    plot_types,plot_dict = get_plot_types(results)
    if ncol is None:
        ncol = len(plot_types)

    
    sub_figs,sub_figs_g,plt_shape = gen_figs(plot_types,ncol=ncol)
    catmx = np.linspace(1940,2020,500)
    carmy = Catm().lin(catmx)
    for fig in sub_figs:
        if fig is not None:
            fig.line(x=catmx,y=carmy ,name=f"c14atm",tags=['c14atm'], color='black',line_dash='dotted')
    
    #for a_i,a in enumerate(m.age):
    
    results_all = results['results']

    for i,(model_name,model) in enumerate(results['models'].items()):
        row = results['data'].loc[subject]
        results_ind = results_all[model_name]

        T = results_ind['c14s_'+model.plot_types[0]].index             
        t = T[T<=row['age']]
      
        for tt in model.plot_types:
            i = np.where(plot_types==tt)[0][0]
            source = ColumnDataSource(data = dict( x = row['Dbirth']+t, y = results_ind['c14s_'+tt][subject]), name=str(model_name))
            sub_figs[i].line('x','y',source=source, name=f"model {model_name} index {subject}  age {row['age']:.1f}",tags=[str(model_name)+'_line'], 
            color=bokeh.palettes.Category20_20[i%20])
        

    
    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height*plt_shape[1], sizing_mode="stretch_width")


from bokeh.models import Range1d

from scipy import interpolate
import pandas as pd
def plot_deltaM_C14sim_b(results,X=None,Mtype=None,ncol=4,plot_width=350,plot_height=400,hheight=10):
    PLT_name = Div(text="""<h1>C14 delta</>""", height=hheight, sizing_mode="stretch_width")  


    data = results['data']
    if Mtype is None:
        M_types = ['none']
    else:
        M_types = np.unique(data[Mtype]) 
    sub_figs,sub_figs_g,plt_shape = gen_figs([j+k for k in M_types for j in ['','Delta ']  ],ncol=ncol,tooltips=TOOLTIPS2)
    
    catmx = np.linspace(1930,2020,500)
    carmy = Catm().lin(catmx)
    for fig_i in sub_figs[::2]:
        fig_i.line(x=catmx,y=carmy ,name=f"c14atm",tags=['c14atm'], color='black',line_dash='dotted')


    
    
    dymin = 999
    dymax=-999
    if X is None:
        X = data['Dbirth']
    
    for i,(model_name,model) in enumerate(results['models'].items()):
        for i_mt,mt in enumerate(M_types):
            if mt == 'none':
                mask = np.ones_like(data['d14C'],dtype=bool)
            else:
                mask = data[Mtype] == mt
            results_m = results['results'][model_name]
            fig_d = sub_figs[2*i_mt]
            fig_Delta = sub_figs[1+2*i_mt]
           
            Xm = X[mask]
            Y = results_m['c14_M'][mask]
            YD = data['d14C'][mask]
            source = ColumnDataSource(data = dict( x =Xm, y = Y,y2=Y-YD,ind=data.index[mask])
                )
            fig_d.circle("x","y",source=source,name=f"model {model_name}", tags=[str(model_name)+'_line'],
                color=bokeh.palettes.Category20_20[i%20],legend_label=model_name)

            fig_Delta.circle("x","y2",source=source, name=f"model {model_name}",tags=[str(model_name)+'_line'],
                color=bokeh.palettes.Category20_20[i%20],legend_label=model_name)


    delta_val = {i :{tt :dict(x=[],y=[]) for tt in M_types } for i in results['models'].keys()}
    
    for subject,row in data.iterrows():
        if len(M_types) == 1 and M_types[0] == 'none':
            fig_d = sub_figs[0]
            fig_Delta = sub_figs[1]
            mt = M_types[0]
        else:
            mt = row[Mtype]
            fig_d = sub_figs[2*np.where(np.array(M_types)==row[Mtype])[0][0]]
            fig_Delta = sub_figs[1+2*np.where(np.array(M_types)==row[Mtype])[0][0]]
        fig_d.scatter(x=X.loc[subject],y=row['d14C'],marker='x',name=f"measurement {subject}",color='black',size=10,legend_label="measurment")
        for i,(model_name,model) in enumerate(results['models'].items()):
            results_m = results['results'][model_name]
            delta_val[model_name][mt]['x'].append(X.loc[subject])
            delta_val[model_name][mt]['y'].append(results_m['c14_M'].loc[subject]-row['d14C'])
            dymin =min(dymin,results_m['c14_M'].loc[subject]-row['d14C'])
            dymax =max(dymax,results_m['c14_M'].loc[subject]-row['d14C'])
    for i,mt in enumerate(M_types):
        fig_D = sub_figs[1+2*i]
        for i,(model_name,model) in enumerate(results['models'].items()):        
            dd = pd.Series(delta_val[model_name][mt]['y'],index=delta_val[model_name][mt]['x']).sort_index()
            Xm = dd.index
            Ym = dd.values
            if len(Xm)<4:
                continue
            f2 = interpolate.UnivariateSpline(Xm, Ym)
            xnew = np.linspace(Xm[0], Xm[-1], 1000)
            fig_D.line(x=xnew,y=f2(xnew), name=f"model {model_name}",tags=[str(model_name)+'_line'],
                    color=bokeh.palettes.Category20_20[i%20])
    


    for i,fig_i in enumerate(sub_figs):
        fig_i.x_range=Range1d(np.min(X)-5, np.max(X)+10)
        if i%2!=0:
            fig_i.y_range=Range1d(dymin-(dymax-dymin)*0.1, dymax+(dymax-dymin)*0.1)
    for p in sub_figs:
        p.legend.click_policy="hide"
    
    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height, sizing_mode="stretch_width")


def plot_delta_C14sim_b(results,ncol=4,plot_width=350,plot_height=400,hheight=10):
    PLT_name = Div(text="""<h1>C14 time</>""", height=hheight, sizing_mode="stretch_width")  


    
    measurment_types = get_measurment_types(results)
    sub_figs,sub_figs_g,plt_shape = gen_figs([j+i for i in measurment_types for j in ['','Delta ']],ncol=ncol,tooltips=TOOLTIPS2)
    
    catmx = np.linspace(1930,2020,500)
    carmy = Catm().lin(catmx)
    for fig_i in sub_figs[::2]:
        fig_i.line(x=catmx,y=carmy ,name=f"c14atm",tags=['c14atm'], color='black',line_dash='dotted')


    
    data = results['data']
    dymin = 999
    dymax=-999
    
    
    for i,(model_name,model) in enumerate(results['models'].items()):
        results_m = results['results'][model_name]
        for j,tt in enumerate(model.plot_types):

            fig_d = sub_figs[2*np.where(np.array(measurment_types)==tt)[0][0]]
            fig_Delta = sub_figs[1+2*np.where(np.array(measurment_types)==tt)[0][0]]
            if tt == 'single':
                mask = data.index
                dd = 'd14C_single'
            else:
                mask = data.index[data['measurment_types'] == tt]
                dd = 'd14C'
            X = data['Dbirth'].loc[mask]
            Y = results_m['c14m_' + tt][mask]
            YD = data[dd].loc[mask]
            source = ColumnDataSource(data = dict( x =X, y = Y,y2=Y-YD,ind=data.index)
                )
            fig_d.circle("x","y",source=source,name=f"model {model_name}", tags=[str(model_name)+'_line'],
                color=bokeh.palettes.Category20_20[i%20],legend_label=model_name)

            fig_Delta.circle("x","y2",source=source, name=f"model {model_name}",tags=[str(model_name)+'_line'],
                color=bokeh.palettes.Category20_20[i%20],legend_label=model_name)


    delta_val = {i :{tt :dict(x=[],y=[]) for tt in measurment_types } for i in results['models'].keys()}


    if 'single' in measurment_types:
        fig = sub_figs[2*np.where(np.array(measurment_types)=='single')[0][0]]
        fig.scatter(x=data['Dbirth'],y=data['d14C_single'],marker='x',name=f"measurement",color='black',size=10,legend_label="measurment")

    
    for subject,row in data.iterrows():

        
        fig_d = sub_figs[2*np.where(np.array(measurment_types)==row['measurment_types'])[0][0]]
        fig_Delta = sub_figs[1+2*np.where(np.array(measurment_types)==row['measurment_types'])[0][0]]
        fig_d.scatter(x=row['Dbirth'],y=row['d14C'],marker='x',name=f"measurement {subject}",color='black',size=10,legend_label="measurment")
        for i,(model_name,model) in enumerate(results['models'].items()):
            results_m = results['results'][model_name]
            if model.plot_types == ['single']:
                tt = 'single'
                dd = 'd14C_single'
            else:
                tt = row['measurment_types']
                dd = 'd14C'
            delta_val[model_name][tt]['x'].append(row['Dbirth'])
            delta_val[model_name][tt]['y'].append(results_m['c14m_' + tt].loc[subject]-row[dd])
            dymin =min(dymin,results_m['c14m_' + tt].loc[subject]-row[dd])
            dymax =max(dymax,results_m['c14m_' + tt].loc[subject]-row[dd])
    for i,measurment_type in enumerate(measurment_types):
        fig_D = sub_figs[1+2*i]
        for i,(model_name,model) in enumerate(results['models'].items()):        
            dd = pd.Series(delta_val[model_name][measurment_type]['y'],index=delta_val[model_name][measurment_type]['x']).sort_index()
            X = dd.index
            Y = dd.values
            if len(X)<2:
                continue
            f2 = interpolate.UnivariateSpline(X, Y)
            xnew = np.linspace(X[0], X[-1], 1000)
            fig_D.line(x=xnew,y=f2(xnew), name=f"model {model_name}",tags=[str(model_name)+'_line'],
                    color=bokeh.palettes.Category20_20[i%20])
        
    for i,fig_i in enumerate(sub_figs):
        fig_i.x_range=Range1d(catmx[0], catmx[-1]+20)
        if i%2!=0:
            fig_i.y_range=Range1d(dymin-(dymax-dymin)*0.1, dymax+(dymax-dymin)*0.1)
    for p in sub_figs:
        p.legend.click_policy="hide"
    
    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height, sizing_mode="stretch_width")


def plot_delta_C14sim_agediag_b(results,agediag='agediag',ncol=4,plot_width=350,plot_height=400,hheight=10):
    PLT_name = Div(text="""<h1>C14 time</>""", height=hheight, sizing_mode="stretch_width")  

    measurment_types = get_measurment_types(results)
    sub_figs,sub_figs_g,plt_shape = gen_figs([j+i for i in measurment_types for j in ['','Delta ']],ncol=ncol ,tooltips=TOOLTIPS2)
    
    
    data = results['data']
    dymin = 999
    dymax=-999


    for i,(model_name,model) in enumerate(results['models'].items()):
        results_m = results['results'][model_name]
        for j,tt in enumerate(model.plot_types):
            fig_d = sub_figs[2*j]
            fig_Delta = sub_figs[1+2*j]
            mask = data.index[data['measurment_types'] == tt]


            X = data['age'].loc[mask] - data[agediag].loc[mask]
            Y = results_m['c14m_' + tt][mask]
            YD = data['d14C'].loc[mask]
            source = ColumnDataSource(data = dict( x =X, y = Y,y2=Y-YD,ind=data.index)
                )
            fig_d.circle("x","y",source=source,name=f"model {model_name}", tags=[str(model_name)+'_line'],
                color=bokeh.palettes.Category20_20[i%20],legend_label=model_name)

            fig_Delta.circle("x","y2",source=source, name=f"model {model_name}",tags=[str(model_name)+'_line'],
                color=bokeh.palettes.Category20_20[i%20],legend_label=model_name)

        '''
        results_m = results['results'][model_name]
        X = data['age'] - data[agediag]
        source = ColumnDataSource(data = dict( x = X, y = results_m['c14m_t']), name=str(model_name))
        sub_figs[0].circle('x','y',source=source, name=f"model {model_name}",tags=[str(model_name)+'_line'],legend_label=model_name,
            color=bokeh.palettes.Category20_20[i%20])
        
        source = ColumnDataSource(data = dict( x = X, y = results_m['c14m_2n']), name=str(model_name))
        sub_figs[2].circle('x','y',source=source, name=f"model {model_name}",tags=[str(model_name)+'_line'],legend_label=model_name,
            color=bokeh.palettes.Category20_20[i%20])


        source = ColumnDataSource(data = dict( x = X, y = results_m['c14m_4n']), name=str(model_name))
        sub_figs[4].circle('x','y',source=source, name=f"model {model_name}",tags=[str(model_name)+'_line'],legend_label=model_name,
            color=bokeh.palettes.Category20_20[i%20])

        source = ColumnDataSource(data = dict( x = X, y = results_m['c14m_pn']), name=str(model_name))
        sub_figs[6].circle('x','y',source=source, name=f"model {model_name}",tags=[str(model_name)+'_line'],legend_label=model_name,
            color=bokeh.palettes.Category20_20[i%20])
        '''
    delta_val = {i :{tt :dict(x=[],y=[]) for tt in measurment_types } for i in results['models'].keys()}

    for subject,row in data.iterrows():
        fig_d = sub_figs[2*np.where(np.array(measurment_types)==row['measurment_types'])[0][0]]       
        X = row['age'] - row[agediag]
        fig_d.scatter(x=X,y=row['d14C'],marker='x',name=f"measurement {subject}",color='black',size=10,legend_label="measurment")
        for i,(model_name,model) in enumerate(results['models'].items()):
            results_m = results['results'][model_name]

            delta_val[model_name][row['measurment_types']]['x'].append(X)
            delta_val[model_name][row['measurment_types']]['y'].append(results_m['c14m_' + row['measurment_types']].loc[subject]-row['d14C'])
            dymin =min(dymin,results_m['c14m_' + row['measurment_types']].loc[subject]-row['d14C'])
            dymax =max(dymax,results_m['c14m_' + row['measurment_types']].loc[subject]-row['d14C'])
    for i,measurment_type in enumerate(measurment_types):
        fig_D = sub_figs[1+2*i]
        for i,(model_name,model) in enumerate(results['models'].items()):        
            dd = pd.Series(delta_val[model_name][measurment_type]['y'],index=delta_val[model_name][measurment_type]['x']).sort_index()
            X = dd.index
            Y = dd.values
            if len(X)<2:
                continue
            f2 = interpolate.UnivariateSpline(X, Y)
            xnew = np.linspace(X[0], X[-1], 1000)
            fig_D.line(x=xnew,y=f2(xnew), name=f"model {model_name}",tags=[str(model_name)+'_line'],
                    color=bokeh.palettes.Category20_20[i%20])
        
    for i,fig_i in enumerate(sub_figs):
        if i%2!=0:
            fig_i.y_range=Range1d(dymin-(dymax-dymin)*0.1, dymax+(dymax-dymin)*0.1)
    for p in sub_figs:
        p.legend.click_policy="hide"

        

    
    return PLT_name,gridplot(sub_figs_g, plot_width=plot_width, plot_height=plot_height, sizing_mode="stretch_width")