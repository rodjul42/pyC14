import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from .tools import *
import seaborn as sb
import pandas as pd
import warnings
import matplotlib.ticker
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()


rstringa="""
    function(chainf){
        require("gplots")
        a<-ci2d(chainf[,1],chainf[,2], show='none',ci.levels=c(0.683))
        a$contours
    }
"""
rstringb="""
    function(chainf){
        require("ks")
        b <- ks::kde(chainf,eval.points=chainf[,c(1,2)])
        b$estimate
    }
"""
        
r_ci2d=robjects.r(rstringa)
r_kde=robjects.r(rstringb)
def ci2d(chainf):
    cc=r_ci2d(chainf)
    conts=[]
    for i,v in cc.items():
        if i=='0.683':
            conts.append(v)
    return conts

def corner_R(chain,parameter_names=None,point_estimate=None,rename=None,kde=True,logparas=[],unitlog="",rasterized=False,mindim=None,axes=None,remove_axis=True):
    leng = chain.shape[2]
    chainf = chain.reshape(-1,leng)
    
    if axes is None:
        #stolen from corner.py
        # Some magic numbers for pretty axis layout.
        factor = 1.3           # size of one side of one panel
        lbdim = 0.5 * factor   # size of left/bottom margin
        trdim = 0.2 * factor   # size of top/right margin
        whspace = 0.5        # w/hspace size
        plotdim = factor * leng + factor * (leng - 1.) * whspace
        dim = lbdim + plotdim + trdim
        if mindim is None:
            fig, axes = plt.subplots(leng, leng, figsize=(dim, dim ))
            delta=None
        else:
            delta = max(mindim,dim)-dim 
            fig, axes = plt.subplots(leng, leng, figsize=(max(mindim,dim),dim ))
        axes = np.array(fig.axes).reshape((leng, leng))
    else:
        delta = None
    cmap = plt.cm.viridis
    if parameter_names is None:
        parameter_names = [i for i in range(leng)]
    for iy,namey in enumerate(parameter_names):
        for ix,namex in enumerate(parameter_names):
            ax = axes[iy,ix]
            if iy==ix:
                sb.histplot(chainf[:,ix],ax=ax,kde=False,element='step',fill=True)#,hist_kws={"histtype": "stepfilled"})
                if rename is not None:
                    ax.set_xlabel( rename[namex])
                else:
                    ax.set_xlabel(namex)
                ax.set_ylabel('')
                ax.set_yticks( [])
                if namex in logparas:
                    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
                    ticks = ax.get_xticks()
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([r"$10^{"+str(int(i))+r"}$"  for i in ticks])
                    label = ax.get_xlabel()
                    ax.set_xlabel(label+" "+unitlog)
                if point_estimate is not None:
                    yval = np.mean(ax.get_ylim())
                    ax.scatter(point_estimate[namex],yval,marker='X',s=70,color='red',ec='black',zorder=99)
            if iy<ix:
                if remove_axis:
                    ax.remove()
            if iy>ix:
                try:
                    z = r_kde(chainf[:,[ix,iy]])
                    idx = z.argsort()
                    sort_chain = chainf[:,[ix,iy]][idx]
                    ax.scatter(sort_chain[:,0],sort_chain[:,1], c=z[idx],cmap=cmap , s=0.1, edgecolor=None,rasterized=rasterized)
                    for i in ci2d(chainf[:,[ix,iy]]):
                        ax.plot(i.x,i.y,color='red')
                except:
                    pass

                if rename is not None:
                    ax.set_ylabel( rename[namey])
                else:
                    ax.set_ylabel(namey)
                if rename is not None:
                    ax.set_xlabel( rename[namex])
                else:
                    ax.set_xlabel(namex)
                if namey in logparas:
                    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
                    ticks = ax.get_yticks()
                    ax.set_yticks(ticks)
                    ax.set_yticklabels([r"$10^{"+str(int(i))+r"}$"  for i in ticks])
                    label = ax.get_ylabel()
                    ax.set_ylabel(label+" "+unitlog)
                if namex in logparas:
                    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
                    ticks = ax.get_xticks()
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([r"$10^{"+str(int(i))+r"}$"  for i in ticks])
                    label = ax.get_xlabel()
                    ax.set_xlabel(label+" "+unitlog)
                if point_estimate is not None:
                    ax.scatter(point_estimate[namex],point_estimate[namey],marker='X',s=70,color='red',ec='black')
                

    
    if delta is not None:
        if delta>0.1:
            for i in axes:
                for j in i:
                    j.set_aspect(1/j.get_data_ratio(), 'box')
            plt.tight_layout()
            plt.subplots_adjust(left=delta/(2*mindim),right=1-delta/(2*mindim) )
        else:
            plt.tight_layout()
    else:
        plt.tight_layout()

def corner_fast(chain,parameter_names=None,point_estimate=None,rename=None,kde=True,logparas=[],unitlog="",rasterized=False,mindim=None,axes=None,remove_axis=True):
    leng = chain.shape[2]
    chainf = chain.reshape(-1,leng)
    
    if axes is None:
        #stolen from corner.py
        # Some magic numbers for pretty axis layout.
        factor = 1.3           # size of one side of one panel
        lbdim = 0.5 * factor   # size of left/bottom margin
        trdim = 0.2 * factor   # size of top/right margin
        whspace = 0.5        # w/hspace size
        plotdim = factor * leng + factor * (leng - 1.) * whspace
        dim = lbdim + plotdim + trdim
        if mindim is None:
            fig, axes = plt.subplots(leng, leng, figsize=(dim, dim ))
            delta=None
        else:
            delta = max(mindim,dim)-dim 
            fig, axes = plt.subplots(leng, leng, figsize=(max(mindim,dim),dim ))
        axes = np.array(fig.axes).reshape((leng, leng))
    else:
        delta = None
    cmap = plt.cm.viridis
    if parameter_names is None:
        parameter_names = [i for i in range(leng)]
    for iy,namey in enumerate(parameter_names):
        for ix,namex in enumerate(parameter_names):
            ax = axes[iy,ix]
            if iy==ix:
                sb.histplot(chainf[:,ix],ax=ax,kde=False,element='step',fill=True)#,hist_kws={"histtype": "stepfilled"})
                if rename is not None:
                    ax.set_xlabel( rename[namex])
                else:
                    ax.set_xlabel(namex)
                ax.set_ylabel('')
                ax.set_yticks( [])
                if namex in logparas:
                    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
                    ticks = ax.get_xticks()
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([r"$10^{"+str(int(i))+r"}$"  for i in ticks])
                    label = ax.get_xlabel()
                    ax.set_xlabel(label+" "+unitlog)
                if point_estimate is not None:
                    yval = np.mean(ax.get_ylim())
                    ax.scatter(point_estimate[namex],yval,marker='X',s=70,color='red',ec='black',zorder=99)
            if iy<ix:
                if remove_axis:
                    ax.remove()
            if iy>ix:
                try:
                    sort_chain = chainf[:,[ix,iy]]
                    ax.scatter(sort_chain[:,0],sort_chain[:,1],cmap=cmap , s=0.1, edgecolor=None,rasterized=rasterized)
                except:
                    pass

                if rename is not None:
                    ax.set_ylabel( rename[namey])
                else:
                    ax.set_ylabel(namey)
                if rename is not None:
                    ax.set_xlabel( rename[namex])
                else:
                    ax.set_xlabel(namex)
                if namey in logparas:
                    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
                    ticks = ax.get_yticks()
                    ax.set_yticks(ticks)
                    ax.set_yticklabels([r"$10^{"+str(int(i))+r"}$"  for i in ticks])
                    label = ax.get_ylabel()
                    ax.set_ylabel(label+" "+unitlog)
                if namex in logparas:
                    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
                    ticks = ax.get_xticks()
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([r"$10^{"+str(int(i))+r"}$"  for i in ticks])
                    label = ax.get_xlabel()
                    ax.set_xlabel(label+" "+unitlog)
                if point_estimate is not None:
                    ax.scatter(point_estimate[namex],point_estimate[namey],marker='X',s=70,color='red',ec='black')
                

    
    if delta is not None:
        if delta>0.1:
            for i in axes:
                for j in i:
                    j.set_aspect(1/j.get_data_ratio(), 'box')
            plt.tight_layout()
            plt.subplots_adjust(left=delta/(2*mindim),right=1-delta/(2*mindim) )
        else:
            plt.tight_layout()
    else:
        plt.tight_layout()


def corner_Prior_R(chain,prior,parameter_names=None,point_estimate=None,rename=None,kde=True,logparas=[],unitlog="",rasterized=False):
    leng = chain.shape[2]
    #stolen from corner.py
    # Some magic numbers for pretty axis layout.
    factor = 1.3           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.5        # w/hspace size
    plotdim = factor * leng + factor * (leng - 1.) * whspace
    dim = lbdim + plotdim + trdim
    fig, axes = plt.subplots(leng, leng+1, figsize=(dim + 1, dim))
    axes = np.array(fig.axes).reshape((leng, leng+1))
    corner_R(chain,remove_axis=False,parameter_names=parameter_names,point_estimate=point_estimate,rename=rename,kde=kde,logparas=logparas,unitlog=unitlog,rasterized=rasterized,mindim=None,axes=axes)
    corner_R(prior[:,:,::-1],axes=axes[::-1,::-1],remove_axis=False,parameter_names=parameter_names,point_estimate=point_estimate,rename=rename,kde=kde,logparas=logparas,unitlog=unitlog,rasterized=rasterized,mindim=None)
    for i in range(leng):
        ax= axes[i,i+1]
        ax.set_xlabel("Prior " + ax.get_xlabel() )
        for j in range(1,leng - i):
            ax= axes[i,i+1+j]
            ax.set_ylabel("Prior  " + ax.get_ylabel() )
            ax.set_xlabel("Prior  " + ax.get_xlabel() )
    plt.tight_layout()
    return
