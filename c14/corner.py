import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from .tools import *
import seaborn as sb
import pandas as pd
import warnings

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
    N = np.sum(cc.names == '0.683')
    conts=[]
    for i in range(N):
        conts.append(cc[i])
    return conts

def corner_R(chain,parameter_names=None,burnin=0,point_estimate=None,rename=None,kde=True,logparas=[],unitlog=""):
    leng = chain.shape[2]
    chainf = chain[:,burnin:,:].reshape(-1,leng)
    #stolen from corner.py
    # Some magic numbers for pretty axis layout.
    factor = 1.3           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.5        # w/hspace size
    plotdim = factor * leng + factor * (leng - 1.) * whspace
    dim = lbdim + plotdim + trdim
    fig, axes = plt.subplots(leng, leng, figsize=(dim, dim ))
    axes = np.array(fig.axes).reshape((leng, leng))
    cmap = plt.cm.viridis
    cmap.set_under('red')
    if parameter_names is None:
        parameter_names = [i for i in range(leng)]
    for iy,namey in enumerate(parameter_names):
        for ix,namex in enumerate(parameter_names):
            ax = axes[iy,ix]
            if iy==ix:
                sb.distplot(chainf[:,ix],ax=ax,kde=False)
                if rename is not None:
                    ax.set_xlabel( rename[namex])
                else:
                    ax.set_xlabel(namex)
                ax.set_ylabel('')
                ax.set_yticks( [])
                if point_estimate is not None:
                    yval = np.mean(ax.get_ylim())
                    ax.scatter(point_estimate[namex],yval,marker='X',s=100,color='red',ec='black')
                if namex in logparas:
                    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3,integer=True))
                    ticks = ax.get_xticks()
                    ax.set_xticklabels([r"$10^{"+str(int(i))+r"}$"  for i in ticks])
                    label = ax.get_xlabel()
                    ax.set_xlabel(label+" "+unitlog)
            if iy<ix:
                ax.remove()
            if iy>ix:
                z = r_kde(chainf[:,[ix,iy]])
                idx = z.argsort()
                sort_chain = chainf[:,[ix,iy]][idx]
                ax.scatter(sort_chain[:,0],sort_chain[:,1], c=z[idx],cmap=cmap , s=0.1, edgecolor=None)
                for i in ci2d(chainf[:,[ix,iy]]):
                    ax.plot(i.x,i.y,color='red')
                if rename is not None:
                    ax.set_ylabel( rename[namey])
                else:
                    ax.set_ylabel(namey)
                if rename is not None:
                    ax.set_xlabel( rename[namex])
                else:
                    ax.set_xlabel(namex)
                if namey in logparas:
                    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5,integer=True))
                    ticks = ax.get_yticks()
                    ax.set_yticklabels([r"$10^{"+str(int(i))+r"}$"  for i in ticks])
                    label = ax.get_ylabel()
                    ax.set_ylabel(label+" "+unitlog)
                if namex in logparas:
                    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3,integer=True))
                    ticks = ax.get_xticks()
                    ax.set_xticklabels([r"$10^{"+str(int(i))+r"}$"  for i in ticks])
                    label = ax.get_xlabel()
                    ax.set_xlabel(label+" "+unitlog)
                if point_estimate is not None:
                    ax.scatter(point_estimate[namex],point_estimate[namey],marker='X',s=100,color='red',ec='black')
    plt.tight_layout()



