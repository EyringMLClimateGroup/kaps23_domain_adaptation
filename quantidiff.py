"""quantifies differences induced by COSP/our DA"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import sys
import tensorflow as tf

if __name__=="__main__":
    work = os.environ["WORK"]
    scratch = os.environ["SCR"]
    ls = glob.glob(os.path.join(work, "pickle/diff*pkl"))
    #from cosp_vs_native.py
    cosp = dict(np.load(os.path.join(work, "cosp_vs_native.npz")))
    if len(sys.argv)>1:
        name = sys.argv[1]
    else:
        name =""
    for file in ls:
        if name in file:
            #dateframes are avaialble for each CM domain and the respective synth observations
            df = pd.read_pickle(file)
            if df.shape[1]==18:    
                df_orig = df.iloc[:,-9:]
                df = df.iloc[:,:-9]
                
            else:
                raise NotImplementedError("whaddayawant")

            fig=plt.figure(figsize=(10,10))#,ax = plt.subplots(3,3, figsize=(10,10), sharex=True)
            df.plot(use_index=True,  kind = "bar", ax=plt.gca(),sharex=True,subplots=True,layout=(3,3))
            for a in fig.axes:
                
                a.set_xticklabels(np.linspace(-1,1,19).round(2))#, labels=np.arange(-1,1,20))
            ids = (df_orig.index.values-min(df_orig.index.values))*10
            for j,bla in enumerate(fig.axes):
                bla.bar(x=ids,height=df_orig.iloc[:,j],alpha=0.5)
                

            fig.savefig(file.replace("pickle","plots/quantidiff").replace("pkl","png")) 
            numplots = len(cosp)+1
            square = int(np.ceil(np.sqrt(numplots)))
            fig,ax=plt.subplots(square,square,figsize=(15,15),sharex=True)
            fig2,ax2 = plt.subplots(figsize=(8,8))
            ax=ax.flatten()
            models = [x for x in cosp.keys()]
            all_y = np.zeros(19)
            for i in range(square**2):
                if i<len(cosp):
                    y=cosp[models[i]].astype(float)
                    y/=np.sum(y)
                    ax[i].bar(x=np.linspace(-1,1,19),width=0.09, height=y)
                    ax[i].set_title(models[i][:-9],fontsize=22)
                    ax[i].tick_params(labelsize=20)
                    all_y +=y
                elif i==len(cosp):
                    clt = df.clt.values.astype(float)
                    clt/=np.sum(clt)
                    ax[i].bar(x=np.linspace(-1,1,19),height=clt,width=0.09)
                    ax[i].set_title("Generative DA",fontsize=22)
                    ax[i].tick_params(labelsize=20)
                    ax2.bar(x=np.linspace(-1,1,19),height=clt,width=0.06, label = "generative DA")
                    ax2.set_title(r"$clt_{synth}-clt_{native}$",fontsize=22)
                    ax2.tick_params(labelsize=20)
                else:
                    ax[i].remove()
            fig.tight_layout()
            fig.savefig(os.path.join(work,"stats/{}_vs_cosp.pdf".format(name)))
            all_y/= np.sum(all_y)
            ax2.bar(x=np.linspace(-1,1,19)+0.035,width=0.06, height=all_y,label="COSP")
            ax2.legend(fontsize=21, loc="upper left")
            fig2.savefig(os.path.join(work, "stats/{}_vs_allcosp.pdf".format(name)))
            