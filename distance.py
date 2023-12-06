"""computes various distances between the result distributions
   this script isnt particularly readable but not actually that complicated"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import scipy as sp
from itertools import product,combinations
import ot # py optimal transport
import pandas as pd
import multiprocessing as mlp
import sys
import warnings


def KLD(dist1, dist2):
    """kullback leibler divergence"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return np.sum([ x for x in dist1*np.log(dist1/dist2) if not np.isnan(x)])

def JSD(df):
    """computes jensen shannon divergence
    df containing dist1,dist2 univariate probability distributions that sum to 1"""
    assert df.shape[1]==2,df.shape
    dist1 = df.values[:,0]
    dist2 = df.values[:,1]

    assert np.allclose(np.sum(dist1),1.),np.sum(dist1)
    assert np.allclose(np.sum(dist2),1.), np.sum(dist2)
    mixture = (dist1+dist2)/2
    kld1 = KLD(dist1,mixture)
    kld2 = KLD(dist2,mixture)
    return (kld1+kld2)/2


def JSD_df(sourcefile):
    """gets the JSD from the pickles saved in apply_to_icon

    Args:
        sourcefile (string): path to the source pickle

    Returns:
        list: jsd improvements for each variable
    """    
    basename = os.path.basename(sourcefile)#ESACCI
    predfile = os.path.join(work,"pickle",basename.replace("source_","pred_"))#fake observations
    targetfile = os.path.join(work,"pickle",basename.replace("source_","target_"))#ICON
    df1 = pd.read_pickle(sourcefile)
    df2 = pd.read_pickle(predfile)
    df3 = pd.read_pickle(targetfile)
    #JSD works on distributions, approximated here with a histogram
    d1_hist = [df1.iloc[:,i].round(3).value_counts( normalize=True).to_frame("source") for i in range(df1.shape[1])]
    d2_hist = [df2.iloc[:,i].round(3).value_counts( normalize=True).to_frame("pred") for i in range(df2.shape[1])]
    d3_hist = [df3.iloc[:,i].round(3).value_counts( normalize=True).to_frame("target") for i in range(df3.shape[1])]
    # I dont know for some reason its easier to have it in one DF
    joints_adapt = [ x.join(y, how="outer").fillna(0) for x,y in zip(d1_hist,d2_hist)]
    joints_orig = [ x.join(y, how="outer").fillna(0) for x,y in zip(d1_hist,d3_hist)]
    #[x.to_pickle(os.path.join(work,"pickle",str(j)+"adaptest.pkl")) for j,x in enumerate(joints_adapt)]
    #[x.to_pickle(os.path.join(work,"pickle",str(j)+"origtest.pkl")) for j,x in enumerate(joints_orig)]
    jsds_adapt = [ JSD(x) for x in joints_adapt ]
    jsds_orig = [ JSD(x) for x in joints_orig ]
    out = []
    for j,(jsd_a,jsd_o) in enumerate(zip(jsds_adapt,jsds_orig)):
        if jsd_o*jsd_a>0:#makes sure i dont divide by 0
            jsd = (jsd_o-jsd_a)/jsd_o#this is the relative improvement
            out.append((j,jsd))
            
    fig,ax = plt.subplots(figsize=(10,10))
    ax.plot([x for (x,y) in out],[y for (x,y) in out], "o",label = basename)
    ax.set_xticks([x for (x,y) in out])
    ax.set_xticklabels([df1.columns[x] for (x,y) in out], fontsize=20)
    fig.legend(fontsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join(work,"plots/distance/","JS"+basename.replace("source","").replace("pkl","pdf")))
    return out



if __name__=="__main__":
    scr = os.environ["SCR"]
    work = os.environ["WORK"]
    files2d = glob.glob(os.path.join(work,"npys/all_jprobs*npy"))
    files_all = glob.glob(os.path.join(work,"pickle/source*pkl"))
    files_all= [x for x in files_all if os.path.exists(os.path.join(scr,"checkpoints",
                          os.path.basename(x)[len("source_"):].replace(".pkl",".hdf5")))]
    files2d = [x for x in files2d if os.path.exists(os.path.join(scr,"checkpoints",
        os.path.basename(x)[len("all_jprobs_"):].replace(".npy",".hdf5")))]

    if len(sys.argv)>1:
        files2d= [x for x in files2d if sys.argv[1] in x]
        files_all= [x for x in files_all if sys.argv[1] in x]
    assert len(files2d)>0
    assert len(files_all)>0
    print(len(files_all), len(files2d))
    
    names = ["pred","source","target"]
    colors = {"predsource":"r","sourcepred":"r",
            "predtarget":"b","targetpred":"b",
            "sourcetarget":"g","targetsource":"g"}

    def d2probdist(file):
        """plots relative improvements of some metrics
        for pairs of properties"""
        name = os.path.basename(file)
        a=np.load(file)
        if a.shape[1]==10:
            variables = ["cwp","lwp","iwp","ptop","tsurf"]
            joint_vars =list( combinations(variables,2))
        elif a.shape[1]==28:
            variables = ["cwp","lwp","iwp","cerl","ceri","cod","ptop","tsurf"]
            joint_vars =list( combinations(variables,2))
        elif a.shape[1]==36:
            variables = ["cwp","lwp","iwp","cerl","ceri","cod","ptop","tsurf","clt"]
            joint_vars =list( combinations(variables,2))
        else:
            #this might be a problem if clt is diagnosed or fewer variables are used but hey
            raise NotImplementedError(("That behaviour is not supported: file has shape {},"
                         "need 5 or 8 or 9 variables, i.e. shape 10 or 28 or 36").format(a.shape))
        joint_vars = ["({},{})".format(x,y) for x,y in joint_vars]
        fig,ax = plt.subplots(figsize=(10,10))
        fig2,ax2 = plt.subplots(figsize=(10,10))
        fig4,ax4 = plt.subplots(figsize=(16,8))
        spbase_emd=np.zeros(a.shape[1])
        stbase_emd=np.zeros(a.shape[1])
        spbase_mse=np.zeros(a.shape[1])
        stbase_mse=np.zeros(a.shape[1])
        for i in range(a.shape[1]):
            pred = a[2,i]
            source = a[0,i]
            target = a[1,i]
            
            xx,yy = np.meshgrid(np.linspace(0,1,len(target)),np.linspace(0,1,len(target)))
            xx=xx.flatten()
            yy=yy.flatten()
            coords = np.stack((xx,yy),1)
            #gets euclidian distance between all pairs of points
            spat_dist =  sp.spatial.distance.cdist(coords,coords)
            ls = [pred,source,target]
            for X,Y in combinations(zip(ls,names),2):
                x,xn = X
                y,yn = Y
                #optimizes for earth movers distance
                dist = ot.emd2(x.flatten(),y.flatten(),spat_dist)
                if colors[xn+yn]=="r": #basically a logical OR
                    spbase_emd[i] = dist
                    spbase_mse[i] = np.mean((x-y)**2)
                elif colors[xn+yn] == "g":
                    stbase_emd[i] = dist
                    stbase_mse[i] = np.mean((x-y)**2)
                
                if i==0 and colors[xn+yn]=="r":
                    red=ax.scatter(i,dist,c=colors[yn+xn])
                    r_mse = ax2.scatter(i,np.mean((x-y)**2),c=colors[yn+xn],marker="d")
                elif i==0 and colors[xn+yn]=="b":
                    blue=ax.scatter(i,dist,c=colors[yn+xn])
                    b_mse = ax2.scatter(i,np.mean((x-y)**2),c=colors[yn+xn],marker="d")
                elif i==0 and colors[xn+yn]=="g":
                    green=ax.scatter(i,dist,c=colors[yn+xn])
                    g_mse = ax2.scatter(i,np.mean((x-y)**2),c=colors[yn+xn],marker="d")
                else:
                    ax.scatter(i,dist,c=colors[yn+xn])
                    ax2.scatter(i,np.mean((x-y)**2),c=colors[yn+xn],marker="d")

        relemd = (stbase_emd-spbase_emd)/stbase_emd
        ax4.plot(relemd,"-o",label = "EMD")
        relmse = (stbase_mse-spbase_mse)/stbase_mse
        ax4.plot(relmse,"-d",label = "MSE")
        if np.any((relemd<0)+(relmse<0)):
            ax4.plot([0,len(stbase_mse)],[0,0],"--k",label=None,)
        ax4.set_ylabel("relative improvement",fontsize=22)
        #ax4.plot((stbase_fid-spbase_fid)/stbase_fid,"-t",label = "rel FID improvement")
        ax4.set_xticks(range(0,a.shape[1],2))
        ax4.set_xticklabels(joint_vars[::2])
        ax4.tick_params(axis="x",labelrotation=45,labelsize=20)
        ax4.grid()
        ax_t = ax4.secondary_xaxis("top")
        ax_t.set_xticks(range(1,a.shape[1],2))
        ax_t.set_xticklabels(joint_vars[1::2])
        ax_t.tick_params(direction="out",rotation=45,labelsize=20)
        fig4.tight_layout()
        ax4.legend(loc="best", fontsize=20)
        ax.set_title("Wasserstein  Distances")
        ax2.set_title("MSE Distances")
        ax.set_xticks(range(a.shape[1]))
        ax.set_xticklabels(joint_vars,fontsize=20)
        ax2.set_xticks(range(a.shape[1]))
        ax2.set_xticklabels(joint_vars,fontsize=20)
        ax.tick_params(labelrotation=80)
        ax2.tick_params(labelrotation=80)
        fig.tight_layout()
        fig2.tight_layout()
        ax.legend([red,blue,green],["pred vs source", "pred vs target",
                     "source vs target"],loc="best",fontsize=20)
        ax2.legend(  [r_mse,b_mse,g_mse] ,["MSE PvS","MSE PvT", "MSE SvT"],
                    loc="best",fontsize=20)
        
        fig.savefig(os.path.join(work, "plots/distance/emd"+name.replace("npy","pdf")))
        fig2.savefig(os.path.join(work, "plots/distance/mse"+name.replace("npy","pdf")))        
        fig4.savefig(os.path.join(work, "plots/distance/rel"+name.replace("npy","pdf")))
        print("processed",name)
        return np.mean((stbase_emd-spbase_emd)/stbase_emd),np.mean((stbase_mse-spbase_mse)/stbase_mse)

    def compute(a,b,coords1,coords2,idcs):
        """computes EMD for the given subsample fo the data because its impossible to do it for the full number of samples"""
        A=a[tuple(idcs)]
        B=b[tuple(idcs)]
        A/=A.sum()
        B/=B.sum()
        #print(coords2.shape,flush=True)
        coords1=coords1[tuple(idcs)]
        coords2=coords2[tuple(idcs)]
        sp_dist = sp.spatial.distance.cdist(coords1,coords2)
        #print(sp_dist.shape,flush=True)
        wdist = ot.emd2(A,B,sp_dist,numItermax=int(1e9))
        return wdist

    #full 
    def fulldjprob(sourcefile):
        """computes the distance in the high-dimensional space of all properties
        is very limited because the euclidian distances of all pairs of points need
        to be stored and that is next to impossible in a big high-dimensional space"""
        basename = os.path.basename(sourcefile)
        predfile = os.path.join(work,"pickle",basename.replace("source_","pred_"))
        targfile = os.path.join(work,"pickle",basename.replace("source_","target_"))
        dfs = pd.read_pickle(sourcefile).values[:,:8]
        dfp = pd.read_pickle(predfile).values[:,:8]
        dft = pd.read_pickle(targfile).values[:,:8]
        chunk = int(1e6)
        hs,bs = np.histogramdd(dfs[:chunk])
        hp,bp = np.histogramdd(dfp[:chunk])
        ht,bt = np.histogramdd(dft[:chunk])
        bs =[(x[1:]+x[:-1])/2 for x in bs]
        bp =[(x[1:]+x[:-1])/2 for x in bp]
        bt =[(x[1:]+x[:-1])/2 for x in bt]
        gs = np.stack(np.meshgrid(*bs),-1)
        gp = np.stack(np.meshgrid(*bp),-1)
        gt = np.stack(np.meshgrid(*bt),-1)
        nonzerosp = np.argwhere((hs>0)+(hp>0))
        nonzerost = np.argwhere((hs>0)+(ht>0))
        nonzerotp = np.argwhere((ht>0)+(hp>0))
        results = []
        for _ in range(10):
            np.random.shuffle(nonzerosp)
            np.random.shuffle(nonzerotp)
            np.random.shuffle( nonzerost)
            #subsamples a bunch of random points between which the distances are computed
            #doing that 10 times hopefully alleviates the stochasticity of the results
            idsp = nonzerosp[:int(2e4)].T
            idtp =  nonzerotp[:int(2e4)].T
            idst = nonzerost[:int(2e4)].T
            wsp = compute(hs,hp,gs,gp,idsp)
            wtp = compute(ht,hp,gt,gp,idtp)
            wst = compute(hs,ht,gs,gt,idst)
            #print("sp {:.3e} , tp {:.3e} , st {:.3e}".format(wsp,wtp,wst))
            results.append([wsp,wtp,wst])
        print("processed",sourcefile)
        return results

    if len(files2d)<=2:
        all_jsds = []
        emds_mses =[]
        r_all= []
        for f in files2d:
            emds_mses.append(d2probdist(f))
        for f in files_all:
            all_jsds.append(JSD_df(f))
            r_all.append(fulldjprob(f))
    else:
        pool=mlp.Pool(min(50,len(files2d)))
        all_jsds = pool.map_async(JSD_df,files_all)
        try:
            r_2d=pool.map_async(d2probdist, files2d)
            r_all=pool.map_async(fulldjprob, files_all)
            emds_mses = r_2d.get(timeout=72000)
            r_all = list(r_all.get(timeout=72000))
            all_jsds = all_jsds.get(timeout=72000)
        except mlp.context.TimeoutError:
            pass
        emds_mses = [x for x in emds_mses if x is not None]
        r_all = [x for x in r_all if x is not None]
        all_jsds = [x for x in all_jsds if x is not None]
    #print(len(r_all))
    for j,name in enumerate(files_all):
        fig,ax = plt.subplots()
        r_a=np.array(r_all[j])
        
        c=["r","b","g"]
        l=["sp","tp","st"]
        for i in range(3):
            ax.plot(r_a[:,i],"o"+c[i],label=l[i])
            ax.plot([0,len(r_a)],[np.mean(r_a[:,i]), np.mean(r_a[:,i])],".--"+c[i],
                    label="mean"+l[i])
        fig.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(work,"plots/distance/all_wdists{}".format(
                        os.path.basename(name)[6:].replace("pkl","pdf"))))
        plt.close("all")
    mean_js_impr = [np.mean([y for _,y in x]) for x in all_jsds]
    mean_wstein_impr = [np.mean(x[0]) for x in emds_mses]
    mean_mse_impr = [np.mean(x[1]) for x in emds_mses]
    print("best JS {}".format(files_all[np.argmax(mean_js_impr)]),np.max(mean_js_impr))
    print("best wstein {}".format(files2d[np.argmax(mean_wstein_impr)]),np.max(mean_wstein_impr))
    print("best mse {}".format(files2d[np.argmax(mean_mse_impr)]),np.max(mean_mse_impr))
