#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:43:26 2022
mostly makes plots for the DA
but also manages performance metric for tuning
and saves some data that are used for analysis eslewhere
@author: arndt
"""
import traceback
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import os
import numpy as np
from dagans import  util_model
from dagans import gan_model
import pandas as pd
import seaborn as sn
from itertools import product,combinations
from matplotlib.colors import LogNorm
from tqdm import tqdm
import seaborn as sns
import warnings

plt.rcParams['text.usetex'] = True


units ={'twp':"g/m²", 'cwp':"g/m²", 'lwp':"g/m²", 'iwp':"g/m²", 'cerl':"µm", 'ceri':"µm", 'cod':"-", 
                'ptop':"hPa", 'tsurf':"K", "clt": "-"}

fullnames={"clear": "Clear sky fraction","Ci":"Cirrus/Cirrostratus fraction",
                  "As":"Altostratus fraction", "Ac":"Altocumulus fraction",
                  "St":"Stratus fraction", "Sc": "Stratocumulus fraction",
                  "Cu": "Cumulus fraction", "Ns": "Nimbostratus fraction",
                  "Dc": "Deep convection fraction","clear_p": "Predicted clear sky fraction",
                  "Ci_p":"Predicted Cirrus/Cirrostratus fraction",
                  "As_p":"Predicted Altostratus fraction", 
                  "Ac_p":"Predicted Altocumulus fraction",
                  "St_p":"Predicted Stratus fraction", "Sc_p": "Predicted Stratocumulus fraction",
                  "Cu_p": "Predicted Cumulus fraction", "Ns_p": "Predicted Nimbostratus fraction",
                  "Dc_p": "Predicted Deep convection fraction",
                  "cwp":"Cld. Water P.", "twp":"Cld. Water P.",
                  "lwp": "Liquid Water P.", "iwp":"Ice Water P.",
                  "cod": "Cld. Opt. Depth", "tsurf": "Surface Temp.",
                  "tsurf": "Surface Temp.", "cee": "Emissivity",
                  "ptop": "Cld. Top Press.", "htop": "cloud top height",
                  "ttop": "cloud top temperature", "cerl": "Eff. Droplet Radius",
                  "ceri": "Eff. Ice Part. Rad.","ptop":"Cld. Top Press.",
                  "clt": "Cloud Area Fraction"}

def grid_axis(ax,x,y,left=True):
    """
    adjust axis ticks and label in a pairgrid plot
    pass the individual axis and its coords on grid
    all y are >=x
    """
    if not left:
        raise NotImplementedError("havent bothered to implement upper diag")
    a=ax[x,y]
    #a.text(0.5,0.5,"({},{})".format(x,y))
    X=len(ax)-1
    Y=ax.shape[1]-1
    a.set_xlim(-0.01,1.01)
    a.tick_params(labelbottom=(x==X),labelleft=(y==0),labelsize=21)
    a.xaxis.label.set_visible(x==X)
    a.yaxis.label.set_visible(y==0)
    a.xaxis.label.set_size(24)
    a.yaxis.label.set_size(24)
    
    if x!=y:
        a.set_ylim(-0.01,1.01)
    else:
        a.tick_params(labelleft=False,labeltop=True,labelsize=21,top=True)
        a.xaxis.set_label_position('top')
        a.xaxis.label.set_visible(True)
        
        

def getback_clt(target, source, converted):
    """
    gigantic function that like under super specific condtions returns clt
    if clt is not in the generator output and if its in the toamodel output
    """
    modelpath = os.path.join(os.environ["WORK"], "models/toa_model_wclt")
    toamodel = tf.keras.models.load_model(modelpath)
    realinputs = tf.concat((target, source),axis=-1)
    fakeinputs = tf.concat((target, converted),axis=-1)
    realtoa = toamodel(realinputs)
    faketoa = toamodel(fakeinputs)
    
    t_realtoa, s_realtoa = tf.split(realtoa,2,axis=-1)
    t_faketoa, s_faketoa = tf.split(faketoa,2,axis=-1)
    t_realclt = t_realtoa[...,-1,tf.newaxis]
    t_faketoa = t_faketoa[...,:-1]
    s_realclt = s_realtoa[...,-1,tf.newaxis]
    s_fakeclt = s_faketoa[...,-1,tf.newaxis]
    s_faketoa = s_faketoa[...,:-1]
    target = tf.concat((target, t_realclt),axis=-1)
    source = tf.concat((source, s_realclt),axis=-1)
    converted = tf.concat((converted, s_fakeclt),axis=-1)

    return target,source,converted,s_faketoa, t_faketoa


def KLD(dist1, dist2):
    """Kullback leibler divergence"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return tf.reduce_sum([ x for x in dist1*tf.math.log(dist1/dist2) if not tf.math.is_nan(x)])

def JSD(prob):
    """jensen shannon divergence 
    tensor containing dist1,dist2 univariate probability distributions that sum to 1"""
    assert prob.shape[1]==2,prob.shape
    dist1 = prob[:,0]
    dist2 = prob[:,1]

    tf.debugging.assert_near(tf.reduce_sum(dist1),1.)
    tf.debugging.assert_near(tf.reduce_sum(dist2),1.)

    mixture = (dist1+dist2)/2
    kld1 = KLD(dist1,mixture)
    kld2 = KLD(dist2,mixture)
    return (kld1+kld2)/2

def normhist(x):
    """I think this is a manually normalize histogram,
    normalize it because it needs to be probability dist for JSD

    Args:
        x (tensor): some data

    Returns:
        tensor: histogram with 100 bins
    """    
    y= x.round(3).flatten()#not sure why round
    return tf.histogram_fixed_width(y,(0,1))/len(y)

def comp_JSD(source, target,pred):
    """Gets the improvement in JSC for the univariate distributions of all variables

    Args:
        source np.ndarray: stack of ESACCI data
        target np.ndarray: stack of ICON data
        pred np.ndarray: stack of synthetic ESACCI

    Returns:
        tf.Tensor: JSDs for all var pairs
    """    
    l=source.shape[-1]
    s_hist = [normhist(source[...,i]) for i in range(l)]
    t_hist = [normhist(target[...,i]) for i in range(l)]
    p_hist = [normhist(pred[...,i]) for i in range(l)]
    joints_adapt = [tf.stack([x,y],axis=1) for x,y in zip(s_hist,p_hist)]
    joints_orig = [tf.stack([x,y],axis=1) for x,y in zip(s_hist,t_hist)]
    jsds_adapt = [JSD(x) for x  in joints_adapt]
    jsds_orig = [ JSD(x) for x in joints_orig ]
    out = []
    for j,(jsd_a,jsd_o) in enumerate(zip(jsds_adapt,jsds_orig)):
        if jsd_o*jsd_a>0:
            jsd = (jsd_o-jsd_a)/jsd_o
            out.append(jsd)
        else:
            out.append(0)
    return tf.stack(out)

def rescale(values,variables,folder):
    """Undoes the minmax scaling required for the model.
    Also adjusts what happened to ptop to be able to infer "missing values" again

    Args:
        values numpy.ndarray: array to rescale
        variables numpy.ndarray: The variables that were included in this particular model
        folder string: where the minmax scale factors are saved

    Returns:
        np.ndarray: rescaled array
    """    
    mima = np.load(os.path.join(os.environ["WORK"],folder,"minmax_both.npy"))
    mi = mima[0][variables]
    ma = mima[1][variables]
    assert np.all(values<=1.)
    out = np.exp(values*(ma-mi)+mi)-1
    maxs = np.max(out,axis=tuple(range(len(out.shape)-1)))
    if np.any(maxs>1100):
        if len(variables)>=8:
            ptopidx=6
        else:
            ptopidx = np.argmin(np.abs(maxs-1100.666))
        out[...,ptopidx] = np.where(out[...,ptopidx]>=1100,np.nan,out[...,ptopidx]) 
    return out

def r2(a,b):
    """r2 score"""
    ssres = np.sum((a-b)**2)
    sstot = np.sum((a-np.mean(a))**2)
    return 1-(ssres/sstot)

def nonzero_sum(arr,base1,base2):
    base1=base1>0
    base2=base2>0
    base=base1+base2
    return np.nansum(arr[base])

def set_ylim(axis,arrays, bins):
    maxi=0
    for arr in arrays:
        h,_ = np.histogram(arr.flatten(),bins=bins)
        assert np.nanmax(h)>0, (np.min(arr), np.max(arr),bins[0],bins[-1])
        if np.max(h)>maxi:
            maxi = np.max(h)
    for a in axis:
        a.set_ylim((1,maxi+1))

def appl(dataset,name,variables,luigiobject,timelimit,tune=False, tune_name ="dagans"):
    """applies the trained model to ICON data and makes a bunch of plots and computes some metrics"""
    starttime = time.time()
    folder=luigiobject.folder
    try:
        generator = load_model("{}/{}".format(
            folder, name), custom_objects = {"InstanceNormalization": util_model.InstanceNormalization,
                                    "ClipConstraint": gan_model.ClipConstraint})
    except OSError:
        name=name.replace("best","")
        generator = load_model("{}/{}".format(
            folder,name ), custom_objects = {"InstanceNormalization": util_model.InstanceNormalization,
                                    "ClipConstraint": gan_model.ClipConstraint})
    

    #defined this way because of backwards compatibiltiy
    def predicter(x,y):
        gpred = generator.predict(x,verbose=0)
        gpred = np.where(gpred>=0, gpred, 0)   
        gpred = np.where(gpred<=1, gpred, 1)
        return gpred

    names = np.array(["cwp", "lwp", "iwp","cerl","ceri","cod","ptop", "tsurf", "clt"])
    locs = ["lat","lon","time"]
    variables = np.array(variables)
    colnames = list(names[variables])
    work = os.environ["WORK"]
    sources,targets,outs,diff, diff_orig = [], [],[],[],[]
    cltmodel = np.all(np.arange(8)==variables)
    if cltmodel:
        colnames += ["clt"]
        variables = np.arange(9)
        
    target_locs, source_locs = [] , []
    assert len(dataset)>0, type(dataset)
    if tune:
        ds_gen = enumerate(dataset)
    else:
        ds_gen = tqdm(enumerate(dataset))
    for i,batch in ds_gen:
        
        target,source,t_toa,s_toa,t_locs,s_locs, = batch 
        converted = predicter(target,source)#create synthetic obs
    
        assert np.all(~np.isnan(converted)),np.sum(np.isnan(converted))/np.prod(converted.shape)
        assert np.all(~np.isnan(source)),np.sum(np.isnan(source))/np.prod(source.shape)
        assert np.all(~np.isnan(target)),np.sum(np.isnan(target))/np.prod(target.shape)
        if cltmodel:#if i want to include clt in the analysis
            target,source,converted,s_toa_fake, t_toa_fake = getback_clt(target,source,converted)
            t_toa_fake = t_toa_fake.numpy()
            s_toa_fake = s_toa_fake.numpy()
                
            assert np.all(~np.isnan(converted)),np.sum(np.isnan(converted))/np.prod(converted.shape)
            assert np.all(~np.isnan(source)),np.sum(np.isnan(source))/np.prod(source.shape)
            assert np.all(~np.isnan(target)),np.sum(np.isnan(target))/np.prod(target.shape)
        else:
            pass
            #raise RuntimeError("temporary error here to make sure i do whats above")

        target = target.numpy()
        source = source.numpy()
        t_locs = t_locs.numpy()
        s_locs = s_locs.numpy()
        t_toa = t_toa.numpy()
        s_toa = s_toa.numpy()
        
        target_locs.append(t_locs)
        source_locs.append(s_locs)
        h = [np.histogram((converted[...,var]-target[...,var]),
                bins = np.linspace(-1,1,20))[0] for var in range(target.shape[-1])]
        diff.append(h)
        h = [np.histogram((source[...,var]-target[...,var]),
                bins = np.linspace(-1,1,20))[0] for var in range(target.shape[-1])]
        diff_orig.append(h)
        if i<3 and not tune:#plot a few sample images
            #reverts the minmax+logscaling
            source_r = rescale(source,variables,luigiobject.dataset_folder)
            target_r = rescale(target,variables,luigiobject.dataset_folder)
            out_r = rescale(converted,variables,luigiobject.dataset_folder)
            #nans are allowed here because ptop
            assert len(out_r.shape)==4,out_r.shape
            fig,ax = plt.subplots(3,target.shape[-1],figsize=(10,5), gridspec_kw={"hspace": 0.01})  
            fig.subplots_adjust(left=0.05, bottom=0.1, right=0.99,top=.95,
                                wspace=0.15,hspace=-0.05)
            pos_xs = [ax[2,i].get_position().x0 for i in range(target.shape[-1])]
            pos_ws = [ax[2,i].get_position().width for i in range(target.shape[-1])]
            pos_y = ax[2,0].get_position().y0
            for var in range(target.shape[-1]):
                mi_beg = np.nanmin([np.nanmin(source_r[0,...,var]), np.nanmin(target_r[0,...,var]), np.nanmin(out_r[0,...,var])])
                ma_beg = np.nanmax([np.nanmax(source_r[0,...,var]), np.nanmax(target_r[0,...,var]), np.nanmax(out_r[0,...,var])])
                assert ~np.isnan(mi_beg),var
                assert ~np.isnan(ma_beg),var
                i0=ax[0,var].imshow(target_r[0,...,var],vmin=mi_beg, vmax=ma_beg)
                ax[0,var].set_title(names[var])
                i1=ax[1,var].imshow(out_r[0,...,var],vmin=mi_beg, vmax=ma_beg)
                o0=ax[2,var].imshow(source_r[0,...,var],vmin=mi_beg, vmax=ma_beg)
                
                ax[0,var].tick_params(bottom=False, left=False, labelbottom=False,      labelleft=False)
                ax[1,var].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
                ax[2,var].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
                
                pos_x = pos_xs[var]
                width = pos_ws[var]
                cbax = fig.add_axes([pos_x+width*0.2,pos_y-0.02,width*0.6,0.03])
                cb=fig.colorbar(o0,cax=cbax,orientation="horizontal", aspect=(ma_beg-mi_beg)*12)
                cbax.set_xticks([round(mi_beg,2),round((mi_beg+ma_beg)/2,2),round(ma_beg,2)])
                cbax.tick_params(labelsize=8)
                
            ax[0,0].set_ylabel(u"$\mathcal{T}$: Climate model\n (ICON-A)")
            ax[1,0].set_ylabel(u"$\mathcal{O}$: Synthetic\n observations")
            ax[2,0].set_ylabel(u"$\mathcal{S}$: Observations\n (ESA-CCI)")
            fig.savefig(os.path.join(work,"plots/images/test{}_{}.pdf".format(i,name[:-5])))
            
            fig,ax = plt.subplots(3,target.shape[-1],figsize=(10,5))#, gridspec_kw={"hspace": 0.01})
            fig.subplots_adjust(left=0.05, bottom=0.1, right=0.99,top=.95,
                                wspace=0.15,hspace=-0.05)
            pos_xs = [ax[2,i].get_position().x0 for i in range(target.shape[-1])]
            pos_ws = [ax[2,i].get_position().width for i in range(target.shape[-1])]
            pos_y = ax[2,0].get_position().y0
            for var in range(target.shape[-1]):
                mi_beg = np.min([np.nanmin(source[0,...,var]),np.nanmin(target[0,...,var]), np.nanmin(converted[0,...,var])])
                ma_beg = np.max([np.nanmax(source[0,...,var]),np.nanmax(target[0,...,var]), np.nanmax(converted[0,...,var])])
                
                i0=ax[0,var].imshow(target[0,...,var],vmin=mi_beg, vmax=ma_beg)
                ax[0,var].set_title(names[var])
                i1=ax[1,var].imshow(converted[0,...,var],vmin=mi_beg, vmax=ma_beg)
                o0=ax[2,var].imshow(source[0,...,var],vmin=mi_beg, vmax=ma_beg)
                
                ax[0,var].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
                ax[1,var].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
                ax[2,var].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
                
                pos_x = pos_xs[var]
                width = pos_ws[var]
                cbax = fig.add_axes([pos_x+width*0.2,pos_y-0.02 ,width*0.6,0.03])
                cb=fig.colorbar(o0,cax=cbax,orientation="horizontal", aspect=(ma_beg-mi_beg)*12)
                cbax.set_xticks([round(mi_beg,2),round((mi_beg+ma_beg)/2,2),round(ma_beg,2)])
                cbax.tick_params(labelsize=8)
            ax[0,0].set_ylabel(u"$\mathcal{T}$: Climate model\n (ICON-A)")
            ax[1,0].set_ylabel(u"$\mathcal{O}$: Synthetic\n observations")
            ax[2,0].set_ylabel(u"$\mathcal{S}$: Observations\n (ESA-CCI)")
            fig.savefig(os.path.join(work,"plots/images/testlog{}_{}.pdf".format(i,name[:-5])))
            
            plt.close(fig)
            sources.append(source)
            targets.append(target)
            outs.append(converted)
            

        else:
            #dont need to do stuff twice
            if os.path.exists(os.path.join(os.environ["WORK"],"pickle/target_{}.pkl".format(name[:-5]))):
                sources_df=pd.read_pickle(os.path.join(os.environ["WORK"],"pickle/source_{}.pkl".format(name[:-5])))
                outs_df=pd.read_pickle(os.path.join(os.environ["WORK"],"pickle/pred_{}.pkl".format(name[:-5])))
                targets_df=pd.read_pickle(os.path.join(os.environ["WORK"],"pickle/target_{}.pkl".format(name[:-5])))
                sources =sources_df.values[:,:-3]
                outs =outs_df.values[:,:-3]
                targets =targets_df.values[:,:-3]
                target_locs = targets_df.values[:,-3:]
                source_locs = sources_df.values[:,-3:]
                assert not isinstance(sources,list)
                break
            else:
                sources.append(source)
                targets.append(target)
                outs.append(converted)
                proctime = time.time()-starttime
                if proctime>timelimit:
                    print("breaking for time (appl)")
                    break

    if isinstance(sources,list):
        target_locs = np.stack(target_locs).squeeze()
        source_locs = np.stack(source_locs).squeeze()
        sources = np.stack(sources, axis=0).squeeze()
        targets = np.stack(targets, axis=0).squeeze()
        outs = np.stack(outs, axis=0).squeeze()
        diff = np.sum(np.stack(diff,axis=0),axis=0)
        diff_orig = np.sum(np.stack(diff_orig,axis=0),axis=0)
        diff=np.vstack((diff,diff_orig))

        diff_df = pd.DataFrame(diff.T,index = np.linspace(-1,1,20)[1:],
                               columns=colnames+[x+"_org" for x in colnames])    
        #this can be used for downstream analysis     
        diff_df.to_pickle(os.path.join(os.environ["WORK"],"pickle","diff_{}.pkl".format(name[:-5])))
    else:
        print("loaded i guess")    

    cols = colnames+list(locs)
    sources_df =pd.DataFrame(np.hstack((sources.reshape(-1,len(variables)),
                                            source_locs.reshape(-1,3))), columns=cols)
    targets_df = pd.DataFrame(np.hstack((targets.reshape(-1,len(variables)),
                                            target_locs.reshape(-1,3))), columns=cols)
    outs_df = pd.DataFrame(np.hstack((outs.reshape(-1,len(variables)),
                                        source_locs.reshape(-1,3))), columns=cols)
    
    sources_df.to_pickle(os.path.join(os.environ["WORK"],"pickle","source_{}.pkl".format(name[:-5])))
    outs_df.to_pickle(os.path.join(os.environ["WORK"],"pickle","pred_{}.pkl".format(name[:-5])))
    targets_df.to_pickle(os.path.join(os.environ["WORK"],"pickle","target_{}.pkl".format(name[:-5])))
    if np.any( outs)>1:
        outs_r=rescale(np.where(outs>1,1.,outs),variables,luigiobject.dataset_folder)
        print(outs_r.shape)
        print(np.max(outs_r,0),flush=True)
    else:
        outs_r=rescale(outs,variables,luigiobject.dataset_folder)
    #more sure that the rescaling worked 
    assert np.nanmax(outs_r)<10000,np.nanmax(outs_r)
    
    #compute this to use as a tuning metric
    out = comp_JSD(sources,targets,outs).numpy()
    print(np.any(np.isnan(outs_r[:,:3])))
    #some of these values are wild
    print("cwpmax", np.nanmax(outs_r[:,0]),"lwpmax",np.nanmax(outs_r[:,1]), "iwpmax",np.nanmax(outs_r[:,2]))
    cwpconst = np.sqrt(np.nanmean((outs_r[:,0]-outs_r[:,1]-outs_r[:,2])**2))
    out = np.hstack([out, cwpconst])
    np.save(os.path.join(os.environ["SCR"],"ray_results",tune_name,
                name.replace("hdf5","npy")),out)
    
    if np.mean(out)<-3:
        #if results are shit dont bother clogging disk
        startsuffix = name.find("tuning")
        assert startsuffix>0
        suffix = name[startsuffix:startsuffix+14]
        os.system("rm -r {}".format(os.path.join(os.environ["SCR"],
                    "logs/*{}*".format(suffix))))
    
    if tune:
        return#dont need all the plots
    else:
        print(out)
    sources_r = rescale(sources,variables,luigiobject.dataset_folder)
    targets_r=rescale(targets,variables,luigiobject.dataset_folder)
    #this is the fake observations from icon
    np.savez_compressed(os.path.join(os.environ["WORK"],"npys/fakeICON_{}.npz".format(name[:-5])),
                            properties=outs_r,locations = target_locs)
        
    fig,ax = plt.subplots(sources.shape[-1],3,figsize=(sources.shape[-1]*3,sources.shape[-1]*2),sharey=True)
    fig_lin,axlin = plt.subplots(sources.shape[-1],3,figsize=(sources.shape[-1]*3,sources.shape[-1]*2),sharey=True)

    for v in range(sources.shape[-1]):
        source = sources_r[...,v]
        target = targets_r[...,v]
        out_v = outs_r[...,v]
        source = source[source>1e-8]
        target = target[target>1e-8]
        out_v = out_v[out_v>1e-8]
        
        s_hist = np.zeros(100)
        t_hist = np.zeros(100)
        #computing histogram unto that point where it cointains 99% of data
        #left to right for most variables, mid symmetric for ptop and tsurf
        _h,_b = np.histogram(out_v,bins=1000)  
        _cs = np.cumsum(_h)/np.sum(_h)
        if names[v]!="tsurf" and names[v]!="ptop":
            o_min = np.nanmin(out_v)      
            o_max = _b[1:][_cs>0.99][0]
        else:
            o_min = _b[1:][_cs>0.005][0]
            o_max = _b[1:][_cs>0.995][0]
        
        _h,_b = np.histogram(source,bins=1000)
        _cs = np.cumsum(_h)/np.sum(_h)
        if names[v]!="tsurf" and names[v]!="ptop":
            s_min = np.nanmin(source)
            s_max = _b[1:][_cs>0.99][0]
        else:
            s_min = _b[1:][_cs>0.005][0]
            s_max = _b[1:][_cs>0.995][0]

        _h,_b = np.histogram(target,bins=1000)
        _cs = np.cumsum(_h)/np.sum(_h)
        if names[v]!="tsurf" and names[v]!="ptop":
            t_min= np.nanmin(target)
            t_max = _b[1:][_cs>0.99][0]
        else:
            t_min= _b[1:][_cs>0.005][0]
            t_max = _b[1:][_cs>0.995][0]
        
        bins = np.linspace(max(0,min(s_min,t_min,o_min)),max(s_max,t_max,o_max),100)

        hs ,_ = np.histogram(source.flatten(), bins=bins)
        ht ,_ = np.histogram(target.flatten(), bins=bins)
        ho ,_ = np.histogram(out_v.flatten(), bins=bins)
        print("best",np.array(["so","to","st"])[np.argmin([np.mean(np.abs(hs-ho)),np.mean(np.abs(ht-ho)),np.mean(np.abs(hs-ht))])])
        axlin[v,0].hist(source.flatten(), bins=bins,log=False)
        axlin[v,1].hist(target.flatten(), bins=bins,log=False)
        axlin[v,2].hist(out_v.flatten(), bins=bins, log=False)
        ax[v,0].hist(source.flatten(), bins=bins,log=True)
        ax[v,1].hist(target.flatten(), bins=bins,log=True)
        ax[v,2].hist(out_v.flatten(), bins=bins, log=True)

        set_ylim(ax[v],[source.flatten(),target.flatten()],bins=bins)
        set_ylim(axlin[v],[source.flatten(),target.flatten()],bins=bins)
        #print some useless histogramm differences
        print("so", np.mean(np.abs(hs-ho)), np.median(np.abs(hs-ho)))
        print("to", np.mean(np.abs(ht-ho)), np.median(np.abs(ht-ho)))
        print("st", np.mean(np.abs(hs-ht)), np.median(np.abs(hs-ht)))
        
        ax[v,0].set_ylabel(fullnames[colnames[v]],fontsize=18)
        axlin[v,0].set_ylabel(fullnames[colnames[v]],fontsize=18)
        for part in range(3):
            if v==0:
                ax[0,part].set_title(["ESA-CCI","ICON-A","synth. Obs."][part],fontsize=18)
                axlin[0,part].set_title(["ESA-CCI","ICON-A","synth. Obs."][part],fontsize=18)

            ax[v,part].tick_params(labelsize=18)
            axlin[v,part].tick_params(labelsize=18)
            ax[v,part].set_xlabel(units[colnames[v]],fontsize=18)
            axlin[v,part].set_xlabel(units[colnames[v]],fontsize=18)
        if 0 in variables and 1 in variables and 2 in variables:   
            # i only need to adjust to cwp if lwp and iwp are also there
            if v==0:#set the xlimit extent according to the cwp limits
                extent = (np.min(bins), np.max(bins))
                which =0
                if 0 in variables:
                    which+=1
                if 1 in variables:
                    which+=1
                if 2 in variables:
                    which += 1

            if v<which:
                [ax[v,part].set_xlim(extent) for part in range(3)]
                [axlin[v,part].set_xlim(extent) for part in range(3)]
            else:
                [ax[v,part].set_xlim((np.min(bins), np.max(bins))) for part in range(3)]
                [axlin[v,part].set_xlim((np.min(bins), np.max(bins))) for part in range(3)]
        
    fig.tight_layout()
    fig.savefig(os.path.join(work,"plots/hist/hist_{}.pdf".format(name[:-5])))
    fig_lin.tight_layout()
    fig_lin.savefig(os.path.join(work,"plots/hist/histlin_{}.pdf".format(name[:-5])))

    
    cmap = sn.color_palette("cubehelix_r", as_cmap=True)
    order = ["s", "t", "o"] 
    bins = 15
    b=(np.linspace(0,1,bins),np.linspace(0,1,bins))
    all_jprobs = np.empty((3,np.arange(len(variables)).sum(),bins-1,bins-1))
    
    for k,df in enumerate([sources_df,targets_df,outs_df]):
        
        """commented out per-domain kde plots for time
        pp=sns.pairplot(df.iloc[np.random.randint(0,len(df),size=int(1e4)),:len(variables)],  hue=None, 
            hue_order=None, palette=None, vars=None,
            x_vars=None, y_vars=None, kind='kde', diag_kind='auto', markers=None,
            height=2.5, aspect=1, corner=False, dropna=False, plot_kws=None,
            diag_kws=None, grid_kws=None, size=None)
        for a in pp.axes.flatten():
            if a is not None:
                a.set_xlim(0,1)
                a.set_ylim(0,1)
        pp.figure.savefig(os.path.join(work, "plots/joint/kde_{}_{}.pdf".format(name[:-5],order[k])))#

        """
        fig,ax = plt.subplots(len(variables),len(variables),figsize=(10,10))
        for i,j in product(range(len(variables)),range(len(variables))):
            a=np.hstack((df.iloc[:,j].values.reshape(-1,1),df.iloc[:,i].values.reshape(-1,1)))
            hist_x,_ = np.histogramdd(a,bins=b,density =True)
         
            try:
                bmax = max(np.max(hist_x),bmax)
            except Exception:
                bmax=np.max(hist_x)
                continue
        agg =0
        for i,j in product(range(len(variables)),range(1,len(variables)+1)):
            
            Norm = LogNorm(vmin= 1e-8,vmax= bmax)
            if i==0 and j==1:
                continue
            
            if i==0:
                df.iloc[:,j-1].hist(ax=ax[i,j-1],log=True)
                ax[i,j-1].set_xlim(0,1)
                ax[i,j-1].set_title(fullnames[df.columns[j-1]]+"[{}]".format(units[df.columns[j-1]]))
            elif j==1:
                df.iloc[:,i-1].hist(ax=ax[i,j-1],log=True, orientation="horizontal")
                ax[i,j-1].set_ylim(0,1)
                ax[i,j-1].set_title(fullnames[df.columns[i-1]]+"[{}]".format(units[df.columns[i-1]]))
            elif j>i:
                """
                hexplot=ax[i,j].hexbin(df.iloc[:,j-1], df.iloc[:,i-1], 
                        cmap=cmap, norm=Norm, gridsize=15)
                """
                a=np.hstack((df.iloc[:,j-1].values.reshape(-1,1),df.iloc[:,i-1].values.reshape(-1,1)))
                #prcompute the distributions needed for the joint distribution metrics
                jprob,edg = np.histogramdd( a,  bins= b,density=True)
                all_jprobs[k, agg] = jprob
                agg+=1
                xx,yy = np.meshgrid(edg[0],edg[1])
                hexplot = ax[i,j-1].pcolormesh(xx,yy,jprob.T,cmap="Greys",norm=Norm)
                ax[i,j-1].set_ylim(0,1)
                ax[i,j-1].set_xlim(0,1)
            else:
                ax[i,j-1].remove()
             
        cbar=plt.colorbar(hexplot, cax=ax[0,0])

        cbar.set_ticks(np.logspace(int(np.log10(1e-4)), int(np.log10(bmax)), 3))
        cbar.set_ticklabels(["{:.1e}".format(x) for x in np.logspace(int(np.log10(1e-4)),int(np.log10(bmax)),3)])
    
        cbar.ax.tick_params(right=False, left=True, labelleft=True, labelright=False )
        fig.tight_layout()
        fig.savefig(os.path.join(work, "plots/joint/joint_{}_{}.pdf".format(name[:-5],order[k])))
        

    kdedf_s = sources_df.iloc[np.random.randint(0,len(df),size=int(1e4)),:]
    kdedf_t = targets_df.iloc[np.random.randint(0,len(df),size=int(1e4)),:]
    kdedf_o = outs_df.iloc[np.random.randint(0,len(df),size=int(1e4)),:]
    
    fig, ax = plt.subplots(3, 3,figsize=(16,16))
    palette = sns.color_palette("colorblind")[:]
    kde_cols=kdedf_s.columns
    #tried to do this with sns.pairgrid, but holy hell thats clunky
    #this has to be adapted depending how many jointplots are desired
    for i_ax,i in enumerate(range(2,5)):#range(len(variables)):
        for j_ax,j in enumerate(range(2,5)):
            
            if i<j:
                ax[i_ax,j_ax].remove()
                continue
            elif i==j:
                sns.kdeplot(data = kdedf_s.iloc[:,i],ax=ax[i_ax,j_ax],
                            color = palette[0],label=r"$\mathcal{S}:$ ESA-CCI" if i_ax==0 else None)
                sns.kdeplot(data = kdedf_t.iloc[:,i],ax=ax[i_ax,j_ax],
                            color = palette[1],label=r"$\mathcal{T}:$ ICON-A" if i_ax==0 else None)
                sns.kdeplot(data = kdedf_o.iloc[:,i],ax=ax[i_ax,j_ax],
                            color = palette[2],label=r"$\mathcal{O}:$ Synthetic ESA-CCI" if i_ax==0 else None)
                
            else:
                sns.kdeplot(data = kdedf_s.iloc[:,[i,j]],y=kde_cols[i],x=kde_cols[j],
                            ax=ax[i_ax,j_ax],alpha=1,color = palette[0], fill =True,
                            )
                sns.kdeplot(data = kdedf_t.iloc[:,[i,j]],y=kde_cols[i],x=kde_cols[j],
                            ax=ax[i_ax,j_ax],alpha=.7,color = palette[1], fill =True
                            )
                sns.kdeplot(data = kdedf_o.iloc[:,[i,j]],y=kde_cols[i],x=kde_cols[j],
                            ax=ax[i_ax,j_ax],alpha=.4,color = palette[2], fill =True
                            )
            grid_axis(ax,i_ax,j_ax,left=True)
        
    fig.legend(loc="center",bbox_to_anchor=[.5,.5,.5,.5],ncol=1,fontsize=24)
    fig.subplots_adjust(bottom =0.05, left=0.05, top=0.95, 
                        right=0.95,wspace=.08)
    fig.savefig(os.path.join(work, "plots/joint/kde_{}_all.pdf".format(name[:-5])))#
    
    print("done",np.nanmax(all_jprobs))
    np.save(os.path.join(os.environ["WORK"],
                         "npys/all_jprobs_{}".format(name[:-5])),all_jprobs)
