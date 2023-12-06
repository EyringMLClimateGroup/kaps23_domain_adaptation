"""computes clt changes induced by COSP on climate models
the change is saved and compared to WGAN-DA in quantidiff.py"""
import numpy as np
import os
from glob import glob
import sys
import netCDF4 as nc4
import h5py
from matplotlib import pyplot as plt
import multiprocessing as mlp
#

def get_clt(file):
    data = nc4.Dataset(file,"r", format="NETCDF4")
    key = [x for x in data.variables.keys() if "clt" in x][0]
    clt = data.variables[key][...]
    clt = np.where(clt>1e8,0,clt)
    return np.where(clt>=0, clt,0)

def minmax_scale(cosp,native,mi=0,ma=1):
    mi = np.log(1+mi)
    ma = np.log(1+ma)
    cosp = (np.log(1+cosp)-mi)/(ma-mi)
    native = (np.log(1+native)-mi)/(ma-mi)
    return cosp, native

def mk_hist(cosp,native):
    h,_ = np.histogram((cosp-native),bins=np.linspace(-1,1,20))
    return h

def proc(f_cosp, f_native, mi=0,ma=100):
    clt_cosp=get_clt(f_cosp)
    clt_native = get_clt(f_native)
    if not np.all(clt_cosp.shape==clt_native.shape):
        return []
    assert np.min(clt_cosp)>=0,np.min(clt_cosp)
    assert np.max(clt_cosp)<=100,np.max(clt_cosp)
    clt_cosp,clt_native = minmax_scale(clt_cosp,clt_native, mi,ma)
    h = mk_hist(clt_cosp, clt_native)
    return h


if __name__=="__main__":

    setups ={"model":["CNRM-CM6-1", "CNRM-ESM2-1", "GFDL-CM4", 
                        "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "IPSL-CM6A-LR",
                        "MRi_ESM2-0"], 
             "institute":["CNRM-CERFACS", "CNRM-CERFACS", "NOAA-GFDL", 
                 "MOHC", "MOHC", "IPSL", "MRI"],
             "ensemble": ["r1i1p1f2", "r1i1p1f2", "r1i1p1f1", "r1i1p1f3",
                        "r2i1p1f3", "r1i1p1f1", "r1i1p1f1"],
             "clt": ["cltcalipso", "cltisccp"]
             
             }
    cosp_folder = "/work/ik1017/CMIP6/data/CMIP6/CMIP/<institute>/<model>/amip/<ensemble>/CFday/<clt>/"
    native_folder = "/work/ik1017/CMIP6/data/CMIP6/CMIP/<institute>/<model>/amip/<ensemble>/day/clt/"
    pool = mlp.Pool(20)
    fig,ax = plt.subplots(1,7,figsize=(15,15)) 
    ax=ax.flatten()
    results = {}
    for i in range(7):
        all_h=[]
        for j in range(2):
            model = setups["model"][i]
            inst = setups["institute"][i]
            ens = setups["ensemble"][i]
            clt_from = setups["clt"][j]
            
            grids = glob(cosp_folder.replace("<institute>",inst).replace("<model>",
                                model).replace("<ensemble>",ens).replace("<clt>",
                                    clt_from)+"/*")
            for grid in grids:
                grid = os.path.basename(grid)
                versions =  glob(os.path.join(cosp_folder.replace("<institute>",inst).replace("<model>",
                                model).replace("<ensemble>",ens).replace("<clt>",
                                clt_from),grid,"*"))
                for version in versions:
                    version = os.path.basename(version)
                    cosp_files = glob(os.path.join(cosp_folder.replace("<institute>",inst).replace("<model>",
                                model).replace("<ensemble>",ens).replace("<clt>",
                                clt_from),grid,version,"*nc"))
                    native_files = glob(os.path.join(native_folder.replace("<institute>",inst).replace("<model>",
                                model).replace("<ensemble>",ens).replace("<clt>",
                                clt_from),grid,version,"*nc")) 
                    print("num files","{}_{}_{}_{}_{}_{}".format(model,
                                                                    inst,ens,
                                                                    clt_from,
                                                                    grid,
                                                                    version),len(cosp_files), len(native_files))
                    #histograms = pool.starmap(proc,list(zip(cosp_files, native_files)))
                    #all_h+=histograms
    """
        all_h = [x for x in all_h if len(x)>0] 
        if len(all_h)==0:
            continue
        print(all_h)

        hist=np.sum(np.stack(all_h,axis=0),axis=0)
        results["{}_{}_{}".format(model,inst,ens)] = hist.copy()
        ax[i].bar(np.linspace(-1,1,19)+0.04,height=hist, width=0.08)
        ax[i].set_title("{}_{}_{}".format(model,inst,ens))
    np.savez_compressed(os.path.join(os.environ["WORK"],"cosp_vs_native.npz"),**results)
    fig.tight_layout()
    fig.savefig(os.path.join(os.environ["WORK"],"stats","cosp_vs_native.pdf".format(model,
                                                                                  inst,ens)))
    """                                                                                  
