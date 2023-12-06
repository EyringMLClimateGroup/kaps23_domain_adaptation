"""just makes the npz files of domain adapted ICON data into dataframes"""

import numpy as np
import pandas as pd
import os
import glob
import sys
import joblib

def choosevars(arr):
    """basically just always returns the variable names for the optimal set of features"""
    if arr.shape[-1]==8:
        return varnames
    elif arr.shape[-1]==5:
        return [varnames[i] for i in [0,1,2,6,7]]
    else:
        print(np.mean(arr,0))
        idcs = input("Please enter varidcs")
        idcs = [int(x) for x in idcs]
        return [varnames[i] for i in idcs]

if __name__=="__main__":
    work = os.environ["WORK"]
    scratch = os.environ["SCR"]
    locations = ["lat","lon","time"]
    varnames = ["cwp","lwp","iwp","cerl","ceri","cod","ptop","tsurf","clt"]
    cnames = ["clear","Ci","As","Ac","St","Sc","Cu","Ns","Dc"]
    #thats the model i used in both papers
    model = joblib.load(os.path.join(work,"models",
                                        "viforest100_10000_sep_0123459.pkl"))
                                                            
    if len(sys.argv)>1:
        fakes = glob.glob(os.path.join(work, "npys/fakeICON*{}*npz".format(sys.argv[1])))    
    else:
        fakes = glob.glob(os.path.join(work, "npys/fakeICON*npz"))
    assert len(fakes)>0

    for fake in fakes:
        #apparently if forgot how i saved it or changed it at some point
        #this loads either, but i really dont know what the first part was for again
        try:
            p=np.load(fake)["All"]
            p=p.reshape(-1,p.shape[3])
            vnames = choosevars(p)
            zero_c = np.zeros((len(p),len(cnames)))
            df= pd.DataFrame(np.hstack((p,zero_c)), columns=vnames+cnames,dtype="float32")
        except KeyError:
            a=np.load(fake)
            p=a["properties"]
            l=a["locations"]
            print(p.shape, l.shape)
            l=l.reshape(-1,l.shape[-1])
            p=p.reshape(-1,p.shape[-1])
            interesting = np.where(np.all(p>=0,axis=1), 1,0).astype(bool)
            p=p[interesting]
            l=l[interesting]
            ppred = p[:,:8]
            #vnames = choosevars(p)
            zero_c = np.zeros((np.sum(interesting),len(cnames)),dtype="float32")
            stepsize = int(1e6)
            #iterativeley predicts cloud types
            for i in range(0,len(p),stepsize):
                outp = model.predict(ppred[i:i+stepsize])
                assert np.allclose(np.sum(outp,1),1,atol=1e-4),i
                assert ~np.any(np.isnan(outp)),i
                zero_c[i:i+stepsize]+=outp
            
            print(zero_c[np.random.randint(0,len(zero_c),size=(15,))])
            print(p.shape,l.shape,zero_c.shape)
            df = pd.DataFrame(np.hstack((p,l,zero_c)),columns=varnames+locations+cnames,dtype="float32")
            time = pd.to_datetime(df.time, format="%Y%m%d",origin="1970-01-01",errors="coerce")
            basetime = pd.Timestamp("1970-01-01")
            df.time = (time-basetime).dt.days
            df = df.dropna(axis="index", how="any")
            print("after dropna", len(df))
        df.to_parquet(os.path.join(work,"frames/parquets", os.path.basename(fake).replace("npz","parquet")),engine="pyarrow")
        print("done {}".format(fake))
        