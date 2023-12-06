import sys
import numpy as np
import os
import glob
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.dataloader import default_collate
import h5py
import warnings
import datetime
from tqdm import tqdm
import time
import torch


def nonehandling_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch)>0:
        tup=(
       (np.vstack([x['values'][0] for x in batch]),np.vstack([x['values'][1] for x in batch])),
      (np.vstack([x['locs'][0] for x in batch]),np.vstack([x['locs'][1] for x in batch]))
        )

        return tup#default_collate(batch)
    else:
        return None,None


def window_nd(a, window, steps = None, axis = None, gen_data = False):
    """
    Create a windowed view over `n`-dimensional input that uses an 
    `m`-dimensional window, with `m <= n`

    Parameters
    -------------
        a : Array-like
            The array to create the view on

        window : tuple or int
                If int, the size of the window in `axis`, or in all dimensions if 
                `axis == None`
    
                If tuple, the shape of the desired window.  `window.size` must be:
                equal to `len(axis)` if `axis != None`, else 
                equal to `len(a.shape)`, or 
                1
                            
        steps : tuple, int or None
                The offset between consecutive windows in desired dimension
                If None, offset is one in all dimensions
                If int, the offset for all windows over `axis`
                If tuple, the steps along each `axis`.  
                `len(steps)` must me equal to `len(axis)`
                    
        axis : tuple, int or None
                The axes over which to apply the window
                If None, apply over all dimensions
                if tuple or int, the dimensions over which to apply the window

        gen_data : boolean
                    returns data needed for a generator
                        
    Returns
    -------
                
        a_view : ndarray
                A windowed view on the input array `a`, or `a, wshp`, where `whsp` is the window shape needed for creating the generator

    """
    ashp = np.array(a.shape)
            
    if axis != None:
        axs = np.array(axis, ndmin = 1)
        assert np.all(np.in1d(axs, np.arange(ashp.size))), "Axes out of range"
    else:
        axs = np.arange(ashp.size)

    window = np.array(window, ndmin = 1)
    assert (window.size == axs.size) | (window.size == 1), "Window dims and axes don't match"
    wshp = ashp.copy()
    wshp[axs] = window
    assert np.all(wshp <= ashp), "Window is bigger than input array in axes"
            
    stp = np.ones_like(ashp)
    if steps:
        steps = np.array(steps, ndmin = 1)
        assert np.all(steps > 0), "Only positive steps allowed"
        assert (steps.size == axs.size) | (steps.size == 1), "Steps and axes don't match"
        stp[axs] = steps
            
    astr = np.array(a.strides)

    shape = tuple((ashp - wshp) // stp + 1) + tuple(wshp)
    strides = tuple(astr * stp) + tuple(astr)
    as_strided = np.lib.stride_tricks.as_strided
    a_view = np.squeeze(as_strided(a, 
                                 shape = shape, 
                                  strides = strides))
    if gen_data :
        return a_view, shape[:-wshp.size]
    else:
        return a_view

class CombinedDataset(Dataset):
    """Combines ESACCI and ICON data,  and makes patches
    """    
    def __init__(self, source_dir, target_dir, variable_names=None,
                 chunksize = 1,
                 tilesize=1,
                 noncloudy_scale=False,
                 daytime = False,
                 roll=True):
        """
        Args:
            source_dir (string): folder of files for source data (ESACCI)
            target_dir (string): folder of files for Target data (ICON-A)
            variable_names (list, optional): list of strings . Defaults to None.
            chunksize (int, optional): length of side of square over which to average source data. Defaults to 1.
            tilesize (int, optional): sidelenght of square patches of chunks. Defaults to 1.
            noncloudy_scale (bool, optional): deprecated way of dealing with missing clouds. Defaults to False.
            daytime (bool, optional): if only patches with a certain number of daytime chunks should be returned. Defaults to False.
            roll (bool, optional): if the data should be rolled to avoid oversampling of patch intersections. Defaults to True.
        """        
        self.tilesize=tilesize
        self.chunksize=chunksize
        self.source_files = glob.glob(os.path.join(source_dir,"*.npz"))
        self.target_files = glob.glob(os.path.join(target_dir, "r360*.npz"))
        self.daytime = daytime
        self.roll=roll
        assert len(self.source_files)>0, os.listdir(source_dir)
        assert len(self.target_files)>0, os.listdir(target_dir)


        source_weeks = [self.get_week(x) for x in self.source_files]
        target_weeks =[self.get_week(x) for x in self.target_files]
        #constructs the pseudosimultaneous pairs, by excluding samples from the set with more samples
        if len(self.source_files)>len(self.target_files):
            new_source = [x for (x,y) in zip(self.source_files, source_weeks) if y in np.unique(target_weeks)]
            new_target = list(map(lambda x: self.get_from_same_week(x,self.target_files, target_weeks),source_weeks))
            new_target = [x for x in new_target if x is not None]
            self.source_files=new_source
            self.target_files = new_target
        else:
            new_target = [x for (x,y) in zip(self.target_files, target_weeks) if y in np.unique(source_weeks)]
            new_source = list(map(lambda x: self.get_from_same_week(x,self.source_files, source_weeks),target_weeks))
            new_source = [x for x in new_source if x is not None]
            self.source_files=new_source
            self.target_files = new_target
        #turns the variable names into the corresponding indices.
        source_var_dict = {"cwp":0,"lwp":1,"iwp":2,"cerl":3,"ceri":4,"cod":5,
                            "ptop":6,"htop":7,"ttop":8,"cee":9, "tsurf":10,
                            "rlut":11,"rsut":12,"rsdt":13,"rsutcs":14, "rlutcs":15,"clt":16}
        target_var_dict = {"cwp":0,"lwp":1,"iwp":2,"cerl":3,"ceri":4,"ptop":5,
                            "tsurf":6, "cod":7, "rlut":9,"rsut":10,"rsdt":11,
                            "rsutcs":12, "rlutcs":13,"clt":14}
        self.source_vars = np.array([source_var_dict[x] for x in variable_names])
        self.target_vars = np.array([target_var_dict[x] for x in variable_names])
        
        #get the incloud variables, only defined for cloudy pixels
        tostack=[np.argwhere(self.source_vars==x)
                                for x in [6,7,8]if x in self.source_vars]
        if len(tostack)>0:
            self.incloud_vars = np.stack(tostack).reshape(-1)
        else:
            self.incloud_vars =np.array([],dtype=bool)
            
        #get the grid-box-average variables
        tostack=[np.argwhere(self.source_vars==x)
                                          for x in [0,1,2,3,4,5,9,10,11,12,13,14,15,16]
                                           if x in self.source_vars]
        if len(tostack)>0:
            self.gba_vars = np.stack(tostack).reshape(-1)
        else:
            self.gba_vars = np.array([],dtype=bool)
            
            
    def __len__(self):
        return min(len(self.source_files),len(self.target_files))

    def __getitem__(self,idx):
        try:
            source = np.load(self.source_files[idx])
        except FileNotFoundError:
            return
        s_locations = source["locs"]
        source = source["props"]
        assert source.shape[1]>source.shape[0],source.shape
        assert source.shape[2]>source.shape[1],source.shape
        assert len(source.shape)==3, source.shape

        if self.roll:
            roll_lon = np.random.randint(0,170,1)
        else:
            roll_lon=0
        
        stride = int((source[:,::self.chunksize].shape[1]-self.tilesize)/2)
        
        if np.any(np.all(np.isnan(source),axis=(1,2))):
            os.rename(self.source_files[idx],self.source_files[idx]+".error")
            return 
        
        if 10 in self.source_vars:
            #remove 0 temperature values, doesnt work if temp is not the last var
            t= source[10]
            
            idx_bad = np.argwhere(np.isnan(t))
            #this imputes valid values of 3x3 tsurf patches minesweeper style
            #if there is a huge patch of bad pixels they will be filled in outside to inside
            while len(idx_bad)>0 or np.any(np.isnan(source[10])):
                idx_x=idx_bad[:,0]
                idx_y=idx_bad[:,1]
                xp1 = np.where(idx_x<source.shape[1]-1,idx_x+1,idx_x)
                xm1 = np.where(idx_x>0,idx_x-1,idx_x)
                yp1 = np.where(idx_y<source.shape[2]-1,idx_y+1,idx_y)
                ym1 = np.where(idx_y>0,idx_y-1,idx_y)
                box = np.stack([t[idx_x,yp1],t[idx_x,ym1],t[xp1,idx_y],t[xp1,yp1],t[xp1,ym1],t[xm1,yp1],t[xm1,idx_y],t[xm1,ym1]])
                idx_x=idx_x[np.any(box>0,0)]
                idx_y=idx_y[np.any(box>0,0)]
                box = box[:,np.any(box>0,0)]
                
                box_av = np.nansum(box,0)/np.sum(box>0,0)
                assert np.all(box_av>0), box_av
                
                source[10,idx_x,idx_y]=box_av 
                t=np.where(np.isnan(source[10]),0,source[10])
                idx_bad = np.argwhere(t<=0)       
        
        source = source[self.source_vars]
        
        source = np.where(source<=-99,0,source)
        if self.chunksize>1:
            if len(self.incloud_vars)>0:
                cloud_amount = ~np.isnan(source)[self.incloud_vars].reshape(-1,
                                                                       source.shape[1],
                                                                       source.shape[2])
            else:
                cloud_amount = np.zeros((1,source.shape[1], source.shape[2]))
            cloud_amount = np.nancumsum(cloud_amount, axis=1)
            cloud_amount = np.nancumsum(cloud_amount, axis=2)
            #not using doubles causes pretty heavy rounding errors
            temp = np.nancumsum(source, axis=1, dtype = np.float64)
            temp = np.nancumsum(temp, axis=2, dtype=np.float64)

            locstemp = np.nancumsum(s_locations, axis=1, dtype=np.float64)
            locstemp = np.nancumsum(locstemp, axis=2, dtype=np.float64)
            #gets the grid box sums form the cumsums, which can then be averaged
            temp = self.extract(np.vstack((temp,locstemp, cloud_amount)))
            cloud_amount = temp[-1]
            s_locations = temp[-4:-1]/(self.chunksize**2)
            source = temp[:-4]
            
            del temp,locstemp
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                #stuff that is only defined if cloud is present is averaged by amount of cloudy pixels
                source[self.incloud_vars] = source[self.incloud_vars]/cloud_amount
            #the others are averaged by gridbox size
            source[self.gba_vars] =source[self.gba_vars]/(self.chunksize**2)

        source = np.where(np.isnan(source),-1,source)
        #rotates the globe longditudinally by a random amount to prevent oversampling
        s_locations = np.roll(s_locations, axis=2,shift = roll_lon)
        source = np.roll(source, axis=2,shift = roll_lon)
        #makes patches
        source = np.copy(window_nd(source, self.tilesize,stride,axis=(1,2))).astype("float32")
        s_locations = np.copy(window_nd(s_locations, self.tilesize,stride,axis=(1,2))).astype("float32")

        #doing the same thing now for the ICON data
        target = np.load(self.target_files[idx])
        t_locations = target["locations"]
        target = target["properties"]
        t_locations = np.roll(t_locations, axis=2,shift = roll_lon)
        target = np.roll(target, axis=2,shift = roll_lon)
        
        #make total water path
        target = np.vstack((target[np.newaxis,1]+target[np.newaxis,0],
                           target))

        target[5] /=100 #from Pa to hPa
        target[5] = np.where(np.isnan(target[5]), -1,target[5])
        target[0] *=1000 #from kg/m^2 to g/m^2
        target[2] *= 1000
        target[1] *=1000
        target[7] = np.where(target[7]>200,200,target[7])

        #remove 0 temperature values
        t=np.where(target[6]<=0,np.nan,target[6]) # maybe dangerous if this is not acutally the right index
        idx_bad = np.argwhere(np.isnan(t))

        while len(idx_bad)>0:
            idx_x=idx_bad[:,0]
            idx_y=idx_bad[:,1]
            xp1 = np.where(idx_x<target.shape[1]-1,idx_x+1,idx_x)
            xm1 = np.where(idx_x>0,idx_x-1,idx_x)
            yp1 = np.where(idx_y<target.shape[2]-1,idx_y+1,idx_y)
            ym1 = np.where(idx_y>0,idx_y-1,idx_y)
            box = np.stack([t[idx_x,yp1],t[idx_x,ym1],t[xp1,idx_y],t[xp1,yp1],t[xp1,ym1],t[xm1,yp1],t[xm1,idx_y],t[xm1,ym1]])
            idx_x = idx_x[np.any(box>0,0)]
            idx_y = idx_y[np.any(box>0,0)]
            box = box[:,np.any(box>0,0)]
            box_av = np.nansum(box,0)/np.sum(box>0,0)
            assert np.all(box_av>0),box_av
            target[6,idx_x,idx_y] = box_av #watch because I might move input variable number or order
            t=np.where(target[6]<=0,np.nan,target[6])
            idx_bad = np.argwhere(np.isnan(t))

        target = target[self.target_vars]
        target = np.where(target<-990,0,target)
        target = np.where(np.isnan(target),0,target)
        
        target = np.copy(window_nd(target, self.tilesize,stride, axis=(1,2))).astype(np.float32)
        t_locations = np.copy(window_nd(t_locations, self.tilesize, stride,axis=(1,2))).astype("float32")
        target = target.reshape((-1,)+target.shape[2:])
        
        t_locations = t_locations.reshape((-1,)+t_locations.shape[2:])
        source = source.reshape((-1,)+source.shape[2:])
        s_locations = s_locations.reshape((-1,)+s_locations.shape[2:])
        #excludes samples that dont have at least 80% "daytime" pixels
        #daytime means that all fluxes are nonzero
        if self.daytime:
            choice_s = np.all(np.sum(source[:,-5:]>0,axis=(2,3))/np.prod(source.shape[2:])>0.8,             
                                axis=1)
            choice_t = np.all(np.sum(target[:,-5:]>0,axis=(2,3))/np.prod(target.shape[2:])>0.8,
                            axis=1)
            choice_both=choice_s*choice_t
            s_locations = s_locations[choice_both]
            t_locations = t_locations[choice_both]
            target = target[choice_both]
            source = source[choice_both]
        l=min(len(target),len(source))
        
        assert np.all(target[:,-5]<1000),target.max(axis=(0,2,3))
        return {"values":(source[:l], target[:l]) , "locs":(s_locations[:l], t_locations[:l])}

    def __str__(self):
        return 'ComboSet'
    
    def get_from_same_week(self,week,namelist,weeklist):
        if week in weeklist:
            wl = np.array(weeklist)
            same = np.argwhere(wl==week).squeeze()
            if len(same)>0:
                choice = np.random.choice(same)
                return namelist[choice]

    def get_week(self,name):
        for i in range(len(name)-9):
            if name[i]=="_":
                try:
                    year = int(name[i+1:i+5])
                    month = int(name[i+5:i+7])
                    day = int(name[i+7:i+9])
                    assert year>1800,year
                    assert month<13,month
                    assert day<32,day
                except Exception:
                    pass
        return datetime.date(year,month,day).isocalendar()[1]

    def extract(self,arr):
        """
            gets the sum of each gridbox. as each element in the array is
            the sum of all the elements up until that (in dim 1 and 2), the2023 Copyright. All Rights Reserved.

            elments outside of the grid box need to be subtracted to get only the
            grid sum. because the top left is subtracted twice, we add it again

            Parameters
            ----------
            arr : array, (n,x,y)
                recursive sum array

            Returns
            -------
            array, (n, (x-d+1),(y-d+1))
            sums of overlapping grid boxes

        """
        arr = np.pad(arr, ((0,0),(1,0), (1,0)),constant_values=0)

        d=self.chunksize    
        return arr[:,d::d,d::d]+arr[:,:-d:d,:-d:d]-arr[:,:-d:d,d::d]-arr[:,d::d,:-d:d]


if __name__=="__main__":
    work =os.environ["WORK"]
    scratch = os.environ["SCR"]
    s_path = os.path.join(scratch, "ESACCI/npz_daily")
    t_path = os.path.join(scratch, "ICON_output/threshcodp2/numpy")
    varnames = ["cwp","lwp","iwp","cerl","ceri","cod","ptop","tsurf","clt","rlut","rsut","rsdt",
                                    "rsutcs","rlutcs"]
    try:
        dsname=sys.argv[1] 
    except Exception:                               
        dsname = "ESAvsICON_small.hdf5"
    foldername = "dataset_threshcodp2"
    if not os.path.exists(os.path.join(work,foldername)):
        os.makedirs(os.path.join(work,foldername))
    filename = os.path.join(work,foldername,dsname)
    if "small" in dsname:
        if os.path.exists(filename):
            os.remove(filename)
    ds = CombinedDataset(s_path,
                         t_path,
                         varnames,
                         20,
                         64,
                         False,
                         daytime=True,
                         roll=True)
    assert len(ds)>0
    h5file = h5py.File(filename, "a")
    start=True
    bs = 60
    #because some stuff gets excluded the DL needs to be able to handle a variable amount of outputs or none at all
    dl=DataLoader(ds,batch_size=bs, num_workers=bs,collate_fn=nonehandling_collate)
    with warnings.catch_warnings(): 
        warnings.simplefilter("default")
        #yeah this next part is atrocious
        for i,(v,l) in tqdm(enumerate(dl)):
            if v is None or l is None:
                continue
            if np.any( [x==0 for x in v[0].shape]):
                #this is a failure handler for when i cant return None because of pytorch
                continue
            #0 source, 1 target
            v=list(v)
            l=list(l)
            assert np.all([x==y for x,y in zip(v[0].shape,v[1].shape)]),(v[0].shape== v[1].shape)
            assert np.all([x==y for x,y in zip(l[0].shape,l[1].shape)]),(l[0].shape== l[1].shape)
            #t for toa (fluxes)
            t_0=v[0][:,-5:]
            t_1=v[1][:,-5:]
            assert np.all(t_1<2000), t_1.max(axis=(0,2,3))
            #the non flux properties
            v[0]=v[0][:,:-5]
            v[1]=v[1][:,:-5]
            #the locations
            l[0]=l[0]
            l[1]=l[1]
            assert len(l[0])==len(v[0]), (l[0].shape, v[0].shape)
            assert len(l[1])==len(v[1]), (l[1].shape, v[1].shape)
            assert len(l[1])==len(v[0]), (l[1].shape, v[0].shape)
            assert len(l[0])==len(v[1]), (l[0].shape, v[1].shape)
            assert len(v[0])==len(v[0]), (v[0].shape, v[0].shape)
            assert len(v[1])==len(v[1]), (v[1].shape, v[1].shape)
            assert len(l[0])==len(l[0]), (l[0].shape, l[0].shape)
            assert len(l[1])==len(l[1]), (l[1].shape, l[1].shape)

            if start:
                h5file.create_dataset('source', data=v[0], 
                        compression="gzip", chunks=True, maxshape=(None,v[0].shape[1],64,64))
                h5file.create_dataset('target', data=v[1], 
                        compression="gzip", chunks=True, maxshape=(None,v[1].shape[1],64,64))
                h5file.create_dataset('s_locs', data=l[0], 
                        compression="gzip", chunks=True, maxshape=(None,l[0].shape[1],64,64))
                h5file.create_dataset('t_locs', data=l[1], 
                        compression="gzip", chunks=True, maxshape=(None,l[1].shape[1],64,64))
                h5file.create_dataset('s_toa', data=t_0, 
                        compression="gzip", chunks=True, maxshape=(None,t_0.shape[1],64,64))
                h5file.create_dataset('t_toa', data=t_1, 
                        compression="gzip", chunks=True, maxshape=(None,t_1.shape[1],64,64))
                start=False
            else:
                # Append new data to it
                h5file['source'].resize((h5file['source'].shape[0] + v[0].shape[0]), axis=0)
                h5file['source'][-v[0].shape[0]:] = v[0]

                h5file['target'].resize((h5file['target'].shape[0] + v[1].shape[0]), axis=0)
                h5file['target'][-v[0].shape[0]:] = v[1]
                h5file['s_locs'].resize((h5file['s_locs'].shape[0] + l[0].shape[0]), axis=0)
                h5file['s_locs'][-l[0].shape[0]:] = l[0]

                h5file['t_locs'].resize((h5file['t_locs'].shape[0] + l[1].shape[0]), axis=0)
                h5file['t_locs'][-l[1].shape[0]:] = l[1]
                h5file['s_toa'].resize((h5file['s_toa'].shape[0] + t_0.shape[0]), axis=0)
                h5file['s_toa'][-t_0.shape[0]:] = t_0

                h5file['t_toa'].resize((h5file['t_toa'].shape[0] + t_1.shape[0]), axis=0)
                h5file['t_toa'][-t_0.shape[0]:] = t_1
            assert h5file["s_locs"].shape[0]==h5file["source"].shape[0],(i,l[0].shape,v[0].shape,h5file["s_locs"].shape,h5file["source"].shape)
            
            if "small" in dsname:
                if i*bs>50:
                    print("breaksmall")
                    break
            if "mid" in dsname:
                if i> len(dl)/2:
                    print("breakmid")
                    break
        

        h5file.close()
        time.sleep(5)
        #sanitycheck
        ds = h5py.File(os.path.join(work,foldername,dsname),"r+") 
        source=ds["source"]
        target=ds["target"]
        s=source[:].copy()
        t=target[:].copy()
        t_locs=ds["t_locs"][...]
        s_locs=ds["s_locs"][...]
        assert len(t_locs)==len(t),(t_locs.shape, t.shape)
        assert len(s_locs)==len(s),(s_locs.shape,s.shape)
        idx = varnames.index("ptop")
        for v in range(s.shape[1]):
            if v!=idx:
                s[:,v]=np.where(np.abs(s[:,v])<1e-8,0,s[:,v])
                t[:,v]=np.where(np.abs(t[:,v])<1e-8,0,t[:,v])
                assert np.all(s[:,v]>=0), (varnames[v], np.min(s[:,v]))
                assert np.all(t[:,v]>=0), (varnames[v], np.min(t[:,v]))
        max_ptop = max(np.max(s[:,idx]), np.max(t[:,idx]),1100.666)
        s[:,idx]=np.where(s[:,idx]<0,max_ptop,s[:,idx])
        t[:,idx]=np.where(t[:,idx]<0,max_ptop,t[:,idx])
        print("source", s.mean((0,2,3)))
        print("target", t.mean((0,2,3)))
        source[...]=s
        target[...]=t
        s_mi=np.min([np.min(s, axis=(0,2,3)),np.min(t,axis=(0,2,3))],axis=0)
        s_mi = np.where(s_mi>0,np.log(s_mi+1),s_mi) 
        s_ma = np.max([np.max(s,axis=(0,2,3)),np.max(t,axis=(0,2,3))],axis=0)
        s_ma = np.where(s_ma>0, np.log(s_ma+1),s_ma)
        print(s_mi,s_ma)
        ds.close()
        ds = h5py.File(os.path.join(work,foldername,dsname),"r") 
        assert np.allclose(ds["source"],s)
        assert np.allclose(ds["target"],t)
        ds.close()
