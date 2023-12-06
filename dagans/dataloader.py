import tensorflow as tf
import random
import h5py
import numpy as np
import os

def get_dataset_inmemory_wlocs(path_cache,variables, scaler=None,shuffle=True):
    """load complete dataset , the validation part also contains temporal and geographical info"""
    variables = np.array(variables)
    with h5py.File(path_cache, "r") as dat:
        ds_size_lim = 75000
        source = dat["source"]
        source = source[:ds_size_lim,variables,...]
        target = dat["target"]
        target = target[:ds_size_lim,variables,...]
        s_locs = dat["s_locs"][:ds_size_lim,...]
        t_locs = dat["t_locs"][:ds_size_lim,...]
        s_toa = dat["s_toa"][:ds_size_lim,...]
        t_toa = dat["t_toa"][:ds_size_lim,...]

        len_ds = min(target.shape[0],len(s_locs))
        tuple_datasets = (target[:int(len_ds*0.6)].transpose(0,2,3,1).astype('float32'),
                          source[:int(len_ds*0.6)].transpose(0,2,3,1).astype('float32'),
                          t_toa[:int(len_ds*0.6)].transpose(0,2,3,1).astype('float32'),
                          s_toa[:int(len_ds*0.6)].transpose(0,2,3,1).astype('float32'))
        pred_datasets = (target[int(len_ds*0.6):len_ds].transpose(0,2,3,1).astype('float32'),
                          source[int(len_ds*0.6):len_ds].transpose(0,2,3,1).astype('float32'),
                          t_toa[int(len_ds*0.6):len_ds].transpose(0,2,3,1).astype('float32'),
                          s_toa[int(len_ds*0.6):len_ds].transpose(0,2,3,1).astype('float32'),
                          t_locs[int(len_ds*0.6):len_ds].transpose(0,2,3,1).astype('float32'),
                          s_locs[int(len_ds*0.6):len_ds].transpose(0,2,3,1).astype('float32'))
    train_ds = tf.data.Dataset.from_tensor_slices(tuple_datasets)
    
    val_ds = tf.data.Dataset.from_tensor_slices(pred_datasets)
    if shuffle:
        return train_ds.shuffle(buffer_size=150000), int(0.6*len_ds),val_ds
    else:
        return train_ds, int(len_ds*0.6),val_ds

def get_dataset_inmemory_wclt(path_cache,variables, scaler=None,shuffle=True):
    """load complete dataset with the clt cotainend in the toa for the auxiliary toa model,
       the validation part also contains temporal and geographical info"""
    variables = np.array(variables)
    with h5py.File(path_cache, "r") as dat:
        ds_size_lim = 75000
        source = dat["source"]
        source = source[:ds_size_lim,variables,...]
        sclt = source[:,-1,np.newaxis]
        source = source[:,:-1]
        target = dat["target"]
        target = target[:ds_size_lim,variables,...]
        tclt = target[:,-1,np.newaxis]
        target = target[:,:-1]
        s_locs = dat["s_locs"][:ds_size_lim,...]
        t_locs = dat["t_locs"][:ds_size_lim,...]
        s_toa = dat["s_toa"][:ds_size_lim,...]
        t_toa = dat["t_toa"][:ds_size_lim,...]
        assert isinstance(s_toa,np.ndarray),type(s_toa)
        s_toa = np.concatenate((s_toa,sclt),axis=1)
        t_toa = np.concatenate((t_toa,tclt),axis=1)
        len_ds = min(target.shape[0],len(s_locs))
        tuple_datasets = (target[:int(len_ds*0.6)].transpose(0,2,3,1).astype('float32'),
                          source[:int(len_ds*0.6)].transpose(0,2,3,1).astype('float32'),
                          t_toa[:int(len_ds*0.6)].transpose(0,2,3,1).astype('float32'),
                          s_toa[:int(len_ds*0.6)].transpose(0,2,3,1).astype('float32'))
        pred_datasets = (target[int(len_ds*0.6):len_ds].transpose(0,2,3,1).astype('float32'),
                          source[int(len_ds*0.6):len_ds].transpose(0,2,3,1).astype('float32'),
                          t_toa[int(len_ds*0.6):len_ds].transpose(0,2,3,1).astype('float32'),
                          s_toa[int(len_ds*0.6):len_ds].transpose(0,2,3,1).astype('float32'),
                          t_locs[int(len_ds*0.6):len_ds].transpose(0,2,3,1).astype('float32'),
                          s_locs[int(len_ds*0.6):len_ds].transpose(0,2,3,1).astype('float32'))
    train_ds = tf.data.Dataset.from_tensor_slices(tuple_datasets)
    
    val_ds = tf.data.Dataset.from_tensor_slices(pred_datasets)
    if shuffle:
        return train_ds.shuffle(buffer_size=150000), int(0.6*len_ds),val_ds
    else:
        return train_ds, int(len_ds*0.6),val_ds
    

def get_dataset_inmemory_single(path_cache,variables, scaler=None, shuffle=False):
    """get only 2 samples from dataset"""
    variables = np.array(variables)
    with h5py.File(path_cache, "r") as dat:
        source = dat["source"]
        source = source[:2,variables,...]
        target = dat["target"]
        target = target[:2,variables,...]
        s_locs = dat["s_locs"][:2,...]
        t_locs = dat["t_locs"][:2,...]
        s_toa = dat["s_toa"][:2,...]
        t_toa = dat["t_toa"][:2,...]
        len_ds = min(target.shape[0],len(s_locs))
        tuple_datasets = (target[:].transpose(0,2,3,1).astype('float32'),
                          source[:].transpose(0,2,3,1).astype('float32'),
                          t_toa[:].transpose(0,2,3,1).astype('float32'),
                          s_toa[:].transpose(0,2,3,1).astype('float32'))
        pred_datasets = (target[:].transpose(0,2,3,1).astype('float32'),
                          source[:].transpose(0,2,3,1).astype('float32'),
                          t_toa[:].transpose(0,2,3,1).astype('float32'),
                          s_toa[:].transpose(0,2,3,1).astype('float32'),
                          t_locs[:].transpose(0,2,3,1).astype('float32'),
                          s_locs[:].transpose(0,2,3,1).astype('float32'))
    train_ds = tf.data.Dataset.from_tensor_slices(tuple_datasets)
    
    val_ds = tf.data.Dataset.from_tensor_slices(pred_datasets)
    if shuffle:
        return train_ds.shuffle(buffer_size=1500), int(len_ds),val_ds
    else:
        return train_ds, int(len_ds),val_ds
    

def d4_data_augmentation(dataset,cutoff=2):
    """
    Flips and mirrors the input "images"
    cutoff says on how many elements of the input tuple this should be applied
    I think this implementation flips every tuple member differently,
    which is fine for domains ands locs(because the dont get flipped for eval,
    but this is bad for toa I think. This would not have a reasonalble fix. 
    But toa performance is still good so either this doesnt actually happen this way
    or predicting the average is fine
    """
    seed = random.randint(0, 2 ** 31 - 1)

    def transform(image):
        r = image
        r = tf.image.random_flip_left_right(r, seed=seed)
        r = tf.image.random_flip_up_down(r, seed=seed)

        k = tf.random.uniform([],
                              dtype=tf.int32,
                              minval=0,
                              maxval=4,
                              seed=seed)

        r = tf.image.rot90(r,k=k)
        tf.debugging.assert_all_finite(r, "during data")
        return r

    return dataset.map(lambda *args: tuple(transform(a) if i< cutoff else a for i,a in enumerate(args)), num_parallel_calls=tf.data.AUTOTUNE
    )


def augmentation_wrapper(function, **kwargs):
    """maybe there is a much easier way to do this but it works"""
    return lambda x: function(x,**kwargs)


def patches_data_augmentation(dataset,s=32,cutoff=2, overlap=False, ds_shape=64):
    """turns the 64x64 patches into smaller ones
    cutoff says on how many elements of the tuple it should be applied
    overlap defines if the new patches should be allowed to overlap"""
    if overlap:
        stride = int((ds_shape-s)/(ds_shape//s))
    else:
        stride = s
    def transform(image):
        r = tf.expand_dims(image, axis=0)
        if s==r.shape[1]:
            pass
        else:
            out=[]
            for ch in range(r.shape[-1]):
                temp = tf.expand_dims(r[...,ch],axis=3)
                temp=tf.image.extract_patches(
                        temp,
                        sizes= [1,s,s,1],
                        strides= [1,stride,stride,1],
                        rates= [1,1,1,1],
                        padding="VALID", name=None
                        )
                temp = tf.reshape(temp,(-1,s,s))
                out.append(temp)
            r=tf.stack(out,axis=3)
        return r
    return dataset.map(lambda *args: tuple(transform(a) if i< cutoff else a for i,a in enumerate(args))
                    , num_parallel_calls=tf.data.AUTOTUNE
                                ).unbatch()

    


def minmax_data_augmentation_log(dataset,folder=os.path.join(os.environ["WORK"],"dataset_all"),
                                     variables=np.array([0,1,2,3,4,5,6,7]),
                                     cutoff = 2):
    """minmaxe scales data that has already been logscaled
        minmaxes are over both domains and extended for outliers in future data and to make sure no value is actually  1
    Args:
        dataset (tf.data.Dataset): dataset returning a tuple of length 4 or 6
        folder (string, optional): where the min and max values are saved. Defaults to os.path.join(os.environ["WORK"],"dataset_all").
        variables (np.ndarray, optional): indices of variables to be augmented. Defaults to np.array([0,1,2,3,4,5,6,7]).
        cutoff (int, optional): elements of the tuple to apply this to. Defaults to 2.

    Raises:
        ValueError: If the input shape is not the same as the loaded mins and maxs

    Returns:
        appropriately augmented dataset
    """    
    try:
        #load the minima and maxima
        s_mi,s_ma = np.load("{}/minmax_both.npy".format(folder))
        toa_mi,toa_ma = np.load("{}/minmax_toa.npy".format(folder))
    except FileNotFoundError:
        ds = h5py.File(os.path.join(folder, "ESAvsICON.hdf5"),"r")
        s=ds["source"][:]
        t=ds["target"][:]
        s_toa = ds["s_toa"][:]
        t_toa = ds["t_toa"][:]
        #create the minima and maxima
        s_mi=np.min([np.min(s, axis=(0,2,3)),np.min(t,axis=(0,2,3))],axis=0)
        s_mi = np.where(s_mi>0,np.log(s_mi+1),0)
        s_ma = np.max([np.max(s,axis=(0,2,3)),np.max(t,axis=(0,2,3))],axis=0)
        s_ma = np.where(s_ma>0, np.log(s_ma+1),0)
        toa_mi=np.min([np.min(s_toa, axis=(0,2,3)),np.min(t_toa,axis=(0,2,3))],axis=0)
        toa_mi = np.where(toa_mi>0,np.log(toa_mi+1),0)
        toa_ma = np.max([np.max(s_toa,axis=(0,2,3)),np.max(t_toa,axis=(0,2,3))],axis=0)
        toa_ma = np.where(toa_ma>0, np.log(toa_ma+1),0)
        print(toa_mi)
        np.save("{}/minmax_both.npy".format(folder),
                np.array([s_mi*0.99,s_ma*1.01]))    
        np.save("{}/minmax_toa.npy".format(folder),
                np.array([toa_mi*0.99,toa_ma*1.01]))

    s_mi = s_mi[variables]
    s_ma = s_ma[variables]
    s_mi = tf.constant(s_mi, dtype="float32")
    s_ma = tf.constant(s_ma, dtype="float32")
    toa_mi = tf.constant(toa_mi, dtype="float32")
    toa_ma = tf.constant(toa_ma, dtype="float32")
    
    def transform(image):
        r = image
        if r.shape[-1]==len(s_mi)-1:    
            #at one stage i removed clt from the input and made it diagnostic
            #via another NN, so its still in variables but does not get processed here
            #and thats why there is a -1
            r = r - s_mi[:-1]
            r = r/(s_ma[:-1] - s_mi[:-1])
        elif r.shape[-1]==len(toa_mi):
            r = r - toa_mi
            r = r/(toa_ma - toa_mi)
        else:
            raise ValueError((" sample has shape {} but I "
                             "expected either (...,{}) or (...,{})").format(r.shape,
                                                                           len(s_mi)-1,
                                                                            len(toa_mi))) 
        return r

    return dataset.map(lambda *args: tuple(transform(a) if i< cutoff else a for i,a in enumerate(args) ), num_parallel_calls=tf.data.AUTOTUNE
    )



def concat_augmentation(dataset):
    """concats some stuff, pretty sure i dont use it"""
    return dataset.map(lambda *args : tf.concat([a for a in args[:2]], axis=2), num_parallel_calls=tf.data.AUTOTUNE
            )


def logn_augmentation(dataset, cutoff = 2):
    """nat-logs the data, with a +1 so that the new minimum is 0. only applies to the first *cutoff* elements of the data"""
    def transform(image):
        return tf.where(image>0,tf.math.log1p(image),image)
    return dataset.map(lambda *args: tuple(transform(a) if i<cutoff else a for i,a in enumerate(args)), num_parallel_calls=tf.data.AUTOTUNE
    )



def make_batches(ds,
                 batch_size=32,
                 data_augmentation_fun=[],
                 drop_remainder=True,
                 repeat=True):
    """applies the all augmentations in the given list to the dataset, then maybe repeats

    Args:
        ds (tf.data.Dataset): the dataset from get_inmemory...
        batch_size (int, optional): jup. Defaults to 32.
        data_augmentation_fun (list, optional): the above augmentations IN ORDER. Defaults to [].
        drop_remainder (bool, optional): wether to drop the last batch, as its smaller and might confuse the model. Defaults to True.
        repeat (bool, optional): if the dataset size should be 'infinite'. Defaults to True.

    Returns:
        the augmened ds
    """    
    
    if len(data_augmentation_fun )>0:
        for fun in data_augmentation_fun:
            ds = fun(ds)

    
    if repeat:
        ds = ds.repeat(repeat)
    return ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.experimental.AUTOTUNE
    )


