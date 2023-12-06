import tensorflow as tf
from dagans import dataloader
import numpy as np
import os
from dagans.dataloader import augmentation_wrapper as wrapper
from datetime import datetime
import sys
import time
from dagans.models_toa import generator_simple,  model_wrapper_wclt,train


import traceback
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"

                
if __name__=="__main__":
    assert len(tf.config.list_physical_devices("GPU"))>0
    start=time.time()
    if len(sys.argv)>1:
        with open(os.path.join(os.environ["HOME"],"pvl8dagans-master","experiment_log.txt"),"a+") as file:
            print(os.environ["SLURM_JOBID"]+" "+ str(datetime.today()) +": "+ str(sys.argv), file=file)
    else:
        raise Exception("forgot exp_name")

    work = os.environ["WORK"]
    scratch = os.environ["SCR"]
    
    shape = 32 #patchsize
    variables = np.arange(9)
    features = np.arange(6)#5 fluxes and clt
    
    
    model = generator_simple(shape=(shape,shape,(len(variables)-1)*2),dropout=True, 
                             output_shape=(shape,shape,len(features)))
    model.summary()
    loss_fct = tf.keras.losses.MeanSquaredError()
    outpath = os.path.join(os.environ["WORK"],"models","test_model_wcltcod")
    try:
        model = tf.keras.models.load_model(outpath)
        print("loading model")
    except OSError:
        traceback.print_exc()
    #the wrapper handles normalization and selection of variables and returns the forward pass
    model_fct = model_wrapper_wclt(model,loss_fct, channels=features,dsfolder="dataset_threshcodp2")
    
    steptime = time.time()
    
    print("Model setup took {:.3f} seconds".format(steptime-start))
    path_cache = os.path.join(work, "dataset_threshcodp2", "ESAvsICON.hdf5")
    assert os.path.exists(path_cache), "File %s does not exists" % path_cache
    dataset_train, len_dataset_train,val_ds = dataloader.get_dataset_inmemory_wclt(path_cache,
                                        variables=variables, shuffle=False)
    
    print("Inmemory took {:.3f} seconds".format(time.time()-steptime))
    steptime = time.time()
    batched_ds = dataloader.make_batches(dataset_train,
                                            data_augmentation_fun=[
                                                wrapper(dataloader.logn_augmentation, cutoff=2),
                                                wrapper(dataloader.minmax_data_augmentation_log,folder=os.path.dirname(path_cache),
                                                            variables=variables,cutoff=2),
                                                wrapper(dataloader.patches_data_augmentation,s=shape,cutoff=6),
                                                ],
                                             batch_size=min(len(dataset_train)//10,200),
                                             repeat=False )      
    
    print("Main batching took {:.3f} seconds".format(time.time()-steptime))
    steptime = time.time()
    val_ds = dataloader.make_batches(val_ds,
                                             data_augmentation_fun=[
                                                 wrapper(dataloader.logn_augmentation, cutoff=2),
                                                 wrapper(dataloader.minmax_data_augmentation_log,
                                                            folder=os.path.dirname(path_cache),
                                                             variables=variables,cutoff=2),
                                                 wrapper(dataloader.patches_data_augmentation,s=shape,cutoff=6),
                                                 ],
                                             batch_size=min(len(val_ds),200),
                                             repeat=False )                                          
    
    print("Val batching took {:.3f} seconds".format(time.time()-steptime))
    lr=1e-4    
    #this is from models toa namespace import
    train(2000,model, model_fct, batched_ds, val_ds, os.path.join(scratch,"logs/cons/{}".format(sys.argv[1].strip(" ")[:10])),
            lr = lr)
    model.save(outpath)
    
