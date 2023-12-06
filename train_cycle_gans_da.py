"""
main script for cyclegan training
most of the code is built upon https://github.com/IPL-UV/pvl8dagans
all the methods are explained in the other files
sorry for all the conditionals that make this hard to read
overall the process is simple: load/define the 4-5 models, load and preprocess the 2 datasets,
fit the cyclegan, do some preliminary plots
"""

import luigi
import os
from dagans import cycle_gan, util_model

from dagans import gan_model,  dataloader
from dagans.dataloader import augmentation_wrapper as wrapper
import tensorflow as tf
import numpy as np
import random
import logging
import itertools
import sys
from datetime import datetime
from dagans.apply_to_icon import appl
from luigi.task import flatten
import joblib
from datetime import datetime
import warnings
import traceback
# Set seed for sanity
tf.random.set_seed(10)
np.random.seed(10)
random.seed(10)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class TrainCycleGAN(luigi.Task):
    folder = luigi.Parameter(default="checkpoints_trained")
    dataset_folder = luigi.Parameter(description="Path to combined model/sat dataset")
    gradient_penalty_mode = luigi.ChoiceParameter(default="meschederreal", choices=["meschederreal","gulrajani", "none"])#meschederreal acutall makes no sense for wstein
    gradient_penalty_weight = luigi.FloatParameter(default=10.)
    df_gen = luigi.IntParameter(default=64)
    df_disc = luigi.IntParameter(default=8)
    epochs = luigi.IntParameter(default=110)
    seed = luigi.IntParameter(default=123)
    batch_size = luigi.IntParameter(default=10)
    lr = luigi.FloatParameter(default=1e-1)
    identity_lambda = luigi.FloatParameter(default=1.)
    water_path_lambda = luigi.FloatParameter(default=1.)
    cycled_lambda = luigi.FloatParameter(default=1.)
    gan_lambda = luigi.FloatParameter(default=1.)
    segmentation_consistency_lambda = luigi.FloatParameter(default=1.)
    suffix = luigi.Parameter(default="")
    normtype = luigi.ChoiceParameter(choices=["batchnorm", "instancenorm", "no"],
                                     default="batchnorm")
    variables = luigi.ListParameter(default=[0,1])
    appl = luigi.BoolParameter()#default False
    explog = luigi.Parameter(default=None)
    gen_groups = luigi.IntParameter(default=1)
    disc_groups = luigi.IntParameter(default=1)
    patch_size = luigi.IntParameter(default=64)
    dummy = luigi.BoolParameter()
    pretrain = luigi.IntParameter(default=0) #default no pretraining
    resume = luigi.BoolParameter()
    timelimit = luigi.Parameter("02:00:00")
    #discriminator block activations
    activation = luigi.ChoiceParameter(choices=["relu","elu","softmax","sigmoid","swish","tanh","softplus","linear"] ,default=None)
    drop = luigi.BoolParameter()#dropout
    reg = luigi.BoolParameter()#l2regularization of generator weights
    wstein = luigi.BoolParameter()#should always be set, standard gan implementation is removed here
    const = luigi.FloatParameter(default=1e-2)#cutoff for lipschitz consistency
    full = luigi.BoolParameter()#if the full dataset should be used
    clr = luigi.BoolParameter(default=True,parsing=luigi.BoolParameter.EXPLICIT_PARSING)#cyclic learning rate
    tune = luigi.BoolParameter(default=False,parsing=luigi.BoolParameter.EXPLICIT_PARSING)#other stuff is returned when HPO
    tune_name = luigi.Parameter(default="dagans")#pretty sure thats unneccesary, but makes sure that stuff isnt overwritten
    n_critic = luigi.IntParameter(default=5)#num gen updates per critic update
    task_complete = False # need that because i have my own definition of completeness


    def experiment_name(self):
        if self.suffix != "":
            suffix = "_"+self.suffix
        else:
            suffix = ""
        return "cycle_%d_%d%s" % (self.df_gen, self.df_disc, suffix)

    def output(self):
       #defines output that luigis checks for existence
        if not self.appl:
            gentargetsource_name = luigi.LocalTarget(os.path.join(self.folder, "gentarget2source%s.hdf5" % self.experiment_name()))
            discsource_name = luigi.LocalTarget(os.path.join(self.folder, "discsource%s.hdf5" % self.experiment_name()))
            gensourcetarget_name = luigi.LocalTarget(os.path.join(self.folder, "gensource2target%s.hdf5" % self.experiment_name()))
            disctarget_name = luigi.LocalTarget(os.path.join(self.folder, "disctarget%s.hdf5" % self.experiment_name()))
            if not self.resume:
                ls = [gentargetsource_name, discsource_name, gensourcetarget_name, disctarget_name]
                if self.tune:
                    ls.append(luigi.LocalTarget(os.path.join(os.environ["SCR"],"ray_results",self.tune_name,"models",self.experiment_name())))
                return ls
            else:
                resumed = luigi.LocalTarget(os.path.join(self.folder, "resumed{}_{}.txt".format( self.experiment_name(), str(datetime.today()))))
                return [gentargetsource_name, discsource_name, gensourcetarget_name, disctarget_name, resumed]
    
    def complete(self):
        #makes sure luigi doesnt just quit if appl is run
        if self.appl:
            return  self.task_complete
        else:
            outputs = flatten(self.output())
            if len(outputs) == 0:
                warnings.warn(
                    "Task %r without outputs has no custom complete() method" % self,
                    stacklevel=2
                )
                return False

            return all(map(lambda output: output.exists(), outputs))

    def run(self):
        #main method 
        if self.explog is not None:
            with open(os.path.join(os.environ["HOME"],"pvl8dagans-master","experiment_log.txt"),"a+") as file:
                print(os.environ["SLURM_JOBID"]+" "+ str(datetime.today()) +": "+ str(sys.argv), file=file)
        else:
            raise Exception("forgot exp_name")
        timelimit = float(self.timelimit[:2])*3600+float(self.timelimit[3:5])*60-1501
        # Set seed for sanity
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        shape = (self.patch_size,self.patch_size)
        random.seed(self.seed)

        if (not self.wstein) or (self.gradient_penalty_mode == "gulrajani" and self.const==1e-2):
            #if not wstein we dont need lipschitz consistency
            #if i use GP with standard const, i am resetting const
            self.const=None
        mirrored_strategy = tf.distribute.MirroredStrategy()
            

        print("using {} GPUS".format(mirrored_strategy.num_replicas_in_sync), flush=True)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        input_shape_gen = shape + (len(self.variables)-1,) # if for any reason these might differ
        input_shape_disc = shape + (len(self.variables)-1,)
        
        if self.full:
            if self.tune:
                path_cache = os.path.join(self.dataset_folder, "ESAvsICON_middle.hdf5")
            else:
                path_cache = os.path.join(self.dataset_folder, "ESAvsICON.hdf5")
            assert os.path.exists(path_cache), "File %s does not exists" % path_cache
            repeat = 2
            dataset_train, len_dataset_train,val_ds = dataloader.get_dataset_inmemory_wclt(path_cache, # this thing should account for clt not being adapted but used in CRE-NN
                                                                                           variables=self.variables,
                                                                                             shuffle=True)
        else:
            path_cache = os.path.join(self.dataset_folder, "ESAvsICON_small.hdf5")
            assert os.path.exists(path_cache), "File %s does not exists" % path_cache
            repeat=4000
            dataset_train, len_dataset_train,val_ds = dataloader.get_dataset_inmemory_single(path_cache,
                                                                                             variables=self.variables[:-1],
                                                                                               shuffle=True)
        logging.info("Loaded dataset file %s. %d pseudo-simultaneous pairs" % (path_cache, len_dataset_train))
        
        assert len_dataset_train>0
        with mirrored_strategy.scope():
            if self.resume:
                try:
                    outs = self.output()
                    
                    gentarget2source = tf.keras.models.load_model(outs[0].path, 
                        custom_objects = {"InstanceNormalization": util_model.InstanceNormalization,
                                        "ClipConstraint": gan_model.ClipConstraint} )
                                        
                    disc_source = tf.keras.models.load_model(outs[1].path, 
                        custom_objects = {"InstanceNormalization": util_model.InstanceNormalization,
                                        "ClipConstraint": gan_model.ClipConstraint})
                                        
                    gensource2target = tf.keras.models.load_model(outs[2].path, 
                        custom_objects = {"InstanceNormalization": util_model.InstanceNormalization,
                                        "ClipConstraint": gan_model.ClipConstraint})
                                        
                    disc_target = tf.keras.models.load_model(outs[3].path, 
                        custom_objects = {"InstanceNormalization": util_model.InstanceNormalization,
                                        "ClipConstraint": gan_model.ClipConstraint})
                    print("loaded last")
                except OSError:
                    traceback.print_exc()
                    
                    gentarget2source = tf.keras.models.load_model(outs[0].path[:-5]+"best.hdf5",
                        custom_objects = {"InstanceNormalization": util_model.InstanceNormalization,
                                            "ClipConstraint": gan_model.ClipConstraint})
                                            
                    disc_source = tf.keras.models.load_model(outs[1].path[:-5]+"best.hdf5",
                        custom_objects = {"InstanceNormalization": util_model.InstanceNormalization,
                                            "ClipConstraint": gan_model.ClipConstraint})
                                            
                    gensource2target = tf.keras.models.load_model(outs[2].path[:-5]+"best.hdf5",
                                    custom_objects = {"InstanceNormalization": util_model.InstanceNormalization,
                                                    "ClipConstraint": gan_model.ClipConstraint})
                                                    
                    disc_target = tf.keras.models.load_model(outs[3].path[:-5]+"best.hdf5",
                        custom_objects = {"InstanceNormalization": util_model.InstanceNormalization,
                                            "ClipConstraint": gan_model.ClipConstraint})

                    print("loaded 'best'")
            

            elif not self.appl:
                
                disc_source = gan_model.disc_model(variables=self.variables[:-1],shape=input_shape_disc, df=self.df_disc*self.disc_groups,
                                            depth=1,
                                            activation=self.activation,normalization=False,
                                            groups=self.disc_groups,
                                            const=self.const)
                                            
                disc_target = gan_model.disc_model(variables=self.variables[:-1],shape=input_shape_disc, df=self.df_disc*self.disc_groups,
                                            activation=self.activation,normalization=False,
                                            depth=1,
                                            groups=self.disc_groups,
                                            const=self.const)
                                            
                gentarget2source = gan_model.generator_simple(variables=self.variables[:-1],shape=input_shape_gen, df=self.df_gen*self.gen_groups,
                                                    normtype=self.normtype,
                                                    normalization=False,
                                                    groups=self.gen_groups,
                                                    dropout=self.drop,
                                                    l2reg=self.reg)
                                                    
                gensource2target = gan_model.generator_simple(variables=self.variables[:-1],shape=input_shape_gen, df=self.df_gen*self.gen_groups,
                                                    normtype=self.normtype,
                                                    normalization=False,
                                                    groups=self.gen_groups,
                                                    dropout=self.drop,
                                                    l2reg=self.reg)
                if not self.tune:
                    print("models initialized")  
                                                    
            else :
                print("dont need models")  
            if self.resume or not self.appl:       
                with tf.device("GPU:1"):#this doesnt seem to do anything but hey
                    #loads the model that predicts both CREs and cloud cover
                    toa_model  = gan_model.model_setup(names = ["wclt"],
                                                        fwp = "CRE_clt",
                                                        folder = self.dataset_folder)
                    
                
                
                cyclegan = cycle_gan.WsteinGAN(gensourcetarget=gensource2target, gentargetsource=gentarget2source, 
                                            discsource=disc_source, disctarget=disc_target,
                                            datafolder = self.dataset_folder,
                                            batch_size=self.batch_size,
                
                                            gradient_penalty_fun=gan_model.gradient_penalty_fun(self.gradient_penalty_mode,center=1),
                                            lr=self.lr,
                                            eps=self.const,clr=self.clr,
                                            steps_for_clr = len_dataset_train/self.batch_size*4,
                                            toa_model = toa_model,
                                            water_path_lambda = self.water_path_lambda,
                                            verbose= not self.tune)
        if self.appl:
            assert len(dataset_train)>0
            batched_ds = dataloader.make_batches(val_ds,
                                             data_augmentation_fun=[
                                                 wrapper(dataloader.logn_augmentation, cutoff=2),
                                                 wrapper(dataloader.minmax_data_augmentation_log,folder=self.dataset_folder,
                                                             variables=np.array(self.variables),cutoff=2),
                                                 wrapper(dataloader.patches_data_augmentation,s=self.patch_size,cutoff=6,overlap=True),
                                                 ],
                                             batch_size=1,
                                             repeat=2).with_options(options)
            outs = self.output() 
            print("len datasets after augmentation",len(batched_ds),len(val_ds))
            assert len(batched_ds)>0
            if (not (os.path.exists(os.path.join(os.environ["WORK"],"pickle","source_gentarget2source%sbest.hdf5" % self.experiment_name()) ) ) and 
                    (os.path.exists(os.path.join(os.environ["WORK"],"pickle","source_gentarget2source%s.hdf5" % self.experiment_name()) ))):
                os.remove(os.path.join(os.environ["WORK"],"pickle","source_gentarget2source%s.hdf5" % self.experiment_name()) )
            appl(batched_ds,"gentarget2source%sbest.hdf5" % self.experiment_name(),self.variables[:-1],self,max(timelimit-600,60)/2)
            appl(batched_ds,"gentarget2source%s.hdf5" % self.experiment_name(),self.variables[:-1],self,max(timelimit-600,60)/2)
            
            print("applied")
            self.task_complete = True

                    
        else:
            if not self.tune:
                print(len(dataset_train),self.batch_size)
            batched_ds = dataloader.make_batches(dataset_train,
                                             data_augmentation_fun=[
                                                 #dataloader.clipping_augmentation,
                                                 wrapper(dataloader.logn_augmentation, cutoff=2),
                                                 wrapper(dataloader.minmax_data_augmentation_log,folder=self.dataset_folder,
                                                                    variables=np.array(self.variables),cutoff =2),
                                                 wrapper(dataloader.patches_data_augmentation,s=self.patch_size,cutoff=6,overlap=True),
                                                 wrapper(dataloader.d4_data_augmentation,cutoff=6)
                                                 ],
                                             batch_size=self.batch_size,
                                             repeat=repeat).with_options(options)    
            if val_ds is None:
                val_ds_batched = batched_ds.shuffle(len(batched_ds))
            else:
                
                val_ds_batched = dataloader.make_batches(val_ds,
                                             data_augmentation_fun=[
                                                 #dataloader.clipping_augmentation,
                                                 wrapper(dataloader.logn_augmentation,cutoff=2),
                                                 wrapper(dataloader.minmax_data_augmentation_log,
                                                            folder=self.dataset_folder,variables=np.array(self.variables),
                                                            cutoff =2),
                                                 wrapper(dataloader.patches_data_augmentation,s=self.patch_size, cutoff=6,overlap=True),
                                                 wrapper(dataloader.d4_data_augmentation, cutoff=6)
                                                 ],
                                             batch_size=min(len(val_ds),self.batch_size),
                                             repeat=1).with_options(options)    
            appl_ds = dataloader.make_batches(val_ds,
                                    data_augmentation_fun=[
                                        wrapper(dataloader.logn_augmentation, cutoff=2),
                                        wrapper(dataloader.minmax_data_augmentation_log,folder=self.dataset_folder,
                                             variables=np.array(self.variables), cutoff=2),
                                        wrapper(dataloader.patches_data_augmentation,s=self.patch_size,cutoff=6,overlap=True),
                                        ],
                                    batch_size=1,
                                    repeat=2).with_options(options)
            

            assert len(batched_ds)>0
            steps=len(batched_ds)
            if not self.tune:
                print("len dataset after augmentation",len(batched_ds),len(val_ds_batched))
            batched_ds = mirrored_strategy.experimental_distribute_dataset(batched_ds)
            
            if not self.resume:
                logdir = os.path.join(os.environ["SCR"],"logs/{}/run_1".format(self.experiment_name()))
            else:
                dt=datetime.now()
                logdir = os.path.join(os.environ["SCR"],"logs/{}/run_{}".format(self.experiment_name(),dt.strftime("%d%m%y%H%M")))
            
            outs = self.output()
            assert timelimit>0,timelimit
            if not self.wstein:
                cycle_gan.fit(train_ds=batched_ds, obj_cyclegan=cyclegan, 
                          strategy=mirrored_strategy,
                          steps_per_epoch=steps, 
                          val_ds=val_ds_batched,
                          gan_lmbda = self.gan_lambda,
                          identity_lmbda=self.identity_lambda,
                          cycled_lmbda=self.cycled_lambda,
                          segmentation_consistency_lmbda=self.segmentation_consistency_lambda,
                          gradient_penalty_weight=self.gradient_penalty_weight,
                          logdir=logdir,
                          epochs=self.epochs
                          ,dummy=self.dummy,
                          pretrain=self.pretrain,
                          timelimit = timelimit,
                          outs=outs)
            else:
                cycle_gan.fit_wstein(train_ds=batched_ds, obj_cyclegan=cyclegan, 
                          strategy=mirrored_strategy,
                          steps_per_epoch=steps, 
                          val_ds=val_ds_batched,
                          gan_lmbda = self.gan_lambda,
                          identity_lmbda=self.identity_lambda,
                          cycled_lmbda=self.cycled_lambda,
                          segmentation_consistency_lmbda=self.segmentation_consistency_lambda,
                          gradient_penalty_weight=self.gradient_penalty_weight,
                          logdir=logdir,
                          epochs=self.epochs
                          ,dummy=self.dummy,
                          pretrain=self.pretrain,
                          timelimit = timelimit,
                          outs=outs,
                          n_critic=self.n_critic
                          )
            appl(appl_ds,"gentarget2source%sbest.hdf5" % self.experiment_name(),self.variables[:-1],self,180,tune=self.tune,tune_name=self.tune_name)

            if self.resume:
                with open(outs[4].path,"w+") as of:
                    print("resumed",file = of)
            


EXP_PAPER = {
    "fullrepr": {},
    "fullreprid0": {"identity_lambda": 0},
    "fullreprseg0": {"segmentation_consistency_lmbda": 0},
    "fullreprid0seg0": {"identity_lambda": 0, "segmentation_consistency_lmbda": 0},
    "fullreprcl0sl0": {"segmentation_consistency_lmbda": 0, "cycled_lambda": 0},
}


class TrainAllCycleGAN(luigi.WrapperTask):
    suffix = luigi.Parameter(default="")
    seed = luigi.IntParameter(default=123)
    
    def requires(self):
        tasks = []
        
        for suffix, kwargs in EXP_PAPER.items():
            tasks.append(TrainCycleGAN(suffix=self.suffix+suffix, seed=self.seed, **kwargs))
            
        return tasks


if __name__ == "__main__":
    
    if np.any(["gpu" in x for x in sys.argv]):
        try:
            assert len(tf.config.list_physical_devices('GPU'))>0,tf.config.list_physical_devices()
        except AttributeError:
            assert tf.test.is_gpu_available()
        
    luigi.run(local_scheduler=True)
