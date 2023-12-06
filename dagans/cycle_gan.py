import tensorflow as tf
from dagans import gan_model
from tqdm import tqdm
import datetime
import logging
import io
import matplotlib.pyplot as plt
import time
import os
import tensorflow_addons as tfa
import numpy as np
from tensorflow.image import ssim as SSIM


def MSE(x,y, gbs=None):
    """multigpu MSE that averages over the correct axes"""
    mse= tf.keras.losses.MeanSquaredError(reduction= tf.keras.losses.Reduction.NONE)
    out = tf.reduce_mean(mse(x,y),axis=(1,2))
    return tf.nn.compute_average_loss(out,global_batch_size = gbs)


def kl_divergence(y_true, y_pred,gbs=None):
    """kullback-leibler-divergence with failsafe EPS"""
    EPS = 1e-6
    y_true = tf.clip_by_value(y_true, EPS, 1 - EPS)
    y_pred = tf.clip_by_value(y_pred, EPS, 1 - EPS)
    kl_div = y_true * tf.math.log(y_true/y_pred) + \
        (1 - y_true) * tf.math.log((1 - y_true)/(1 - y_pred))
    return tf.nn.compute_average_loss(tf.reduce_mean(kl_div,axis=(1,2,3)),
                                      global_batch_size=gbs)




class WsteinGAN:
    def __init__(self, gentargetsource, gensourcetarget, disctarget, discsource, 
                 batch_size, datafolder,
                 gradient_penalty_fun,lr=0.0001, beta1=.5,
                 eps=1e-2,steps_for_clr=400,clr=False,
                 toa_model=None,water_path_lambda=0.001,
                 verbose=True):
        """Wasserstein Cycle-GAN

        Args:
            gentargetsource (tf.keras.models.Model): generator 1
            gensourcetarget (tf.keras.models.Model): generator 2
            disctarget (tf.keras.models.Model): critic 1
            discsource (tf.keras.models.Model): critic 2
            batch_size (int): ...
            datafolder (string): path to folder where normalization constants are saved
            gradient_penalty_fun (callable): what to use as GP function, the input to this deals with handling of no GP
            lr (float, optional): starting learning rate. Defaults to 0.0001.
            beta1 (float, optional): momentum for Adam optimizer?. Defaults to .5.
            eps (float, optional): Lipschitz continuity clipping threshold. Defaults to 1e-2.
            steps_for_clr (int, optional): period of cyclic learning rate. Defaults to 400.
            clr (bool, optional): Use cyclic learning rate. Defaults to False.
            toa_model (callable, optional): function that returns the toa consistency loss from. Is able to handle input of the form (target_fake, source_fake, target_toa, source_toa) managing itself. Defaults to None.
            water_path_lambda (float, optional): water path consistency lamba. need it here already because the function needs to be a method of the GAN that can be de-lambda-fied. Defaults to 0.001.
            verbose (bool, optional): Progress bars and whatnot are turned off for tuning. Defaults to True.
        """        
        self.verbose = verbose
        self.gentargetsource = gentargetsource
        self.gensourcetarget = gensourcetarget
        self.disctarget = disctarget
        self.discsource = discsource
        self.toa_model = toa_model
        self.eps=eps
        #needed for early stopping
        self.rolling_metric = np.zeros(50)
        if clr:  
            if self.verbose:
                print("clr set up", steps_for_clr)    
            #configured this way because it worked
            lr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=lr,
                maximal_learning_rate=50*lr,
                scale_fn=lambda x: 1/tf.math.sqrt(tf.math.log1p(x)),
                step_size=steps_for_clr# * steps_per_epoch
            )

        self.gentargetsource_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta1)
        self.gensourcetarget_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta1)
        self.disctarget_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta1)
        self.discsource_optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta1)
        """
        #if you think RMSprop is more stable (results inconclusive)
        self.gentargetsource_optimizer = tf.keras.optimizers.RMSprop(lr)
        self.gensourcetarget_optimizer = tf.keras.optimizers.RMSprop(lr)
        self.disctarget_optimizer = tf.keras.optimizers.RMSprop(lr)
        self.discsource_optimizer = tf.keras.optimizers.RMSprop(lr)
        print("USING RMSPROP")
        """
        
        self.wstein_loss = lambda x,y: gan_model.wstein_dist(x,y,batch_size)
        self.gradient_penalty_fun = gradient_penalty_fun
        self.identity_loss = lambda x,y: gan_model.weighted_loss(x,y,batch_size)
        
        self.cycle_loss = lambda x,y: gan_model.mae_loss(x,y, batch_size)
        self.loss_cc = lambda x,y : gan_model.cloud_type_loss(x,y,batch_size)
        
        self.toa_loss = lambda x,y : tf.nn.compute_average_loss(tf.sqrt(tf.reduce_mean((
                                                        x-y)**2, axis=(1,2,3))),
                                                        global_batch_size = batch_size)
        cwploss = gan_model.cwp_loss(datafolder)
        self.water_path_loss = lambda x,y: tf.nn.compute_average_loss(cwploss(x,y),
                                                                global_batch_size = batch_size) *water_path_lambda
        self.wp_lam = water_path_lambda                                                                
        self.lr = lr
        
    def early_stopping(self,new_metric):
        if np.sum(self.rolling_metric>0):
            prev = np.mean(self.rolling_metric[self.rolling_metric!=0])
        else:
            prev=1e6
        self.rolling_metric[-1]=new_metric
        self.rolling_metric = np.roll(self.rolling_metric,shift=-1)
        if new_metric <= (prev-prev*0.005) or np.any(self.rolling_metric==0):
            return False#dont stop
        else:
            return True#stop

    def train_step_fun_pretrain(self, gan_lmbda=1, identity_lmbda=5, cycled_lmbda=5,
                       segmentation_consistency_lmbda=1, gradient_penalty_weight=0.,
                       update_gen=True):
        """this function returns the stepping function used during pretraining, where mostly the gan losses apply.
            has to be a separate function because i can not pass flags to the function in multiGPU training
            or im just stupid and this is unneccesary
        """
        object_cyclegan = self
        def train_step_pt(inputs):
            #makes a  forward pass, computes and applies the gradients
            #only updates 
            target_image,source_image,_, _ = inputs
            tf.debugging.assert_all_finite(target_image,"targ nan")
            tf.debugging.assert_all_finite(source_image,"source nan")
            with tf.GradientTape(persistent=True) as tape:

                source_fake_image = object_cyclegan.gentargetsource(
                    target_image, training=True)
                target_fake_image = object_cyclegan.gensourcetarget(
                    source_image, training=True)
                
                cycled_source_image = object_cyclegan.gentargetsource(
                    target_fake_image, training=True)
                cycled_target_image = object_cyclegan.gensourcetarget(
                    source_fake_image, training=True)
                concat_sourceimages = tf.concat([source_image, source_fake_image], axis=0)
                disc_concat_output_source = object_cyclegan.discsource(concat_sourceimages,
                                                               training=True)
                
                disc_sourcereal = disc_concat_output_source[:source_image.shape[0]]
                disc_sourcefake = disc_concat_output_source[source_image.shape[0]:]

                concat_targetimages = tf.concat([target_image, target_fake_image], axis=0)
                disc_concat_output_target = object_cyclegan.disctarget(concat_targetimages,
                                                               training=True)

                disc_targetreal = disc_concat_output_target[:target_image.shape[0]]
                disc_targetfake = disc_concat_output_target[target_image.shape[0]:]

                gen_targetsource_loss = object_cyclegan.wstein_loss(disc_sourcefake,tf.zeros_like(disc_sourcefake))

                disc_source_loss = object_cyclegan.wstein_loss(disc_sourcereal, disc_sourcefake)
                gp_source_loss = object_cyclegan.gradient_penalty_fun(
                    object_cyclegan.discsource, source_image, target_fake_image)
                disc_source_total_loss = disc_source_loss + gradient_penalty_weight * gp_source_loss

                gen_sourcetarget_loss = object_cyclegan.wstein_loss(disc_targetfake, tf.zeros_like(disc_targetfake))
                disc_target_loss = object_cyclegan.wstein_loss(
                    disc_targetreal, disc_targetfake)
                gp_target_loss = object_cyclegan.gradient_penalty_fun(
                    object_cyclegan.disctarget, target_image, source_fake_image)
                disc_target_total_loss = disc_target_loss + gradient_penalty_weight * gp_target_loss
                #tf.debugging.assert_all_finite(disc_target_loss ,"disctarget_loss_alone nan")
                #tf.debugging.assert_all_finite(gp_target_loss ,"gp_target_loss nan")
                identity_loss_targetsource = object_cyclegan.identity_loss( target_image,
                        source_fake_image)
    
                identity_loss_sourcetarget = object_cyclegan.identity_loss( source_image, 
                                target_fake_image)
                cycle_loss_source = object_cyclegan.cycle_loss(source_image, cycled_source_image)
                cycle_loss_target = object_cyclegan.cycle_loss(target_image, cycled_target_image)
            
                cons_loss_s2t =0
                cons_loss_t2s =0
                total_consistency_gen_loss =0
                total_cycle_loss = cycle_loss_source + cycle_loss_target
                #apply the auxiliary losses with a fraction of the strength
                gen_targetsource_total_loss = gan_lmbda*gen_targetsource_loss + identity_lmbda/100 * identity_loss_targetsource + \
                    cycled_lmbda/100 * total_cycle_loss +\
                    segmentation_consistency_lmbda*0 * cons_loss_t2s +\
                        self.water_path_loss(target_image, source_fake_image)
                gen_sourcetarget_total_loss = gan_lmbda*gen_sourcetarget_loss + identity_lmbda/100 * identity_loss_sourcetarget + \
                    cycled_lmbda/100 * total_cycle_loss +\
                    segmentation_consistency_lmbda*0 * cons_loss_s2t+\
                        self.water_path_loss(source_image, target_fake_image)

            disctarget_gradients = tape.gradient(disc_target_total_loss,
                                             object_cyclegan.disctarget.trainable_variables)
            discsource_gradients = tape.gradient(disc_source_total_loss,
                                             object_cyclegan.discsource.trainable_variables)
            """
            tf.debugging.assert_all_finite(gen_targetsource_total_loss ,"gentargetsource_loss nan")
            tf.debugging.assert_all_finite(gen_sourcetarget_total_loss ,"gensourcetarget_loss nan")
            tf.debugging.assert_all_finite(disc_target_total_loss ,"disctarget_loss nan")
            tf.debugging.assert_all_finite(disc_source_total_loss ,"discsource_loss nan")
            for totest in gentargetsource_gradients:
                tf.debugging.assert_all_finite( totest,"gentargetsource_gradients nan")
            for totest in gensourcetarget_gradients:
                tf.debugging.assert_all_finite( totest,"gensourcetarget_gradients nan")
            for totest in disctarget_gradients:
                tf.debugging.assert_all_finite( totest,"disctarget_gradients nan")
            for totest in discsource_gradients:
                tf.debugging.assert_all_finite(  totest," discsource_gradients nan")
            """
            if update_gen:
                    
                gentargetsource_gradients = tape.gradient(gen_targetsource_total_loss,
                                                object_cyclegan.gentargetsource.trainable_variables)
                gensourcetarget_gradients = tape.gradient(gen_sourcetarget_total_loss,
                                              object_cyclegan.gensourcetarget.trainable_variables)
            
                object_cyclegan.gentargetsource_optimizer.apply_gradients(zip(gentargetsource_gradients,
                                                object_cyclegan.gentargetsource.trainable_variables))
                object_cyclegan.gensourcetarget_optimizer.apply_gradients(zip(gensourcetarget_gradients,
                                                object_cyclegan.gensourcetarget.trainable_variables))
            
            object_cyclegan.disctarget_optimizer.apply_gradients(zip(disctarget_gradients,
                                             object_cyclegan.disctarget.trainable_variables))
            object_cyclegan.discsource_optimizer.apply_gradients(zip(discsource_gradients,
                                             object_cyclegan.discsource.trainable_variables))
            """
            for totest in object_cyclegan.gentargetsource.trainable_variables:
                tf.debugging.assert_all_finite( totest,"gentargetsource nan")
            for totest in object_cyclegan.gensourcetarget.trainable_variables:
                tf.debugging.assert_all_finite( totest,"gensourcetarget nan")
            for totest in object_cyclegan.disctarget.trainable_variables:
                tf.debugging.assert_all_finite( totest,"disctarget nan")
            for totest in object_cyclegan.discsource.trainable_variables:
                tf.debugging.assert_all_finite(  totest," discsource nan")
            """
            return tf.stack((gen_targetsource_loss, gen_sourcetarget_loss, identity_loss_targetsource, identity_loss_sourcetarget, \
                cycle_loss_target, cycle_loss_source, disc_target_loss, disc_source_loss, \
                gp_target_loss, gp_source_loss, \
                total_consistency_gen_loss, \
                gen_targetsource_total_loss, gen_sourcetarget_total_loss),axis=0)

        return train_step_pt

       

    def train_step_fun(self, gan_lmbda=1, identity_lmbda=5, cycled_lmbda=5,
                       segmentation_consistency_lmbda=1, gradient_penalty_weight=0.,
                       update_gen=True):
        """same as train_step_fun_pt but without modification of loss terms"""

        object_cyclegan = self
        def train_step(inputs):
            target_image,source_image,targ_toa, sour_toa = input
            tf.debugging.assert_all_finite(target_image,"targ nan")
            tf.debugging.assert_all_finite(source_image,"source nan")
            tf.debugging.assert_all_finite(sour_toa,"sour toa nan")
            tf.debugging.assert_all_finite(targ_toa,"targ toa nan")
            with tf.GradientTape(persistent=True) as tape:
                source_fake_image = object_cyclegan.gentargetsource(
                    target_image, training=True)
                target_fake_image = object_cyclegan.gensourcetarget(
                    source_image, training=True)
                #tf.debugging.assert_all_finite(target_fake_image ,"targetfake nan")
                #tf.debugging.assert_all_finite(source_fake_image ,"sourcefake nan")
                cycled_source_image = object_cyclegan.gentargetsource(
                    target_fake_image, training=True)
                cycled_target_image = object_cyclegan.gensourcetarget(
                    source_fake_image, training=True)
                
                concat_sourceimages = tf.concat([source_image, source_fake_image], axis=0)
                disc_concat_output_source = object_cyclegan.discsource(concat_sourceimages,
                                                               training=True)
                #tf.debugging.assert_all_finite(disc_concat_output_source ,"disc_concat_output_source nan")
                
                disc_sourcereal = disc_concat_output_source[:source_image.shape[0]]
                disc_sourcefake = disc_concat_output_source[source_image.shape[0]:]
                concat_targetimages = tf.concat([target_image, target_fake_image], axis=0)
                disc_concat_output_target = object_cyclegan.disctarget(concat_targetimages,
                                                               training=True)
                #tf.debugging.assert_all_finite(disc_concat_output_target ,"disc_concat_output_target nan")

                disc_targetreal = disc_concat_output_target[:target_image.shape[0]]
                disc_targetfake = disc_concat_output_target[target_image.shape[0]:]

                gen_targetsource_loss = object_cyclegan.wstein_loss(disc_sourcefake,tf.zeros_like(disc_sourcefake))

                disc_source_loss = object_cyclegan.wstein_loss(disc_sourcereal, disc_sourcefake)
                gp_source_loss = object_cyclegan.gradient_penalty_fun(
                    object_cyclegan.discsource, source_image, target_fake_image)
                disc_source_total_loss = disc_source_loss + gradient_penalty_weight * gp_source_loss

                gen_sourcetarget_loss = object_cyclegan.wstein_loss(disc_targetfake, tf.zeros_like(disc_targetfake))
                disc_target_loss = object_cyclegan.wstein_loss(
                    disc_targetreal, disc_targetfake)
                gp_target_loss = object_cyclegan.gradient_penalty_fun(
                    object_cyclegan.disctarget, target_image, source_fake_image)
                disc_target_total_loss = disc_target_loss + gradient_penalty_weight * gp_target_loss
                #tf.debugging.assert_all_finite(disc_target_loss ,"disctarget_loss_alone nan")
                #tf.debugging.assert_all_finite(gp_target_loss ,"gp_target_loss nan")
                identity_loss_targetsource = object_cyclegan.identity_loss(
                        target_image, source_fake_image)
    
                identity_loss_sourcetarget = object_cyclegan.identity_loss(
                    source_image, target_fake_image)
                cycle_loss_source = object_cyclegan.cycle_loss(source_image, cycled_source_image)
                cycle_loss_target = object_cyclegan.cycle_loss(target_image, cycled_target_image)
                
                if self.toa_model is not None:
                    targ_toa_loss, sour_toa_loss= self.toa_model(target_fake_image, source_fake_image,
                                                                targ_toa, sour_toa, self.toa_loss)
                     
                    tf.debugging.assert_all_finite(targ_toa_loss ,"targ_toa_loss nan")
                    tf.debugging.assert_all_finite(sour_toa_loss ,"sour_toa_loss nan")
                    total_consistency_gen_loss = sour_toa_loss + targ_toa_loss
                    cons_loss_s2t = sour_toa_loss
                    cons_loss_t2s = targ_toa_loss
                else:
                    cons_loss_t2s =0
                    cons_loss_s2t =0
                    total_consistency_gen_loss =0

                total_cycle_loss = cycle_loss_source + cycle_loss_target

                gen_targetsource_total_loss = gan_lmbda*gen_targetsource_loss + identity_lmbda * identity_loss_targetsource + \
                    cycled_lmbda * total_cycle_loss +\
                    segmentation_consistency_lmbda * cons_loss_t2s +\
                        self.water_path_loss(target_image, source_fake_image)
                gen_sourcetarget_total_loss = gan_lmbda*gen_sourcetarget_loss + identity_lmbda * identity_loss_sourcetarget + \
                    cycled_lmbda * total_cycle_loss +\
                    segmentation_consistency_lmbda * cons_loss_s2t +\
                        self.water_path_loss(source_image, target_fake_image)

            disctarget_gradients = tape.gradient(disc_target_total_loss,
                                             object_cyclegan.disctarget.trainable_variables)
            discsource_gradients = tape.gradient(disc_source_total_loss,
                                             object_cyclegan.discsource.trainable_variables)
            """
            tf.debugging.assert_all_finite(gen_targetsource_total_loss ,"gentargetsource_loss nan")
            tf.debugging.assert_all_finite(gen_sourcetarget_total_loss ,"gensourcetarget_loss nan")
            tf.debugging.assert_all_finite(disc_target_total_loss ,"disctarget_loss nan")
            tf.debugging.assert_all_finite(disc_source_total_loss ,"discsource_loss nan")
            for totest in gentargetsource_gradients:
                tf.debugging.assert_all_finite( totest,"gentargetsource_gradients nan")
            for totest in gensourcetarget_gradients:
                tf.debugging.assert_all_finite( totest,"gensourcetarget_gradients nan")
            for totest in disctarget_gradients:
                tf.debugging.assert_all_finite( totest,"disctarget_gradients nan")
            for totest in discsource_gradients:
                tf.debugging.assert_all_finite(  totest," discsource_gradients nan")
            """
            if update_gen:
                    
                gentargetsource_gradients = tape.gradient(gen_targetsource_total_loss,
                                                object_cyclegan.gentargetsource.trainable_variables)
                gensourcetarget_gradients = tape.gradient(gen_sourcetarget_total_loss,
                                              object_cyclegan.gensourcetarget.trainable_variables)
            
                object_cyclegan.gentargetsource_optimizer.apply_gradients(zip(gentargetsource_gradients,
                                                object_cyclegan.gentargetsource.trainable_variables))
                object_cyclegan.gensourcetarget_optimizer.apply_gradients(zip(gensourcetarget_gradients,
                                                object_cyclegan.gensourcetarget.trainable_variables))
            
            object_cyclegan.disctarget_optimizer.apply_gradients(zip(disctarget_gradients,
                                             object_cyclegan.disctarget.trainable_variables))
            object_cyclegan.discsource_optimizer.apply_gradients(zip(discsource_gradients,
                                             object_cyclegan.discsource.trainable_variables))
            """
            for totest in object_cyclegan.gentargetsource.trainable_variables:
                tf.debugging.assert_all_finite( totest,"gentargetsource nan")
            for totest in object_cyclegan.gensourcetarget.trainable_variables:
                tf.debugging.assert_all_finite( totest,"gensourcetarget nan")
            for totest in object_cyclegan.disctarget.trainable_variables:
                tf.debugging.assert_all_finite( totest,"disctarget nan")
            for totest in object_cyclegan.discsource.trainable_variables:
                tf.debugging.assert_all_finite(  totest," discsource nan")
            """
            
            return tf.stack((gen_targetsource_loss, gen_sourcetarget_loss, identity_loss_targetsource, identity_loss_sourcetarget, \
                cycle_loss_target, cycle_loss_source, disc_target_loss, disc_source_loss, \
                gp_target_loss, gp_source_loss, \
                total_consistency_gen_loss, \
                gen_targetsource_total_loss, gen_sourcetarget_total_loss),axis=0)

        return train_step



def fit_wstein(train_ds, epochs, obj_cyclegan,strategy, steps_per_epoch=None, frec_update=10, val_ds=None,
        logdir=None, gan_lmbda=1,identity_lmbda=5, cycled_lmbda=5, 
        segmentation_consistency_lmbda=1, gradient_penalty_weight=0,
        dummy=0,pretrain=0,timelimit=3600,outs=None, n_critic=5):
    """the training function: forward passing, gradient updates, validation, early stopping

    Args:
        train_ds (tf.data.Dataset): with all the preprocessing, ready to be iterated over
        epochs (int): number interations over full dataset
        obj_cyclegan (WsteinGAN): class from above
        strategy (tf strategy): MultiWorkerGPU or something. 
        steps_per_epoch (int, optional): I thinks thats deprecated or uses len(train_ds)/batch_size. Defaults to None.
        frec_update (int, optional): progressbar update frequency. Defaults to 10.
        val_ds (tf.data.Dataset, optional): equivalent to train_ds. Defaults to None.
        logdir (string, optional): where to write to tensorbaord. Defaults to None.
        gan_lmbda (int, optional): weight for generator loss. would argue its completely inconsequential in this configuration. Defaults to 1.
        identity_lmbda (int, optional): gradient weight. Defaults to 5.
        cycled_lmbda (int, optional): gradient weight. Defaults to 5.
        segmentation_consistency_lmbda (int, optional): gradient weight. Defaults to 1.
        gradient_penalty_weight (int, optional): gradient weight. Defaults to 0.
        dummy (int, optional): deprecated. Defaults to 0.
        pretrain (int, optional): number of pretraining epochs. Defaults to 0.
        timelimit (int, optional): breaks if more seconds than this elapse. Defaults to 3600.
        outs (list, optional): paths to where to save the 4 models. Defaults to None.
        n_critic (int, optional): number of updates of the critic to do per generator update. Defaults to 5.

    Raises:
        ValueError: we need to provide the paths where to save the models or luigi complains

    """    
    if outs is None:
        raise ValueError("Need outpaths")
    bad_count =0 # counts how often the model performance thecreases
    train_step_crit = obj_cyclegan.train_step_fun( gan_lmbda = gan_lmbda,identity_lmbda=identity_lmbda, cycled_lmbda=cycled_lmbda,
                                             segmentation_consistency_lmbda=segmentation_consistency_lmbda,
                                             gradient_penalty_weight=gradient_penalty_weight,
                                             update_gen=False)
    train_step_pretrain_crit = obj_cyclegan.train_step_fun_pretrain( gan_lmbda = gan_lmbda,identity_lmbda=identity_lmbda, cycled_lmbda=cycled_lmbda,
                                             segmentation_consistency_lmbda=segmentation_consistency_lmbda,
                                             gradient_penalty_weight=gradient_penalty_weight,
                                             update_gen=False)
    train_step_gen = obj_cyclegan.train_step_fun( gan_lmbda = gan_lmbda,identity_lmbda=identity_lmbda, cycled_lmbda=cycled_lmbda,
                                             segmentation_consistency_lmbda=segmentation_consistency_lmbda,
                                             gradient_penalty_weight=gradient_penalty_weight)
    train_step_pretrain_gen = obj_cyclegan.train_step_fun_pretrain( gan_lmbda = gan_lmbda,identity_lmbda=identity_lmbda, cycled_lmbda=cycled_lmbda,
                                             segmentation_consistency_lmbda=segmentation_consistency_lmbda,
                                            gradient_penalty_weight=gradient_penalty_weight)
    #there is probably a good reason for this but i cant remember
    #these functions are defined here and selected for each batch as applicable during the training to use as the forward pass
    @tf.function
    def distributed_train_step_crit(dist_inputs):         
        per_replica_losses = strategy.run(train_step_crit,args=(dist_inputs,
                                                     ))
        return strategy.reduce(tf.distribute.ReduceOp.SUM,per_replica_losses,
                               axis=None)

    @tf.function
    def distributed_train_step_pretrain_crit(dist_inputs):         
        per_replica_losses = strategy.run(train_step_pretrain_crit,args=(dist_inputs,
                                                     ))
        return strategy.reduce(tf.distribute.ReduceOp.SUM,per_replica_losses,
                               axis=None)

    @tf.function
    def distributed_train_step_gen(dist_inputs):         
        per_replica_losses = strategy.run(train_step_gen,args=(dist_inputs,
                                                     ))
        return strategy.reduce(tf.distribute.ReduceOp.SUM,per_replica_losses,
                               axis=None)

    @tf.function
    def distributed_train_step_pretrain_gen(dist_inputs):         
        per_replica_losses = strategy.run(train_step_pretrain_gen,args=(dist_inputs,
                                                     ))
        return strategy.reduce(tf.distribute.ReduceOp.SUM,per_replica_losses,
                               axis=None)

    if obj_cyclegan.verbose:
        pbar = tqdm(total=epochs,leave=True,position=0)
    else:
        pbar =None
    steps_per_epoch_str = "undef" if steps_per_epoch is None else str(
        steps_per_epoch)
    
    if val_ds is None:
        val_ds = train_ds.shuffle(len(train_ds))
    if logdir is None:
        logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    summary_writer = tf.summary.create_file_writer(logdir)
    logging.info("Run:\n tensorboard --logdir %s" % logdir)

    total = tf.cast(0, "int64")
    
    metric= kl_divergence
    
    best_metric = 1e6

    start = time.time()
    best_ssim = np.zeros(4)
    for  epoch in range(epochs):
        pretraining = epoch<pretrain # check if we are still pretraining
        assert len(val_ds)>0
        for count,(target_image, source_image, targ_toa, sour_toa, *_) in enumerate(val_ds):
            broken=False # failsafe reset
            if dummy:
                continue
            
            if (count<(epoch%len(val_ds))) and epoch!=0:
                #validation is only performed on one sample, which rotates
                continue
            
            source_fake = obj_cyclegan.gentargetsource(target_image)
            target_fake = obj_cyclegan.gensourcetarget(source_image)
            source_cycled = obj_cyclegan.gentargetsource(target_fake)
            target_cycled = obj_cyclegan.gensourcetarget(source_fake)

            
            cycle_loss_source = obj_cyclegan.cycle_loss(source_image, source_cycled)
            cycle_loss_target = obj_cyclegan.cycle_loss(target_image, target_cycled)
            #this metric is only relevant if we dont use toa model which si used as a metric then
            metric_value_source = metric(source_image,source_fake)
            metric_value_target = metric(target_image,target_fake)
            metric_value=(metric_value_source + metric_value_target)/2

            #structural similiarity index measure
            ssim_sourcetarget = tf.reduce_mean(SSIM(source_image, target_fake,max_val=1))
            ssim_targetsource = tf.reduce_mean(SSIM(target_image, source_fake,max_val=1))
            ssim_sourcecycled = tf.reduce_mean(SSIM(source_image, source_cycled,max_val=1))
            ssim_targetcycled = tf.reduce_mean(SSIM(target_image, target_cycled,max_val=1))
            current_ssim = np.array([ssim_sourcetarget,
                                    ssim_targetsource,
                                    ssim_sourcecycled,
                                    ssim_targetcycled])
            best_ssim = np.where(current_ssim>best_ssim,current_ssim, best_ssim)
            #de-lambda-fying the cwp loss for tensorboard
            cwp_loss_t2s = obj_cyclegan.water_path_loss(target_image,
                             source_fake)/obj_cyclegan.wp_lam
            cwp_loss_s2t = obj_cyclegan.water_path_loss(source_image, 
                            target_fake)/obj_cyclegan.wp_lam
            

            if obj_cyclegan.toa_model is not None:
                metric_value_source,metric_value_target = obj_cyclegan.toa_model(target_fake,source_fake,targ_toa, sour_toa, obj_cyclegan.toa_loss,
                                        sw=summary_writer,step=total)
                metric_value = (metric_value_source+metric_value_target)/2
            #dont save the randomly initialized model but save at the very beginning
            if epoch==1 and ( not os.path.exists(outs[0].path) ) :
                best_metric = metric_value
                obj_cyclegan.gentargetsource.save(outs[0].path)
                obj_cyclegan.discsource.save(outs[1].path)
                obj_cyclegan.gensourcetarget.save(outs[2].path)
                obj_cyclegan.disctarget.save(outs[3].path)
                obj_cyclegan.gentargetsource.save(outs[0].path.replace(".hdf5","best.hdf5"))
                obj_cyclegan.discsource.save(outs[1].path.replace(".hdf5","best.hdf5"))
                obj_cyclegan.gensourcetarget.save(outs[2].path.replace(".hdf5","best.hdf5"))
                obj_cyclegan.disctarget.save(outs[3].path.replace(".hdf5","best.hdf5"))

            elif metric_value< best_metric:
                if obj_cyclegan.verbose:
                    print("step {}: saving with mean metric={}".format(total,metric_value)) 
                obj_cyclegan.gentargetsource.save(outs[0].path.replace(".hdf5","best.hdf5"))
                obj_cyclegan.discsource.save(outs[1].path.replace(".hdf5","best.hdf5"))
                obj_cyclegan.gensourcetarget.save(outs[2].path.replace(".hdf5","best.hdf5"))
                obj_cyclegan.disctarget.save(outs[3].path.replace(".hdf5","best.hdf5"))
                best_metric = metric_value
            
            with summary_writer.as_default():
                figure = image_grid(target_image)
                tf.summary.image(
                        "Targ Image",plot_to_image(figure), step=total)
                figure = image_grid(source_fake)
                tf.summary.image(
                        "Sour Fake", plot_to_image(figure), step=total)
                figure = image_grid(source_image)
                tf.summary.image(
                        "Sour Image", plot_to_image(figure), step=total)
                figure = image_grid(target_fake)
                tf.summary.image(
                        "Targ Fake", plot_to_image(figure), step=total)
                figure = image_grid(target_cycled)
                tf.summary.image(
                        "Targ Cycled", plot_to_image(figure), step=total)
                figure = image_grid(source_cycled)
                tf.summary.image(
                        "Sour Cycled", plot_to_image(figure), step=total)
                
                tf.summary.scalar('Metric (Sour,Sour_fake)', metric_value_source, step=total)
                tf.summary.scalar('Metric (Targ,Targ_fake)', metric_value_target, step=total)
                tf.summary.scalar('cyc (Sour,Sour_cyc)', cycle_loss_source, step=total)
                tf.summary.scalar('cyc (Targ,Targ_cyc)', cycle_loss_target, step=total)
                tf.summary.scalar('SSIM Targetsource', ssim_targetsource, step=total)
                tf.summary.scalar('SSIM Sourcetarget', ssim_sourcetarget, step=total)
                tf.summary.scalar('SSIM TCyc', ssim_targetcycled, step=total)
                tf.summary.scalar('SSIM SCyc', ssim_sourcecycled, step=total)
                tf.summary.scalar('CWP S2T', cwp_loss_s2t, step=total)
                tf.summary.scalar('CWP T2S', cwp_loss_t2s, step=total)
            
                if total>2000:
                    for channel in range(target_image.shape[-1]):
                        tf.summary.histogram("hist target cycled"+str(channel),target_cycled[...,channel],step=total)
                        tf.summary.histogram("hist source cycled"+str(channel),source_cycled[...,channel],step=total)
                        tf.summary.histogram("hist target image"+str(channel),target_image[...,channel],step=total)
                        tf.summary.histogram("hist source image"+str(channel),source_image[...,channel],step=total)
                        tf.summary.histogram("hist target fake"+str(channel),target_fake[...,channel],step=total)
                        tf.summary.histogram("hist source fake"+str(channel),source_fake[...,channel],step=total)

            break

        if metric_value>best_metric:
            bad_count+=1
            if np.all(current_ssim<(best_ssim-0.4*best_ssim)):
                print("breaking for drastically reduced performance")
                break
            elif bad_count>100:
                print("breaking due to repeated divergence")
                break

        if obj_cyclegan.early_stopping(metric_value):
            print("breaking for early stopping convergence with last metrics {}".format(obj_cyclegan.rolling_metric))
            break

        # Train
        for n, (target_image, source_image, targ_toa,sour_toa) in enumerate(train_ds):
            if dummy:
                continue
            if n%n_critic==0:
                if pretraining:
                    dist_fct = distributed_train_step_pretrain_gen
                else:
                    dist_fct = distributed_train_step_gen
            else:#these update only the critic
                if pretraining:
                    dist_fct = distributed_train_step_pretrain_crit
                else:
                    dist_fct = distributed_train_step_crit
            
            gen_targetsource_loss, gen_sourcetarget_loss, identity_loss_targetsource, identity_loss_sourcetarget, \
                cycle_loss_target, cycle_loss_source, disc_target_loss, disc_source_loss, \
                gp_target_loss, gp_source_loss, \
                total_consistency_gen_loss, \
                gen_targetsource_total_loss, gen_sourcetarget_total_loss = dist_fct(

                    (target_image, source_image,targ_toa, sour_toa))
            

            if pretraining:
                tf.debugging.assert_equal(total_consistency_gen_loss,0.,message="impossibru", summarize=1)
            with summary_writer.as_default():
                tf.summary.scalar('gen_targetsource_loss', gen_targetsource_loss, step=total)
                tf.summary.scalar('gen_sourcetarget_loss', gen_sourcetarget_loss, step=total)
                tf.summary.scalar('identity_loss_targetsource',
                                  identity_loss_targetsource, step=total)
                tf.summary.scalar('identity_loss_sourcetarget',
                                  identity_loss_sourcetarget, step=total)
                tf.summary.scalar('cycle_loss_target', cycle_loss_target, step=total)
                tf.summary.scalar('cycle_loss_source', cycle_loss_source, step=total)
                tf.summary.scalar('total_consistency_gen_loss',
                                  total_consistency_gen_loss, step=total)
                tf.summary.scalar('gen_targetsource_total_loss',
                                  gen_targetsource_total_loss, step=total)
                tf.summary.scalar('gen_sourcetarget_total_loss',
                                  gen_sourcetarget_total_loss, step=total)
                tf.summary.scalar('disc_target_loss', disc_target_loss, step=total)
                tf.summary.scalar('disc_source_loss', disc_source_loss, step=total)
                tf.summary.scalar('gp_target_loss', gp_target_loss, step=total)
                tf.summary.scalar('gp_source_loss', gp_source_loss, step=total)
                #apparetnly this doesnt work in tf 2.11.0
                #tf.summary.scalar('LR',tf.reduce_mean(obj_cyclegan.gentargetsource_optimizer._learning_rate()),step=total)
                """
                tf.summary.scalar('disc_target_grad', disctarget_grad, step=total)
                tf.summary.scalar('disc_source_grad', discsource_grad, step=total)
                tf.summary.scalar('gen_targetsource_grad', gentargetsource_grad, step=total)
                tf.summary.scalar('gen_sourcetarget_grad', gensourcetarget_grad, step=total)
                """
            if not (tf.reduce_all(tf.math.is_finite((gen_targetsource_loss, gen_sourcetarget_loss,
                    identity_loss_targetsource, identity_loss_sourcetarget, 
                    cycle_loss_target, cycle_loss_source, disc_target_loss, disc_source_loss, 
                    gp_target_loss, gp_source_loss, 
                    total_consistency_gen_loss, 
                    gen_targetsource_total_loss, gen_sourcetarget_total_loss
                    )))):
                print("breaking for nan during epoch",tf.reduce_mean((gen_targetsource_loss, gen_sourcetarget_loss,
                    identity_loss_targetsource, identity_loss_sourcetarget, 
                    cycle_loss_target, cycle_loss_source, disc_target_loss, disc_source_loss, 
                    gp_target_loss, gp_source_loss, 
                    total_consistency_gen_loss, 
                    gen_targetsource_total_loss, gen_sourcetarget_total_loss)),tf.math.is_finite((gen_targetsource_loss, gen_sourcetarget_loss,
                    identity_loss_targetsource, identity_loss_sourcetarget, 
                    cycle_loss_target, cycle_loss_source, disc_target_loss, disc_source_loss, 
                    gp_target_loss, gp_source_loss, 
                    total_consistency_gen_loss, 
                    gen_targetsource_total_loss, gen_sourcetarget_total_loss)))
                broken=True
                break
            total += 1
            if (n % frec_update) == 0 and obj_cyclegan.verbose:
                pbar.set_description('Epoch %d step %d/%s: last batch Targ->Sour loss = %.4f\t Sour->Targ loss: %.4f Disc Targ loss: %.4f Disc Sour loss: %.4f' %
                                     ( epoch, int(n), steps_per_epoch_str, float(gen_targetsource_total_loss), float(gen_sourcetarget_total_loss),
                                      float(disc_target_loss), float(disc_source_loss)))
        if broken:
            break
        time_taken = time.time()-start
        if time_taken > timelimit:
            print("breaking due to time")
            obj_cyclegan.gentargetsource.save(outs[0].path)
            obj_cyclegan.discsource.save(outs[1].path)
            obj_cyclegan.gensourcetarget.save(outs[2].path)
            obj_cyclegan.disctarget.save(outs[3].path)
            break
        if obj_cyclegan.verbose:
            pbar.update(1)
        
        
    logging.info("Train finished. Run:\n tensorboard --logdir %s" % logdir)
    obj_cyclegan.gentargetsource.save(outs[0].path)
    obj_cyclegan.discsource.save(outs[1].path)
    obj_cyclegan.gensourcetarget.save(outs[2].path)
    obj_cyclegan.disctarget.save(outs[3].path)



def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image



def image_grid(_images):
    """ Create a figure to contain the plot."""
    figure,ax = plt.subplots(1,4,figsize=(10,3),
                    gridspec_kw ={"width_ratios":[5,5,5,1]})
    for j,i in enumerate([0,4,6]):
        i = i%_images.shape[-1]
        # Start next subplot.
        ax[j].set_xticks([])
        ax[j].set_yticks([])
        ax[j].grid(False)
        ish=ax[j].imshow(_images[0,:,:,i], cmap="viridis",vmin=0,vmax=1)
    
    figure.colorbar(ish,cax=ax[-1])
    figure.tight_layout()
    return figure

