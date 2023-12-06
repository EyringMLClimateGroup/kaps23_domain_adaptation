from tensorflow import keras
import tensorflow as tf
from dagans import util_model
from keras.constraints import Constraint
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from tensorflow.image import ssim as SSIM
import h5py
tf.keras.backend.set_floatx('float32')



def d_layer(layer_input, filters, f_size=4, bn=True, name="d_layer",groups=1,const=None):
    """Discriminator layer"""
    d = keras.layers.Conv2D(filters,
                            kernel_size=f_size,
                            strides=2, padding='same',
                            name=name+"_conv2d",
                            groups=groups,
                            kernel_constraint=const)(layer_input)
    if bn:
        d = keras.layers.BatchNormalization(momentum=0.8,
                                            beta_constraint=const,
                                            gamma_constraint = const,
                                            name = name + "_bn")(d)
        
    d = keras.layers.LeakyReLU(alpha=0.2,name=name+"_leakyrelu")(d)

    return d


# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)

    
    def get_config(self):
        config = super(ClipConstraint,self).get_config()
        config.update({"clip_value": self.clip_value})
        return config


def disc_model(variables,shape=(64, 64, 4), df=8,
               activation=None,normalization=True, depth=3
               ,groups=1, const=None):
    """Discriminator (or with wgan, critic) model

    Args:
        variables (array or list?): not being used because normalization is deprecated
        shape (tuple, optional):  Defaults to (64, 64, 4).
        df (int, optional): filters. Defaults to 8.
        activation (str, optional): activation string known to tf. Defaults to None.
        normalization (bool, optional): if the weights should be initialized normalized. Deprecated. Defaults to True.
        depth (int, optional): number of layers. Defaults to 3.
        groups (int, optional): groupwise convolution. dont. Defaults to 1.
        const (_type_, optional): clipping threshold. Defaults to None.

    Returns:
        _type_: _description_
    """    
    ip = keras.layers.Input(shape,name="ip_disc")
    if const is not None:
        const= ClipConstraint(const)

    if normalization:
        raise NotImplementedError("have not been using that")
    else:
        x_init = ip
    
    d1 = d_layer(x_init, df, bn=False, name="d_layer_1",groups=groups,const=const)
    
    for cd in range(depth):
        d1 = d_layer(d1, df * 2**(cd+1),
                     name="d_layer_%d" % (cd+2), const=const)

    validity = keras.layers.Conv2D(1,
                                   kernel_size=1,
                                   strides=1,
                                   activation=activation,
                                   name="conv1x1",
                                   groups=groups,
                                   kernel_constraint=const)(d1) # (None, 2, 2, 1)
                                
    # validity = keras.layers.GlobalAveragePooling2D()(validity) # (None,  1)
    validity = keras.layers.Flatten()(validity)
    validity = keras.layers.Dense(1)(validity)

    return keras.models.Model(inputs=[ip],outputs=[validity],
                              name="discriminator")


def generator_simple(variables,shape=(64, 64, 4), df=64, l2reg=None, 
                     normalization=True,normtype="batchnorm",
                     groups=1, dropout=False, output_shape =None):
    """generator network

    Args:
        variables (whatever): not being used in here for backward compatibility
        shape (tuple, optional):  Defaults to (64, 64, 4).
        df (int, optional): filters. Defaults to 64.
        l2reg (?, optional): I thought bool but maybe a coeff or a function. check what is actually being passed. Defaults to None.
        normalization (bool, optional): Deprecated, use False. Defaults to True.
        normtype (str, optional): instancenorm or batchnorm. Defaults to "batchnorm".
        groups (int, optional): dont do it. Defaults to 1.
        dropout (bool, optional):  Defaults to False.
        output_shape (tuple, optional): number of output channels. Defaults to the number of input channels.

    Returns:
        tf.keras.models.Model: stack of tf operations
    """    
    ip = keras.layers.Input(shape, name="ip_gen")
    if output_shape is None:
        output_shape = shape
    if l2reg is None:
        reg = None
    else:
        reg = tf.keras.regularizers.l2(l2reg)

    # Input centering
    if normalization:
        raise NotImplementedError("yeah nah not doing that")
    else:
        x_init = ip
    x = util_model.conv_blocks_sep(x_init, df, -1, reg=reg,
                               name="generator_block_1", batch_norm=normtype != "no",
                               normtype=normtype, groups=groups)
    x2 = keras.layers.concatenate([x_init, x], axis=-1, name="gen_concatenate_1")
    if dropout:
        x2 = keras.layers.Dropout(.1)(x2)
    x3 = util_model.conv_blocks_sep(x2, df, -1, reg=reg,
                                name="generator_block_2", batch_norm=normtype != "no",
                                dilation_rate=(2, 2),
                               normtype=normtype,
                               groups=groups)
    x3 = keras.layers.concatenate([x3, x2], axis=-1, name="gen_concatenate_2")
    if dropout:
        x3 = keras.layers.Dropout(.1)(x3)
    out_conv = keras.layers.Conv2D(output_shape[-1]
                                   , (1, 1), name="out_generator",
                                   kernel_regularizer=None,
                                   groups=groups,
                                   activation = "softplus")
    
    out = out_conv(x3)
    # could do this but doint training its kinda pointless
    #out = tf.where(out>1,1.,out)
    if normalization:
        # Output centering
        raise NotImplementedError("yeah nah not implemented")

    return keras.models.Model(inputs=[ip],outputs=[out])


def discriminator_loss(disc_real_output, disc_generated_output,gbs=None):
    """just your standard crossentropy loss what can account for distributed training

    Args:
        disc_real_output (tf.Tensor): discriminators/critics opinion of the real domain
        disc_generated_output (tf.Tensor): discriminators/critics opinion of the fake domain
        gbs (int?, optional): Can be passed to change the denominator in distributed training scenarios. Defaults to None.

    Returns:
        tf float: the disc loss
    """    
    #does model think real data is real(1)?
    real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(disc_real_output), 
            logits=disc_real_output),axis=(1))
    #does model think fake data is fake (0)?
    generated_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(disc_generated_output),
                    logits=disc_generated_output),axis=(1) )

    total_disc_loss = tf.nn.compute_average_loss((real_loss + generated_loss),
                                                 global_batch_size=gbs)
    
    return total_disc_loss


def generator_gan_loss(disc_generated_output,gbs=None):
    """can generator fool discriminator?

    Args:
        disc_generated_output (tf tensor): discriminator opinion on fake data
        gbs (int?, optional): Can be passed to change the denominator in distributed training scenarios. Defaults to None.

    Returns:
        tf float: loss
    """    
    return tf.nn.compute_average_loss( tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(disc_generated_output),
        logits=disc_generated_output),axis=(1)),global_batch_size=gbs)

def cloud_mask_loss(true,false,gbs=None):
    """binary crossentropy that is probably not relevant anywhere anymore"""
    loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    return tf.nn.compute_average_loss( tf.reduce_mean(loss(
                     true, false),
                axis=(1)),
            global_batch_size=gbs)

def cloud_type_loss(true,false,gbs=None):
    """to use for a cloud type based task loss. not in use"""
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                    reduction=tf.keras.losses.Reduction.NONE)
    true = tf.expand_dims(tf.argmax(tf.nn.softmax(true),1),1)
    return tf.nn.compute_average_loss( loss( true, false),                    
                  global_batch_size=int(gbs))

def rel_loss(gen_input, gen_output, gbs=None):
    """it might not be good to use straight up MAE/MSE for
    cycle and identity losses, because we want the data to change, just not the structure

    Args:
        gen_input (tf tensor): whatever is fed into the generator
        gen_output (tf tensor): what then comes out of the generator
        gbs (int?, optional): Can be passed to change the denominator in distributed training scenarios. Defaults to None.

    Returns:
        tf float: loss
    """    
    #the MAE between the two tensors, both distributed in (0,1]
    gen_in = gen_input/(tf.reduce_max(gen_input,axis=(1,2),keepdims=True)+1e-8)
    gen_out = gen_output/(tf.reduce_max(gen_output,axis=(1,2),keepdims=True)+1e-8)
    return tf.nn.compute_average_loss(tf.reduce_mean(tf.abs(gen_in - gen_out),
                                      axis=(1,2,3)),global_batch_size=gbs)


def mae_loss(gen_input, gen_output, gbs=None):
    """what it says on the tin"""
    return tf.nn.compute_average_loss(tf.reduce_mean(tf.abs(gen_input - gen_output),
                                      axis=(1,2,3)),global_batch_size=gbs)

def weighted_loss(gen_input, gen_output, gbs=None):
    """weighting some variables more than others, espacially ptop and clt have shown issues so this is
    trying to counteract that"""
    weights = tf.reshape(tf.constant([ 1., 1., 1., 1.1, 2., 1.1, 2., 1.,5.][:gen_input.shape[-1]]),
                         (1,1,1,gen_input.shape[-1]))
    wdiff = (tf.abs(gen_input - gen_output)*weights)
    return tf.nn.compute_average_loss(tf.reduce_mean(wdiff,
                                      axis=(1,2,3)),global_batch_size=gbs)


def wstein_dist(real_crc, fake_crc,gbs=None):
    """this is the big one
    computes the average of difference between real and fake critic observations
    

    Args:
        real_crc (_type_): _description_
        fake_crc (_type_): _description_
        gbs (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    #uses critic output
    return tf.nn.compute_average_loss(-real_crc +fake_crc, global_batch_size=gbs)

def cwp_loss(folder):
    """
    constrains the model to predict iwp and lwp that add up to the cwp
    needs to rescale all variables for that to work
    """
    try:
        mi,ma = np.load("{}/minmax_both.npy".format(folder))
    except FileNotFoundError:
        print("computing minmax")
        ds = h5py.File(os.path.join(folder, "ESAvsICON.hdf5"),"r")
        s=ds["source"][:]
        t=ds["target"][:]

        mi=np.min([np.min(s, axis=(0,2,3)),np.min(t,axis=(0,2,3))],axis=0)
        mi = np.where(mi>0,np.log(mi+1),0)
        ma = np.max([np.max(s,axis=(0,2,3)),np.max(t,axis=(0,2,3))],axis=0)
        ma = np.where(ma>0, np.log(ma+1),0)

        np.save("{}/minmax_both.npy".format(folder),
                np.array([mi*0.99,ma*1.01]))    
    cwpmi,lwpmi,iwpmi = mi[:3]
    cwpma,lwpma,iwpma = ma[:3]
    assert np.sum((cwpmi,lwpmi,iwpmi))==0,(cwpmi,lwpmi,iwpmi)
    assert np.exp(cwpma)<20000,np.exp(cwpma)
    assert np.exp(lwpma)<20000,np.exp(lwpma)
    assert np.exp(iwpma)<20000,np.exp(iwpma)
    def cwploss(real,fake):
        tf.debugging.assert_all_finite(fake,message="generator produced nan")
        rcwp = tf.math.exp(real[...,0]*cwpma)-1
        rlwp = tf.math.exp(real[...,1]*lwpma)-1
        riwp = tf.math.exp(real[...,2]*iwpma)-1
        fcwp = tf.math.exp(tf.where(fake[...,0]>1,1.,fake[...,0])*\
                           (cwpma-cwpmi)+cwpmi)-1
        flwp = tf.math.exp(tf.where(fake[...,1]>1,1.,fake[...,1])*\
                            (lwpma-lwpmi)+lwpmi)-1
        fiwp = tf.math.exp(tf.where(fake[...,2]>1,1.,fake[...,2])*\
                            (iwpma-iwpmi)+iwpmi)-1
        tf.debugging.assert_less_equal(x=fcwp,y=20000.,message="fake water path too big")
        tf.debugging.assert_less_equal(x=flwp,y=20000.,message="fake liquid path too big")
        tf.debugging.assert_less_equal(x=fiwp,y=20000.,message="fake ice path too big")
        tf.debugging.assert_near(x = rcwp, y= rlwp+riwp,
                                 message="real water paths wrong",
                                summarize=4, atol=0.1, rtol=0.01)
        diff = (fcwp-(flwp+fiwp))**2
        tf.debugging.assert_less_equal(x=diff,y=40000.**2,message="diff too big")
        loss = tf.sqrt(tf.reduce_mean(diff, axis=(1,2)))
        return loss
    return cwploss




def _gradient_penalty(f, inputs, center=0.,gbs=None):
    with tf.GradientTape() as t:
        t.watch(inputs)
        pred = tf.reduce_mean(f(inputs)) # the discriminator is PatchGAN
    grad = t.gradient(pred, inputs)
    norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    gp = tf.nn.compute_average_loss((norm - center)**2,global_batch_size=gbs)

    return gp

def _gradient_penalty_new(f, inputs,outputs, center=1.,gbs=None):
    eps = tf.random.uniform((1,),0,1)
    xhat = eps*inputs+(1-eps)*outputs
    with tf.GradientTape() as t:
        t.watch(xhat)
        crit =f(xhat)
    grad = t.gradient(crit, xhat)
    norm = tf.norm(grad, axis=1)
    gp = tf.nn.compute_average_loss((norm - center)**2,global_batch_size=gbs)

    return gp

def gradient_penalty_fun(mode,center=0.,gbs=None):
    """
    Implementation taken from: https://github.com/LynnHo/CycleGAN-Tensorflow-2/blob/master/tf2gan/loss.py
    :param discriminator:
    :param real:
    :param fake:
    :param mode:
    :return:
    """

    if mode == 'none':
        gp = lambda discriminator, real, fake: tf.constant(0, dtype=real.dtype)
    elif mode == "meschederreal": # https://arxiv.org/pdf/1801.04406.pdf
        gp = lambda discriminator, real, fake: _gradient_penalty(discriminator, real, center=center,gbs=gbs)
    elif mode == "gulrajani": #https://proceedings.neurips.cc/paper/7159-improved-training-of-wasserstein-gans
        gp = lambda discriminator, real, fake: _gradient_penalty_new(discriminator,real, fake, gbs=gbs)
    else:
        raise NotImplementedError("Gradient penalty loss %s not recognized" % mode)

    return gp

def model_setup(names, fwp, folder="dataset_dt"):
    """sets up the forward pass of a consistency model
        using trained models 

    Args:
        names (str): model name suffixes
        fwp (str): which configuration of the forward pass to use

    Raises:
        Exception: refuses to go on if an unknown fwp is specified

    Returns:
        function: forward pass taking varyin input arguments
    """    

    toa_models = [tf.keras.models.load_model(os.path.join(os.environ["WORK"],"models",
                                                        "toa_model_"+name),compile=False) 
                                                        for name in names]
    #i removed all the other options i tried here        
    if fwp == "CRE_clt":
        #the model also predicts cloud cover in addition to the CREs
        toa_model = toa_models[0] 
        #print("only taking lw differences into account")
        def forward_pass(target_fake,source_fake,targ_toa, sour_toa, loss_fct,
                sw=None,step=0):
            fake = tf.concat((target_fake, source_fake),axis=-1)
            tf.debugging.assert_all_finite(fake,message="input to toa model nan")
            mi,ma = np.load(os.path.join(os.environ["WORK"],folder,"minmax_log.npy"))#this should be the correct minmax for the cres, not the toas
            mi = tf.cast(mi, target_fake.dtype)
            ma = tf.cast(ma,target_fake.dtype)
            
            if sw is not None:
                with sw.as_default():
                    for v in range(targ_toa.shape[-1]+sour_toa.shape[-1]):
                        if v<targ_toa.shape[-1]:
                            tf.summary.image(str(v)+"in", targ_toa[:3,:,:,v, tf.newaxis],step=step,max_outputs=3)
                        else:
                            tf.summary.image(str(v)+"in", sour_toa[:3,:,:,v%targ_toa.shape[-1], tf.newaxis],step=step,max_outputs=3)
            targ_toa_only = tf.concat([(targ_toa[...,0,tf.newaxis]- targ_toa[...,4,tf.newaxis]),
                                (targ_toa[...,1,tf.newaxis]- targ_toa[...,3,tf.newaxis])],
                                    axis=-1)-mi[...,:2]
            
            tf.debugging.assert_positive(targ_toa_only+1.,"minmaxing somehow went wrong for targ")
            targ_toa_only = tf.math.log1p(targ_toa_only)/ma[...,:2]
            
            sour_toa_only = tf.concat([(sour_toa[...,0,tf.newaxis] -sour_toa[...,4,tf.newaxis]),
                                    (sour_toa[...,1,tf.newaxis] -sour_toa[...,3,tf.newaxis])],
                                        axis=-1)-mi[...,2:]
            tf.debugging.assert_positive(sour_toa_only+1.,"minmaxing somehow went wrong for sour")
            sour_toa_only = (tf.math.log1p(sour_toa_only)/ma[...,2:])
            targ_toa = tf.concat((targ_toa_only, targ_toa[...,-1,tf.newaxis]),axis=-1)
            sour_toa = tf.concat((sour_toa_only, sour_toa[...,-1,tf.newaxis]),axis=-1)
            tf.debugging.assert_all_finite(targ_toa, "scaled targ_toa nan")
            tf.debugging.assert_all_finite(sour_toa, "scaled sour_toa nan")
            # this returns the tcre(2), tclt, scre(2),sclt
            fake_toa = toa_model(fake, training=False)#assuming that target_fake is adapted version of ESA
            targ_fake_toa, sour_fake_toa = tf.split(fake_toa,2,axis=-1)

            tf.debugging.assert_all_finite(sour_fake_toa, "scaled fakes_toa nan")
            tf.debugging.assert_all_finite(targ_fake_toa, "scaled faket_toa nan")
            
            #ensure that fake image has same toa as original
            targ_loss = loss_fct(targ_toa, sour_fake_toa)
            sour_loss = loss_fct(sour_toa, targ_fake_toa)
            if sw is not None:
                targ_ssim = tf.reduce_mean(SSIM(targ_toa,sour_fake_toa,max_val=1))
                sour_ssim = tf.reduce_mean(SSIM(sour_toa, targ_fake_toa,max_val=1))
                with sw.as_default():
                    figure = image_grid_toa_nonorm(targ_toa)
                    tf.summary.image("Targ TOA",
                    plot_to_image(figure), step=step)
                    figure = image_grid_toa_nonorm(targ_fake_toa)
                    tf.summary.image("Targ FAKE TOA",
                        plot_to_image(figure), step=step)
                    figure = image_grid_toa_nonorm(sour_toa)
                    tf.summary.image("Sour TOA",
                        plot_to_image(figure), step=step)
                    figure = image_grid_toa_nonorm(sour_fake_toa)
                    tf.summary.image("Sour FAKE TOA",
                        plot_to_image(figure), step=step)                                
                    tf.summary.histogram("Targ TOAh",
                        targ_toa, step=step)
                    tf.summary.histogram("Targ FAKE TOAh",
                        targ_fake_toa, step=step)
                    tf.summary.histogram("Sour TOAh",
                        sour_toa, step=step)
                    tf.summary.histogram("Sour FAKE TOAh",
                        sour_fake_toa, step=step)
                    tf.summary.scalar("targ toa loss",
                        targ_loss, step=step)
                    tf.summary.scalar("sour toa loss",
                        sour_loss, step=step)
                    tf.summary.scalar("targ toa SSIM",
                        targ_ssim, step=step)
                    tf.summary.scalar("sour toa SSIM",
                        sour_ssim, step=step)
            return targ_loss, sour_loss
    else: 
        raise Exception("incorrect scheme name")
    return forward_pass



def image_grid_toa(_images):
    # Create a figure to contain the plot.
    #allows for more customizable tensorboard logging
    l=_images.shape[-1]
    figure,ax = plt.subplots(1,l+1,figsize=(5,3),gridspec_kw ={"width_ratios":[5 for _ in range(l)]+[1]})
    mi,ma = np.load(os.path.join(os.environ["WORK"],"dataset_all_dt/minmax_difflin.npy"))
    mean,std = np.load(os.path.join(os.environ["WORK"],"dataset_all_dt/minmax_statlin.npy"))
    mi=(mi-mean)/std
    ma=(ma-mean)/std
    for j in range(l):
        # Start next subplot.
        ax[j].set_xticks([])
        ax[j].set_yticks([])
        ax[j].grid(False)
        ish=ax[j].imshow(_images[0,:,:,j], cmap="viridis",vmin=mi[j],vmax=ma[j])
    
    figure.colorbar(ish,cax=ax[-1])
    figure.tight_layout()
    return figure

def image_grid_toa_nonorm(_images):
    # Create a figure to contain the plot.
    l=_images.shape[-1]
    figure,ax = plt.subplots(1,l+1,figsize=(5,3),gridspec_kw ={"width_ratios":[5 for _ in range(l)]+[1]})
    
    for j in range(l):
        # Start next subplot.
        ax[j].set_xticks([])
        ax[j].set_yticks([])
        ax[j].grid(False)
        ish=ax[j].imshow(_images[0,:,:,j], cmap="viridis",vmin=0,vmax=1)
    
    figure.colorbar(ish,cax=ax[-1])
    figure.tight_layout()
    return figure

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
