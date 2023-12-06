
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.image import ssim as SSIM
from tensorflow import keras
from dagans import util_model
import os
import io
import h5py


def generator_simple(shape=(64, 64, 4), df=64, l2reg=None, 
                     normtype="batchnorm",
                     groups=1, dropout=False, output_shape =None):
    """basically the same as the WGAN generator just put here again in case i need to change something independently

    """    
    ip = keras.layers.Input(shape, name="ip_gen")
    if output_shape is None:
        output_shape = shape
    if l2reg is None:
        reg = None
    else:
        reg = tf.keras.regularizers.l2(l2reg)

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
                                   activation = "sigmoid")
    out = out_conv(x3)

    return keras.models.Model(inputs=[ip],outputs=[out])



def mk_minmax_log(folder):
    print("making minmax_log")
    ds = h5py.File(os.path.join(os.environ["WORK"],folder,"ESAvsICON.hdf5"), "r")
    s_toa = ds["s_toa"]
    t_toa = ds["t_toa"]
    outputs_all = np.stack([np.hstack((t_toa,s_toa))[:,x] for x in [0,1,5,6]], axis=1)
    #weird order because this is ordered this way in the dataset
    #now contains tlw,tsw,slw,ssw
    outputs_cl = np.stack([np.hstack((t_toa,s_toa))[:,x] for x in [3,4,9,8]], axis=1)
    outputs=(outputs_all-outputs_cl)

    mi = np.min(outputs,axis=(0,2,3)).reshape(1,-1,1,1)
    mi -=np.abs(0.01*mi)
    ma = np.max(outputs,axis=(0,2,3)).reshape(1,-1,1,1)
    ma += np.abs(0.01*ma)
    ma = np.log(1+ma-mi)
    outputs_l = np.log(1+outputs-mi)/ma
    assert np.all(outputs_l>=0)
    
    np.save(os.path.join(os.environ["WORK"],folder,"minmax_log.npy"),
                    np.stack([mi.squeeze(),ma.squeeze()]))
    return mi,ma


def model_wrapper_wclt(model, loss_fct, channels=None,dsfolder="dataset_dt"):
    """Return the forward pass for the model. THis wrapper handles definitions
        like normalization and whatnot

    Args:
        model (tf.keras.models.Model): the model predicting the CREs and clt
        loss_fct (callable): preferably a tf loss function
        channels (list, optional): only here for backward compatibility. Defaults to None.
        dsfolder (str, optional): name of the lowest-level folder. Defaults to "dataset_dt".

    Returns:
        callable: forward pass for model
    """    
    try:
        mi, ma = np.load(os.path.join(os.environ["WORK"], dsfolder,"minmax_log.npy"))
    except FileNotFoundError:
        mi, ma = mk_minmax_log(dsfolder)
    mi = mi[:].reshape(1,1,1,-1)
    ma = ma[:].reshape(1,1,1,-1)
    print(mi)
    print(ma)
    def forward_pass(data,writer,step,training=True):
        target,source, targ_toa, sour_toa,*_ = data
        assert target.shape[-1]==8,target.shape
        tclt = targ_toa[...,-1,tf.newaxis]
        sclt = sour_toa[...,-1,tf.newaxis]
        target = target[...,:-1]
        source = source[...,:-1]
        
        inputs = tf.concat((target,source),axis=-1)
        outputs_allsky_t = tf.stack([tf.concat((targ_toa),axis=-1)[...,x] 
                                for x in [0,1]], axis=-1)
        outputs_allsky_s = tf.stack([tf.concat((sour_toa),axis=-1)[...,x] 
                                for x in [0,1]], axis=-1)
        #weird order because this is ordered this way in the dataset
        #now contains tlw,tsw,slw,ssw
        outputs_cl_t = tf.stack([tf.concat((targ_toa),axis=-1)[...,x] 
                               for x in [4,3]], axis=-1)
        outputs_cl_s = tf.stack([tf.concat((sour_toa),axis=-1)[...,x] 
                               for x in [4,3]], axis=-1)
        cre_s = outputs_allsky_s-outputs_cl_s
        cre_t = outputs_allsky_t-outputs_cl_t
        cre_t = tf.math.log1p(cre_t-mi[...,:2])/ma[...,:2]
        cre_s = tf.math.log1p(cre_s-mi[...,-2:])/ma[...,-2:]
        outputs = tf.concat((cre_t,tclt,cre_s,sclt ),axis=-1)
        
        assert tf.reduce_all(outputs>=0),outputs
        fake = model(inputs,training =training)
        if writer is not None:#during training this saves stuff to tensorboard
            with writer.as_default():
                figure = image_grid_toa_nonorm(outputs[:3,...])
                tf.summary.image(
                    "in", plot_to_image(figure), step=step)
                figure = image_grid_toa_nonorm(fake[:3,...])
                tf.summary.image(
                    "out", plot_to_image(figure), step=step)
                for v in range(fake.shape[-1]):
                    ssim = SSIM(outputs[...,v],fake[...,v],max_val=1)
                    tf.summary.scalar( str(v)+ "SSIM",ssim,step=step)
                    tf.summary.histogram(str(v)+"inh", outputs[...,v],step=step)
                    tf.summary.histogram(str(v)+"outh", fake[...,v],step=step)
                for v in range(inputs.shape[-1]):
                    tf.summary.image("feat"+str(v),inputs[...,v, tf.newaxis],step=step)
                 
        return loss_fct(outputs,fake)
    return forward_pass


def train(epochs, model, fwd_pass, dataset, val_ds, logdir, lr =0.01):
    
    optimizer = tf.keras.optimizers.Adam(lr)
    summary_writer = tf.summary.create_file_writer(logdir)
    pbar=tqdm(total=epochs,leave=True,position=0)
    plateau_test = np.zeros((5,))

    for epoch in range(epochs):
        for i,batch in enumerate(dataset):
            step= epoch*len(dataset)+i  
            with tf.GradientTape() as tape:
                losses = fwd_pass(batch,writer = None,step=step)
                if losses is None:
                    continue
                loss = tf.reduce_mean(losses)
            grads = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            with summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=step)
        val_loss=0
        #to each time plot a different element of the validation dataset
        toplot = np.random.randint(0,len(val_ds),1)
        
        for i, batch in enumerate(val_ds):
            cond = i==toplot
            val_losses = fwd_pass(batch,writer = summary_writer if cond else None,step=step,
                                    training = False)
            #add up loss of each validation batch
            val_loss+=  tf.reduce_mean(val_losses)
        with summary_writer.as_default():
            #record obviously the average loss
            tf.summary.scalar('valloss', val_loss/(i+1), step=step)
        pbar.set_description("Epoch {}, step {}: loss {:.3e}, val_loss {:.3e}".format(epoch,
                                i,loss,val_loss/len(val_ds)))
        pbar.update(1)
        plateau_test[0]=val_loss/len(val_ds)
        plateau_test = np.roll(plateau_test,shift=1)
        if epoch>=10:
            assert np.all(plateau_test>0),plateau_test
            if np.std(plateau_test)<(np.mean(plateau_test)/200):
                print("plateauing, stopping early with {}".format(plateau_test.round(3)))
                break
        #learning rate decay
        if epoch>0 and epoch%5==0:
            old_lr = optimizer.lr.read_value()
            optimizer.lr.assign(old_lr*0.95)
        with summary_writer.as_default():
            tf.summary.scalar('LR',optimizer.lr.read_value() , step=step)
    


def image_grid_toa_nonorm(_images):
    # Create a figure to contain the plot.
    l=_images.shape[-1]
    figure,ax = plt.subplots(1,l+1,figsize=(10,3),gridspec_kw ={"width_ratios":[5 for _ in range(l)]+[1]})
    
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

