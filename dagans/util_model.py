import tensorflow as tf
import numpy as np
import h5py
import os
NORM_OFF_PROBAV = np.array([0.43052389, 0.40560079, 0.46504526, 0.23876471])


class InstanceNormalization(tf.keras.layers.Layer):
  """
  Instance Normalization Layer (https://arxiv.org/abs/1607.08022).
  Taken from : tensorflow_examples.models.pix2pix import pix2pix.
  """

  def __init__(self, **kwargs):
    super(InstanceNormalization, self).__init__(**kwargs)
    self.epsilon = 1e-5

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

    def get_config(self):
        config =  super(InstanceNormalization, self).get_config()
        config.update({"scale": self.scale, "offset": self.offset, "epsilon": self.epsilon})
        return config

def conv_blocks_sep(ip_, nfilters, axis_batch_norm, reg, name, batch_norm,
                remove_bias_if_batch_norm=False, dilation_rate=(1,1),
                normtype="batchnorm",groups=1):
    """convolutional block with seperable conv layers

    Args:
        ip_ (tf.Tensor): input 
        nfilters (int): base convolution layer filters
        axis_batch_norm (int?): axis to normalize
        reg (bool?): wether to do l2 regularization
        name (str): name of the block for tensorflow
        batch_norm (bool): batch_norm
        remove_bias_if_batch_norm (bool, optional): wysiwyg. Defaults to False.
        dilation_rate (tuple, optional):  Defaults to (1,1).
        normtype (str, optional):  Defaults to "batchnorm".
        groups (int, optional): whether to use grouped convolutions which only seem to make things worse
                                since num groups is set to 1 in the code this doenst really matter. Defaults to 1.

    Raises:
        NotImplementedError: if normtype is unknown
        

    Returns:
        activation: tensorflow stack of layers: (Conv2D ,(norm),relu,Conv2D,(norm),relu)
    """    
    use_bias = not (remove_bias_if_batch_norm and batch_norm)

    conv = tf.keras.layers.SeparableConv2D(nfilters, (3, 3),
                                           padding='same',# so there is 0 padding
                                           name=name+"_conv_1",
                                           depthwise_regularizer=reg,
                                           pointwise_regularizer=reg,
                                           dilation_rate=dilation_rate,
                                           use_bias=use_bias,
                                           groups=groups,
                                           activation="linear")(ip_)

    if batch_norm:
        if normtype == "batchnorm":
            conv = tf.keras.layers.BatchNormalization(axis=axis_batch_norm,name=name + "_bn_1")(conv)
        elif normtype == "instancenorm":
            conv = InstanceNormalization(name=name + "_bn_1")(conv)
        else:
            raise NotImplementedError("Unknown norm %s" % normtype)

    conv = tf.keras.layers.Activation('relu',name=name + "_act_1")(conv)


    conv = tf.keras.layers.SeparableConv2D(nfilters, (3, 3),
                           padding='same',name=name+"_conv_2",
                           use_bias=use_bias,dilation_rate=dilation_rate,
                           depthwise_regularizer=reg,pointwise_regularizer=reg)(conv)

    if batch_norm:
        if normtype == "batchnorm":
            conv = tf.keras.layers.BatchNormalization(axis=axis_batch_norm,name=name + "_bn_2")(conv)
        elif normtype == "instancenorm":
            conv = InstanceNormalization(name=name + "_bn_2")(conv)
        else:
            raise NotImplementedError("Unknown norm %s" % normtype)


    return tf.keras.layers.Activation('relu',name=name + "_act_2")(conv)

def conv_blocks_ch(ip_, nfilters, axis_batch_norm, reg, name, batch_norm,
                remove_bias_if_batch_norm=False, dilation_rate=(1,1),normtype="batchnorm",
                groups=1):
    """same as above but with standard conv2d blocks"""
    use_bias = not (remove_bias_if_batch_norm and batch_norm)
    
    conv = tf.keras.layers.Conv2D(nfilters
                                  , (3, 3),
                                           padding='same',
                                           name=name+"_conv_1",
                                           kernel_regularizer=reg,
                                           dilation_rate=dilation_rate,
                                           use_bias=use_bias,
                                           groups=groups)(ip_)

    if batch_norm:
        if normtype == "batchnorm":
            conv = tf.keras.layers.BatchNormalization(axis=axis_batch_norm,name=name + "_bn_1")(conv)
        elif normtype == "instancenorm":
            conv = InstanceNormalization(name=name + "_bn_1")(conv)
        else:
            raise NotImplementedError("Unknown norm %s" % normtype)

    conv = tf.keras.layers.Activation('relu',name=name + "_act_1")(conv)


    conv = tf.keras.layers.Conv2D(nfilters
                                  , (3, 3),
                           padding='same',name=name+"_conv_2",
                           use_bias=use_bias,dilation_rate=dilation_rate,
                           kernel_regularizer=reg,
                           groups=groups)(conv)

    if batch_norm:
        if normtype == "batchnorm":
            conv = tf.keras.layers.BatchNormalization(axis=axis_batch_norm,name=name + "_bn_2")(conv)
        elif normtype == "instancenorm":
            conv = InstanceNormalization(name=name + "_bn_2")(conv)
        else:
            raise NotImplementedError("Unknown norm %s" % normtype)


    return tf.keras.layers.Activation('relu',name=name + "_act_2")(conv)

