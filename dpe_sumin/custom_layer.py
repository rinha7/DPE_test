import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from data_init import *
from functools import partial

#custom layer 목록
# 이후 class로 바꿀 수 있다면 바꿀 것

def lrelu_layer(leak):
    return dict(
        name='lrelu',
        leak=leak)
def exe_lrelu_layer(tensor, layer_o): # leaky ReLU 를 구현한 layer
    leak = layer_o['leak']
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    tensor = f1 * tensor + f2 * abs(tensor)

    return tensor


def global_concat_layer(index):
    return dict(
        name='g_concat',
        index=index)
def exe_global_concat_layer(tensor, tensor_list, index): # global_concat 에 사용되는 layer
    h = tf.shape(tensor)[1]
    w = tf.shape(tensor)[2]
    concat_t = tf.squeeze(tensor_list[index],[1,2])
    dims = concat_t.shape[-1]
    batch_l = tf.unstack(concat_t, axis=0)
    bs = []

    for batch in batch_l:
        batch = tf.tile(batch, [h * w])
        batch = tf.reshape(batch, [h, w, -1])
        bs.append(batch)
    concat_t = tf.stack(bs)
    concat_t.set_shape(concat_t.get_shape().as_list()[:3] + [dims])
    tensor = tf.concat([tensor, concat_t], axis=-1)

    return tensor


def resize_layer(scale, method, align_corners=False):
    return dict(
        name='resize',
        scale=scale,
        method=method,
        align_corners=align_corners)
def exe_resize_layer(tensor, scale, align_corners=False):
    t_shape = tensor.get_shape().as_list()
    if t_shape[1] == None or t_shape[2] == None:
        t_shape = tf.shape(tensor)

    t_size = [t_shape[1] * scale, t_shape[2] * scale]
    tensor = tf.image.resize(tensor,t_size,tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return tensor


def res_layer(index, axis=None):
    return dict(
        name='res',
        index=index,
        axis=axis)
def exe_res_layer(tensor, tensor_list, index, axis):
    res_tensor = tensor_list[index]

    if axis is not None:
        l = [res_tensor[:,:,:,i] for i in axis]
        res_tensor = tf.stack(l,-1)
    tensor = tf.add(tensor, res_tensor)

    return tensor


def concat_layer(index):
    return dict(
        name='concat',
        index=index)
def exe_concat_layer(tensor, layer_o, tensor_list):
    index = layer_o['index']
    concat_t = tensor_list[index]
    tensor = tf.concat(3, [tensor, concat_t])
    return tensor


def selu_layer():
    return dict(name='selu')
def exe_selu_layer(tensor):
    #alpha = 1.6732632423543772848170429916717
    #scale = 1.0507009873554804934193349852946
    alpha, scale = (1.0198755295894968, 1.0026538655307724)
    return scale*tf.where(tensor>=0.0, tensor, alpha*tf.nn.elu(tensor))

def conv_layer(kernel, stride, filter, pad_mode):
    return dict(
        name='conv',
        kernel=kernel,
        stride=stride,
        filter=filter,
        pad_mode=pad_mode)
def exe_conv_layer(tensor, layer_o, net_info, l_indx, is_first, is_training, trainable, seed):
    # get_variable을 이용한 namescope variable control은 무시한채 진행한다

    kernel = layer_o['kernel']
    stride = layer_o['stride']
    filter = layer_o['filter']
    pad_mod = layer_o['pad_mode']

    if pad_mod == 'same':
        tensor = conv2d_layer_same(filter,kernel,stride)(tensor)
    else:
        tensor = conv2d_layer_valid(filter,kernel,stride)(tensor)

    return tensor


conv2d_layer_same = partial(keras.layers.Conv2D,
                            padding="same",
                            kernel_initializer=keras.initializers.VarianceScaling())
conv2d_layer_valid = partial(keras.layers.Conv2D,
                             padding="valid",
                             kernel_initializer=keras.initializers.VarianceScaling())


def reduce_mean_layer(axis=None, keep_dims=False):
    return dict(name='reduce_mean', axis=axis, keep_dims=keep_dims)

def exe_reduce_mean_layer(tensor, layer_o):
    axis = layer_o['axis']
    keep_dims = layer_o['keep_dims']
    return tf.reduce_mean(tensor, axis, keep_dims)


# residual block
# 추후 추가 필요함

def exe_res_block():
    return None