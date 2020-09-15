import tensorflow as tf

class global_concat_layer(tf.keras.layers.Layer):
    def __init__(self,):
        model = tf.keras.layers.concatenate()
        
    def exe_global_concat_layer(tensor, layer_o, tensor_list):
        index = layer_o['index']
        h = tf.shape(tensor)[1]
        w = tf.shape(tensor)[2]
        concat_t = tf.squeeze(tensor_list[index], [1, 2])
        dims = concat_t.get_shape()[-1]
        batch_l = tf.unpack(concat_t, axis=0)
        bs = []
        for batch in batch_l:
            batch = tf.tile(batch, [h * w])
            batch = tf.reshape(batch, [h, w, -1])
            bs.append(batch)
        concat_t = tf.pack(bs)
        concat_t.set_shape(concat_t.get_shape().as_list()[:3] + [dims])
        tensor = tf.concat(3, [tensor, concat_t])
        return tensor