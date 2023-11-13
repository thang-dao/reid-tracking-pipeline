import os
import cv2
import tensorflow as tf
import numpy as np
# from reidd.model_base import BaseReidModel


class ConvnextNetReid():
# class EfficientNetReid(BaseReidModel):
    def __init__(self, cfg, device):
        self.cfg = cfg
        os.environ["CUDA_VISIBLE_DEVICES"]= device
        self.model = self.load_model()

    def apply_transform(self, img):
        img = cv2.resize(img, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        return img

    def load_model(self):
        strategy = auto_select_accelerator()
        with strategy.scope():
            if self.cfg['version'] == 'tiny':
                model = tf.keras.models.load_model(self.cfg['weights'])
            elif self.cfg['version'] == 'medium':
                custom_objects = {
                    'resmlp>ChannelAffine' : ChannelAffine,
                    'nfnets>ZeroInitGain' : ZeroInitGain,
                }
                model = tf.keras.models.load_model(self.cfg['weights'], custom_objects=custom_objects)
            elif self.cfg['version'] == 'novelty':
                custom_objects = {
                    'resmlp>ChannelAffine' : ChannelAffine,
                    'ChannelAffine' : ChannelAffine,
                    'nfnets>ZeroInitGain' : ZeroInitGain,
                    'CrossAttn' : CrossAttn,
                }
                model = tf.keras.models.load_model(self.cfg['weights'], custom_objects=custom_objects)

        return model
    
    def run(self, batch_data):
        input_model = np.stack([self.apply_transform(img) for img in batch_data])
        embedding = self.model(input_model)
        normalized_embedd = tf.linalg.normalize(embedding, axis=1)
        return normalized_embedd[0].numpy()
    

class ChannelAffine(tf.keras.layers.Layer):
    def __init__(self, use_bias=True, weight_init_value=1, axis=-1, **kwargs):
        super(ChannelAffine, self).__init__(**kwargs)
        self.use_bias, self.weight_init_value, self.axis = use_bias, weight_init_value, axis
        self.ww_init = tf.keras.initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
        self.bb_init = "zeros"
        self.supports_masking = False

    def build(self, input_shape):
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            ww_shape = (input_shape[-1],)
        else:
            ww_shape = [1] * len(input_shape)
            axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
            for ii in axis:
                ww_shape[ii] = input_shape[ii]
            ww_shape = ww_shape[1:]  # Exclude batch dimension

        self.ww = self.add_weight(name="weight", shape=ww_shape, initializer=self.ww_init, trainable=True)
        if self.use_bias:
            self.bb = self.add_weight(name="bias", shape=ww_shape, initializer=self.bb_init, trainable=True)
        super(ChannelAffine, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs * self.ww + self.bb if self.use_bias else inputs * self.ww

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(ChannelAffine, self).get_config()
        config.update({"use_bias": self.use_bias, "weight_init_value": self.weight_init_value, "axis": self.axis})
        return config

class ZeroInitGain(tf.keras.layers.Layer):
    def __init__(self, use_bias=False, weight_init_value=0, bias_init_value=0, **kwargs):
        super().__init__(**kwargs)
        self.use_bias = use_bias
        self.ww_init = tf.keras.initializers.Constant(weight_init_value) if weight_init_value != 0 else "zeros"
        self.bb_init = tf.keras.initializers.Constant(bias_init_value) if bias_init_value != 0 else "zeros"

    def build(self, input_shape):
        self.gain = self.add_weight(name="gain", shape=(), initializer=self.ww_init, dtype="float32", trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name="bias", shape=(), initializer=self.bb_init, dtype="float32", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return (inputs * self.gain + self.bias) if self.use_bias else (inputs * self.gain)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"use_bias": self.use_bias})
        return base_config


class CrossAttn(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CrossAttn, self).__init__(**kwargs)
        self.gamma_init = "zeros"

    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma", shape=(), initializer=self.gamma_init, dtype="float32", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        xa, xb = inputs
        m_bs = tf.shape(xa)[0]
        height = tf.shape(xa)[1]
        width = tf.shape(xa)[2]
        C = xa.shape[-1]

        querya = tf.reshape(xa, [m_bs, -1, C])
        keya = tf.transpose(tf.reshape(xa, [m_bs, -1, C]), perm=[0, 2, 1])

        queryb = tf.reshape(xb, [m_bs, -1, C])
        keyb = tf.transpose(tf.reshape(xb, [m_bs, -1, C]), perm=[0, 2, 1])

        energya = tf.linalg.matmul(querya, keyb, transpose_b=False)
        energyb = tf.linalg.matmul(queryb, keya, transpose_b=False)

        def get_output(energy, xin):
            max_energy_0 = tf.reduce_max(energy, axis=-1, keepdims=True)
            energy_new = max_energy_0 - energy
            attention = tf.nn.softmax(energy_new, axis=-1)
            proj_value = tf.reshape(xin, [m_bs, -1, C])

            out = tf.linalg.matmul(attention, proj_value)
            out = tf.reshape(out, [m_bs, height, width, C])
            out = self.gamma * out + xin
            return out

        return get_output(energya, xa), get_output(energyb, xb)

    def get_config(self):
        base_config = super().get_config()
        return base_config
    

def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('Using TPU')
    except:
        strategy = tf.distribute.MirroredStrategy()
        if strategy.num_replicas_in_sync == 1:
            strategy = tf.distribute.get_strategy()
            print('Using 1 GPU')
        else:
            print('Using GPUs')
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy