import inspect
from keras import backend as K
from keras import initializers
from keras.layers.core import Lambda, Dense, Dropout, Masking
from keras.layers import merge
from keras.engine.topology import Layer


class AdditiveWeightLayer(Layer):
    def __init__(self, kernel_initializer='glorot_uniform',
                 wk_kernel_initializer='glorot_uniform',
                 wq_kernel_initializer='glorot_uniform',
                 **kwargs):
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.wk_kernel_initializer = initializers.get(
            wk_kernel_initializer)
        self.wq_kernel_initializer = initializers.get(
            wq_kernel_initializer)
        super().__init__(**kwargs)

    def build(self, input_shape):
        print(input_shape[0])
        if not isinstance(input_shape, list):
            raise ValueError('A additive weight layer should be called '
                             'on a list of inputs.')
        if len(input_shape) > 2:
            raise ValueError('A additive weight layer should be called '
                             'on a list of at least 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')

        if len(input_shape[0]) != 2:
            raise ValueError('A additive weight layer should be called '
                             'on a list of 2 inputs,and the first one '
                             'for Key must be 2D')
        if len(input_shape) == 2:
            if len(input_shape[1]) != 1:
                raise ValueError('A additive weight layer should be called '
                                 'on a list of 2 inputs,and the second one '
                                 'for Query must be 1D')
            else:
                self.wq_kernel = self.add_weight(
                    name='wq_kernel',
                    shape=(
                        input_shape[0][-1],
                        input_shape[0][-2]),
                    initializer=self.wq_kernel_initializer,
                    trainable=True)

        if input_shape[0][-1] != input_shape[1][-1]:
            raise ValueError('A additive weight layer should be called '
                             'on a list of 2 inputs who have the same dim')

        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(
                                          1, input_shape[0][-2]),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        self.wk_kernel = self.add_weight(
            name='wk_kernel',
            shape=(
                input_shape[0][-1], input_shape[0][-2]),
            initializer=self.wk_kernel_initializer,
            trainable=True)

        # Be sure to call this somewhere!
        super().build(input_shape)

    def call(self, inputs):
        Key = inputs[0]
        print(Key.shape)
        if len(inputs) == 2:
            Query = inputs[1]
            temp = K.dot(Key, self.wk_kernel) + K.dot(Query, self.wq_kernel)
        else:
            Query = None
            temp = K.dot(Key, self.wk_kernel)
        result = K.dot(self.kernel, K.tanh(temp))
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0][0]


def Attention3DBlock(simi_func="dot_product",
                     scale=True,
                     key_masking=True,
                     query_masking=True,
                     causality=False,
                     drop_out=False):
    def attention3D(Key, Value, Query):
        self_name = inspect.stack()[1][3]
        if K.backend() != 'tensorflow':
            raise RuntimeError('attention block are only available '
                               'with the TensorFlow backend.')
        num_units = int(Key.shape[-1])
        num_vector = int(Key.shape[-2])
        if simi_func == 'dot_product':
            K_transport = Lambda(lambda x: K.transpose(x),
                                 name=self_name + '_Key_transport')(Key)
            simi_func_result = merge([Query, K_transport],
                                     name=self_name + '_weight_',
                                     mode='mul')
            # Scale
            if scale:
                simi_func_result = Lambda(
                    lambda x: x / (num_units ** 0.5),
                    name=self_name + '_weight_scale')(
                        simi_func_result)
        elif simi_func == 'dot_product':
            simi_func_result = AdditiveWeightLayer()([Key, Query])
        else:
            raise RuntimeError(
                'unknown similarity function: {}'.format(simi_func))
        # Key Masking

        if key_masking:
            simi_func_result = Masking(0)(simi_func_result)

        if causality:
            pass

        soft_max_result = Dense(num_vector,
                                activation='softmax',
                                name=self_name + '_softmax')(simi_func_result)
        if query_masking:
            pass
        if drop_out and 0 < drop_out < 1:
            soft_max_result = Dropout(
                drop_out,
                name=self_name + '_dropout')(soft_max_result)

        outputs = merge([Value, soft_max_result],
                        name=self_name, mode='mul')
        return outputs
    f = attention3D
    f.__name__ = simi_func + '_' + f.__name__
    return f

def SelfAttention3D(simi_func="dot_product",
                     scale=True,
                     key_masking=True,
                     query_masking=True,
                     causality=False,
                     drop_out=False):

    def self_attention3D(Key, Value, Query):
        self_name = inspect.stack()[1][3]
        if K.backend() != 'tensorflow':
            raise RuntimeError('attention block are only available '
                               'with the TensorFlow backend.')
        num_units = int(Key.shape[-1])
        num_vector = int(Key.shape[-2])
        if simi_func == 'dot_product':
            K_transport = Lambda(lambda x: K.transpose(x),
                                 name=self_name + '_Key_transport')(Key)
            simi_func_result = merge([Query, K_transport],
                                     name=self_name + '_weight_',
                                     mode='mul')
            # Scale
            if scale:
                simi_func_result = Lambda(
                    lambda x: x / (num_units ** 0.5),
                    name=self_name + '_weight_scale')(
                        simi_func_result)
        elif simi_func == 'dot_product':
            simi_func_result = AdditiveWeightLayer()([Key, Query])
        else:
            raise RuntimeError(
                'unknown similarity function: {}'.format(simi_func))
        # Key Masking

        if key_masking:
            simi_func_result = Masking(0)(simi_func_result)

        if causality:
            pass

        soft_max_result = Dense(num_vector,
                                activation='softmax',
                                name=self_name + '_softmax')(simi_func_result)
        if query_masking:
            pass
        if drop_out and 0 < drop_out < 1:
            soft_max_result = Dropout(
                drop_out,
                name=self_name + '_dropout')(soft_max_result)

        outputs = merge([Value, soft_max_result],
                        name=self_name, mode='mul')
        return outputs
    f = self_attention3D
    f.__name__ = simi_func + '_' + f.__name__
    return f