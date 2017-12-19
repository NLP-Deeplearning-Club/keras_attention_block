"""加性相似度additive similarity
"""

from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer


class AdditiveSimilarity2DLayer(Layer):
    """加性相似度,最经典的注意力相似度机制,如果是在self attention中\
则该层有一个dim为Key_time_step的向量和一个(Key_dim,Key_time_step)的矩阵作为用于训练的参数
    
    .. math::  Similarity(Key) = v \cdot tanh(W_k\cdot Key)


如果不是在self attention中,则该层有一个dim为Key_time_step的向量和两个(Key_dim,Key_time_step)\
的矩阵作为用于训练的参数

    .. math::  Similarity(Key) = v \cdot tanh(W_k\cdot Key+W_q\cdot Query)
    """
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
