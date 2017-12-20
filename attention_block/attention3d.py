from collections.abc import Callable
from keras import backend as K
from keras import initializers
from keras import activations
from keras.engine.topology import Layer


class Attention3DLayer(Layer):
    """attention3d的特点是自己为输入的Key和Value,输出的是Query的timestep为长度,dim一致的张量
    """

    def __init__(self, similarity="additive",
                 kernel_initializer='glorot_uniform',
                 wk_kernel_initializer='glorot_uniform',
                 wq_kernel_initializer='glorot_uniform',
                 **kwargs):
        if isinstance(similarity, Callable):
            self.similarity = similarity
        elif isinstance(similarity, str) and similarity in (
                "multiplicative", "dot_product", "additive"):
            self.similarity = similarity
        else:
            raise ValueError(
                'similarity now only support '
                '"multiplicative","dot_product","additive",'
                'and you can input a function as the similarity function!'
            )

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.wk_kernel_initializer = initializers.get(
            wk_kernel_initializer)
        self.wq_kernel_initializer = initializers.get(
            wq_kernel_initializer)
        super().__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A attention layer should be called '
                             'on a list of 2 inputs.')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError('A attention layer should be called '
                             'by 2 (batch,time_step,dim)3D inputs.'
                             'Got ' + str(input_shape) + ' inputs.')
        if input_shape[0][-1] != input_shape[1][-1]:
            raise ValueError('A attention layer should be called '
                             'by 2 (batch,time_step,dim)3D inputs '
                             'with the same dim.'
                             'Got ' + str(input_shape) + ' inputs.')
        dim = input_shape[0][-1]
        s_time = input_shape[0][-2]
        q_time = input_shape[1][-2]
        if self.similarity == "additive":
            self.kernel = self.add_weight(
                name='kernel',
                shape=(s_time, q_time),
                initializer=self.kernel_initializer,
                trainable=True)
            self.wk_kernel = self.add_weight(
                name='wk_kernel',
                shape=(dim, q_time),
                initializer=self.wk_kernel_initializer,
                trainable=True)
            self.wq_kernel = self.add_weight(
                name='wq_kernel',
                shape=(s_time, dim),
                initializer=self.wk_kernel_initializer,
                trainable=True)
        elif self.similarity == "multiplicative":
            self.kernel = self.add_weight(
                name='kernel',
                shape=(
                    dim, dim),
                initializer=self.kernel_initializer,
                trainable=True)

        # Be sure to call this somewhere!
        super().build(input_shape)

    def multiplicative(self, Source, Query):
        """点乘相似度,在google的attention is all you need 中看到的.\
很迷,没有要训练的矩阵,直接转置点乘

        .. math::  Similarity(Source) = Source^T \cdot W \cdot Query
        """

        sim = K.dot(K.dot(Source, self.kernel), Query)
        return sim

    def dot_product(self, Source, Query):
        r"""点乘相似度,在google的attention is all you need 中看到的.\
很迷,没有要训练的矩阵,直接转置点乘

        .. math::  Similarity(Source) = \frac{Source^T\cdot Query}{\sqrt{d_k}}
        """
        sim = K.dot(Source, Query)
        return sim

    def additive(self, Source, Query):
        """"
        加性相似度,最经典的注意力相似度机制,如果是在self attention中\
则该层有一个dim为Key_time_step的向量和一个(Key_dim,Key_time_step)的矩阵作为用于训练的参数

        .. math::  Similarity(Source)=V \cdot tanh(W_k\cdot Source+W_q\cdot Query)
        """
        f_att = K.dot(Source, self.wk_kernel) + K.dot(self.wq_kernel, Query)
        sim = K.dot(K.tanh(f_att), self.kernel)
        return sim

    def call(self, inputs):
        """self-attention就是通过相似度函数计算得的相似矩阵过softmax后与自身点乘得到
        .. math::  A = Softmax(Similarity(Source,Query))
        .. math::  C = A \cdot Source
        """
        Source = inputs[0]
        Query = K.permute_dimensions(inputs[1], (0, 2, 1))
        if isinstance(self.similarity, Callable):
            sim = self.similarity(Source, Query)
        else:
            sim = getattr(self, self.similarity)(Source, Query)
        sm = activations.softmax(sim)
        sm_t = K.permute_dimensions(sm, (0, 2, 1))
        result = K.batch_dot(sm_t, Source)
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[-1]



