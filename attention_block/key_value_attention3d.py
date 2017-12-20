from collections.abc import Callable
from keras import backend as K
from keras import initializers
from keras import activations
from keras.engine.topology import Layer


class Key_Value_Attention3DLayer(Layer):
    """key-value-attention3d的特点是三个输入Key,Value,Query均不指定,输出的是Query的timestep为长度,dim一致的张量
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
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError('A attention layer should be called '
                             'on a list of 3 inputs.')

        if len(input_shape[0]) != 3 or len(
                input_shape[1]) != 3 or len(input_shape[2]) != 3:
            raise ValueError('A attention layer should be called '
                             'by 3 (batch,time_step,dim)3D inputs.'
                             'Got ' + str(input_shape) + ' inputs.')
        if not (input_shape[0][-1] == input_shape[
                1][-1] == input_shape[2][-1]):
            raise ValueError('A attention layer should be called '
                             'by 3 (batch,time_step,dim)3D inputs '
                             'with the same dim.'
                             'Got ' + str(input_shape) + ' inputs.')
        dim = input_shape[0][-1]
        s_time = input_shape[0][-2]
        q_time = input_shape[2][-2]
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

    def multiplicative(self, Key, Query):
        """点乘相似度,在google的attention is all you need 中看到的.\
很迷,没有要训练的矩阵,直接转置点乘

        .. math::  Similarity(Key) = Key^T \cdot W \cdot Query
        """

        sim = K.dot(K.dot(Key, self.kernel), Query)
        return sim

    def dot_product(self, Key, Query):
        r"""点乘相似度,在google的attention is all you need 中看到的.\
很迷,没有要训练的矩阵,直接转置点乘

        .. math::  Similarity(Key) = \frac{Source^T\cdot Query}{\sqrt{d_k}}
        """
        sim = K.dot(Key, Query)
        return sim

    def additive(self, Key, Query):
        """"
        加性相似度,最经典的注意力相似度机制,如果是在self attention中\
则该层有一个dim为Key_time_step的向量和一个(Key_dim,Key_time_step)的矩阵作为用于训练的参数

        .. math::  Similarity(Key)=V \cdot tanh(W_k\cdot Key+W_q\cdot Query)
        """
        f_att = K.dot(Key, self.wk_kernel) + K.dot(self.wq_kernel, Query)
        sim = K.dot(K.tanh(f_att), self.kernel)
        return sim

    def call(self, inputs):
        """self-attention就是通过相似度函数计算得的相似矩阵过softmax后与自身点乘得到
        .. math::  A = Softmax(Similarity(Source,Query))
        .. math::  C = A \cdot Source
        """
        Key = inputs[0]
        Value = inputs[1]
        Query = K.permute_dimensions(inputs[2], (0, 2, 1))
        if isinstance(self.similarity, Callable):
            sim = self.similarity(Key, Query)
        else:
            sim = getattr(self, self.similarity)(Key, Query)
        sm = activations.softmax(sim)
        sm_t = K.permute_dimensions(sm, (0, 2, 1))
        result = K.batch_dot(sm_t, Value)
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[-1]
