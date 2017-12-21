from collections.abc import Callable, Sequence
import numpy as np
from keras import backend as K
from keras import initializers
from keras import activations
from keras.engine.topology import Layer


class SelfAttention1DLayer(Layer):
    """self-attention的特点是自己为输入,输出也是一个和自己一样shape的张量.

    Attributes:
        similarity (Union[Callable,str]): - 指定使用的相似度计算函数,目前可选的有\
        加性相似度(additive),乘性相似度(multiplicative),点乘相似度(dot_product),\
        当然也可以自己写一个,只要最终输出的是一个(?,output,input_timestep)的
        kernel_size (tuple[int,int]): - 指定使用加性相似度(additive)时才能指定,\
        用于指定第一个权重的形状,各维的意义[输出的纬度,第二个权重矩阵的第一个纬度]
        kernel_initializer (str): - 第一个权重的初始化函数,默认glorot_uniform
        wk_kernel_initializer (str): - 第二个权重的初始化函数,默认glorot_uniform
    """

    def __init__(self, similarity="additive", *,
                 kernel_size=None,
                 kernel_initializer='glorot_uniform',
                 wk_kernel_initializer='glorot_uniform',
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
        if similarity == "additive" and kernel_size is None:
            raise ValueError(
                'additive similarity need '
                'hyperparameter kernel_size!'
            )
        if similarity != "additive" and kernel_size:
            raise ValueError(
                'only additive similarity support '
                'hyperparameter kernel_size!'
            )

        if (isinstance(
            kernel_size,
                Sequence) and len(kernel_size) == 2) or kernel_size is None:
            self.kernel_size = kernel_size
        else:
            raise ValueError(
                'kernel_size must be a Sequence with 2 int element')
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.wk_kernel_initializer = initializers.get(
            wk_kernel_initializer)
        self.dim = None
        if self.similarity == "additive":
            self.kernel = None
            self.wk_kernel = None
            self.wq_kernel = None
        elif self.similarity == "multiplicative":
            self.kernel = None
        super().__init__(**kwargs)

    def _build_w(self, time, dim):
        self.dim = dim
        if self.similarity == "additive":
            r, d_a = self.kernel_size
            self.kernel = self.add_weight(name='kernel',
                                          shape=(r, d_a),
                                          initializer=self.kernel_initializer,
                                          trainable=True)

            self.wk_kernel = self.add_weight(
                name='wk_kernel',
                shape=(d_a, dim),
                initializer=self.wk_kernel_initializer,
                trainable=True)
        elif self.similarity == "multiplicative":
            self.kernel_size = (time, dim)
            self.kernel = self.add_weight(name='kernel',
                                          shape=(
                                              dim, dim),
                                          initializer=self.kernel_initializer,
                                          trainable=True)
        else:
            self.kernel_size = (time, dim)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('A additive weight layer should be called '
                             'by a (batch,time_step,dim)3D inputs.'
                             'Got ' + str(input_shape) + ' inputs.')

        time = input_shape[-2]
        dim = input_shape[-1]
        self._build_w(time, dim)
        # Be sure to call this somewhere!
        super().build(input_shape)

    def multiplicative(self, Source):
        r"""乘性相似度,其中的权重矩阵形状为[dim,dim]\
        输出的固定为与原输入一样形状

        .. math::  Similarity(Source) =  Source\cdot W \cdot Source^T
        """
        Source_t = K.permute_dimensions(Source, (0, 2, 1))
        s = K.dot(Source, self.kernel)
        sim = K.batch_dot(s, Source_t)
        return sim

    def dot_product(self, Source):
        r"""点乘相似度,在google的attention is all you need 中看到的.\
        很迷,没有要训练的矩阵,直接转置点乘,输出的固定为与原输入一样形状

        .. math::  Similarity(Source) = \frac{Source^T\cdot Source}{\sqrt{d_k}}
        """
        Source_t = K.permute_dimensions(Source, (0, 2, 1))
        sim = K.batch_dot(Source, Source_t)
        return sim

    def additive(self, Source):
        r"""
        加性相似度,最经典的注意力相似度机制,如果是在self attention中\
        则该层有两个权重矩阵形状为(r,d_a)和(d_a,dim)

        .. math::  Similarity(Source) = V \cdot tanh(W_k\cdot Source^T)
        """
        Source_t = K.permute_dimensions(Source, (0, 2, 1))
        f_att = K.dot(self.wk_kernel, Source_t)
        f_att = K.permute_dimensions(f_att, (1, 0, 2))
        sim = K.dot(self.kernel, K.tanh(f_att))
        sim = K.permute_dimensions(sim, (1, 0, 2))
        return sim

    def _call_attention(self, Source):
        """self-attention就是通过相似度函数计算得的相似矩阵过softmax后与自身点乘得到

        .. math::  A = Softmax(Similarity(Source))
        .. math::  C = A \cdot Source
        """
        if isinstance(self.similarity, Callable):
            sim = self.similarity(Source)
        else:
            sim = getattr(self, self.similarity)(Source)
        sm = activations.softmax(sim)
        result = K.batch_dot(sm, Source)
        return result

    def call(self, inputs):
        Source = inputs
        result = self._call_attention(Source)
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.kernel_size[0], self.dim)


class SelfAttention2DLayer(SelfAttention1DLayer):
    """self-attention的特点是自己为输入,输出也是一个和自己一样shape的张量.
    2d的self-attention是为2dcnn设计的,其原理就是将4维的中间两维压缩为一维,之后输出的时候再解压缩出来.

    Attributes:
        output_size (tuple[int,int]): - 指定输出的形状,如果是加性相似度(additive),\
        则必须指定,且可以随意指定,如果是其他则可以不指定,那就是原样形状输出,\
        如果指定的话则必须积与原形状第1,2纬度的积相等
        similarity (Union[Callable,str]): - 指定使用的相似度计算函数,目前可选的有\
        加性相似度(additive),乘性相似度(multiplicative),点乘相似度(dot_product),\
        当然也可以自己写一个,只要最终输出的是一个(?,output,input_timestep)的\
        用于指定第一个权重的形状,各维的意义[输出的纬度,第二个权重矩阵的第一个纬度]
        d_a (int): - 使用加性相似度(additive)时才要指定,超参,用于指定第二个权重的第一维
        kernel_initializer (str): - 第一个权重的初始化函数,默认glorot_uniform
        wk_kernel_initializer (str): - 第二个权重的初始化函数,默认glorot_uniform
    """

    def __init__(self, output_size=None,
                 similarity="additive", *,
                 d_a=None,
                 kernel_initializer='glorot_uniform',
                 wk_kernel_initializer='glorot_uniform',
                 **kwargs):
        self.output_size = output_size
        if similarity == "additive":
            if d_a is not None and output_size is not None and len(
                    output_size) == 2:
                kernel_size = (output_size[0] * output_size[1], d_a)
            else:
                raise ValueError(
                    'additive similarity need  hyperparameter d_a,output_size')
        else:
            kernel_size = None

        super().__init__(similarity=similarity,
                         kernel_size=kernel_size,
                         kernel_initializer=kernel_initializer,
                         wk_kernel_initializer=wk_kernel_initializer, **kwargs)

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('A additive weight layer should be called '
                             'by a (batch,time_step,dim)3D inputs.'
                             'Got ' + str(input_shape) + ' inputs.')
        if self.similarity != "additive" and (
                self.output_size is not None) and (
                (input_shape[1] * input_shape[2]) != (
                    self.output_size[0] * self.output_size[1])):
            raise ValueError('output size error ')
        X = input_shape[-3]
        Y = input_shape[-2]
        if self.output_size is None:
            self.output_size = (X, Y)
        time = X * Y
        dim = input_shape[-1]
        self.dim = dim
        self._build_w(time, dim)
        # Be sure to call this somewhere!
        super(SelfAttention1DLayer, self).build(input_shape)

    def call(self, inputs):
        """self-attention就是通过相似度函数计算得的相似矩阵过softmax后与自身点乘得到

        .. math::  A = Softmax(Similarity(Source))
        .. math::  C = A \cdot Source
        """
        input_shape = inputs.shape
        # help(input_shape[1])
        Source = K.reshape(
            inputs, shape=np.asarray(
                [-1, (input_shape[1] * input_shape[2]).value,
                    input_shape[3].value]))
        _result = self._call_attention(Source)
        result = K.reshape(
            _result,
            np.asarray([-1,
                        self.output_size[0],
                        self.output_size[1],
                        self.dim]))
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0],  self.output_size[0],
                self.output_size[1], self.dim)


__all__ = ["SelfAttention1DLayer", "SelfAttention2DLayer"]
