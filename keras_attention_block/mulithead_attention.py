import math
import keras.backend as K
from keras.layers import Dense, Lambda
from keras.layers.normalization import BatchNormalization
from .self_attention import SelfAttention1DLayer


class MulitheadAttention:
    """多头注意力机制(MulitheadAttention).是我在google的all you need is attention中看到的注意力机制.
    其核心思想是将dim一层拆分后各自单独进入attention中,以适用于多GPU并行计算,有点map-reduce的意思在里面.
    在实现上,我使用的是:
    linear_layer一层线性层将输入的dim扩展到相同的某个数,
        |
    split_

    之后拆分,过attention再
    """

    def __init__(self,
                 heads=5,
                 input_linear_kwargs={
                     'units': 60,
                     'activation': 'relu'
                 },
                 output_linear_kwargs={
                     'activation': 'relu'
                 },
                 attention=SelfAttention1DLayer,
                 attention_kwargs={
                     'similarity': 'dot_product'
                 }):
        """
        切分dim为几个小块,inputs必须为一个list
        """
        if input_linear_kwargs.get("units") is None:
            raise ValueError(
                "input linear layer's units must be set"
            )
        if output_linear_kwargs.get("units") is not None:
            raise ValueError(
                "output linear layer's units can not be set"
            )

        self.heads = heads
        self.input_linear_kwargs = input_linear_kwargs
        self.output_linear_kwargs = output_linear_kwargs
        self.attention = attention
        self.attention_kwargs = attention_kwargs
        self.step = math.ceil(input_linear_kwargs.get("units") / heads)

    def __call__(self, inputs):
        datas = [Dense(
            name='mulithead_' + self.attention.__name__ + "input_linear_layer",
            **self.input_linear_kwargs)(inpu) for inpu in inputs]

        mulithead_datas = [
            Lambda(lambda x:[x[:, :, i * self.step:(
                i + 1) * self.step] for i in range(
                self.heads)],
                name='mulithead_' + self.attention.__name__ + "split_layer"
            )(j) for j in datas]
        att_res = []
        for data in zip(*mulithead_datas):
            if len(data) > 1:
                res = self.attention(**self.attention_kwargs)(list(data))
            else:
                res = self.attention(**self.attention_kwargs)(data[0])
            att_res.append(res)

        datas = Lambda(
            lambda x: K.concatenate(x, axis=-1),
            name='mulithead_' + self.attention.__name__ + "concatenate_layer")(
                att_res)
        if len(inputs) == 3:
            value = inputs[1]
        else:
            value = inputs[0]
        self.output_linear_kwargs["units"] = int(value.shape[-1])
        datas = Dense(
            name=(
                'mulithead_' + self.attention.__name__ + "output_linear_layer"
            ),
            **self.output_linear_kwargs)(datas)
        outputs = BatchNormalization()(datas)
        print(outputs)
        return datas


__all__ = ["MulitheadAttention"]
