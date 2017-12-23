
# keras attention block

+ version: 0.0.2
+ status: dev
+ author: hsz
+ email: hsz1273327@gmail.com

## 简介

keras-attention-block是一个关于attention的keras的扩展模块.可以和其他自带的层一样使用.并且没有额外依赖.整个模块并不依赖固定的后端,但本人只有tensorflow作为后端,因此其他后端并未测试.

keywords:keras,deeplearning,attention

## Feature

+ 支持1D的attention,可以输入形如(batch,time,dim)的训练数据

+ 支持2D的attention,可以输入形如(batch,X,Y,dim)的训练数据

+ 支持self-attention,输入为一个tensor,可以使用加性,乘性,点积,以及线性四种定义好的相似算法

+ 支持attention,输入为两个tensor,可以使用加性,乘性,点积三种定义好的相似算法

+ 支持key-value attention,可以使用加性,乘性,点积三种定义好的相似算法

+ 支持multihead attention.

+ 支持自定义相似算法用于计算Key和Query的相似性

+ 支持自定义合并算法用于合并相似算法的结果Value

## 例子

```python
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.layers import Convolution2D
from keras.models import *
from keras.layers.normalization import BatchNormalization
from keras_attention_block import *

INPUT_DIM = 32
TIME_STEPS = 20
SINGLE_ATTENTION_VECTOR = False
APPLY_ATTENTION_BEFORE_LSTM = False

inputs = Input(shape=(TIME_STEPS, INPUT_DIM))
attention_mul =  SelfAttention1DLayer(similarity="linear",dropout_rate=0.2)(inputs)#MyLayer((20,32))(inputs)#
lstm_units = 32
#attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
attention_mul = Flatten()(attention_mul)
output = Dense(1, activation='sigmoid')(attention_mul)
m = Model(inputs=[inputs], outputs=output)

m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(m.summary())

train_data = np.random.random((1000,20,32))
train_lab = np.random.randint(0,2,1000)
m.fit(train_data,train_lab , epochs=1, batch_size=100 )

```

## 安装

`python -m pip install keras_attention_block`

## 文档

文档部署在github page

<https://nlp-deeplearning-club.github.io/keras_attention_block/>

## 未完成的

+ 3D attention
