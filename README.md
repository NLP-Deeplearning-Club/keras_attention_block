# keras attention block

+ version: 0.0.2
+ status: dev
+ author: hsz
+ email: hsz1273327@gmail.com

## Description

keras-attention-block is an extension for keras to add attention. It was born from lack of existing function to add attention inside keras.
The module itself is pure Python with no dependencies on modules or packages outside the standard Python distribution and keras.

keywords:keras,deeplearning,attention

## Feature

+ support one dimensional attention, that is to take in inputs whose dimensions are batch_size * time_step * hidden_size
+ support two dimensional attention, that is to take in inputs of dimensions are batch_size * X * Y * hidden_size
+ support self-attention, that is to take in tensors. Four well defined calculations are included : additive, multiplicative, dot-product based and  as well as linear.
+ support attention, that is to take in two tensors. Three well defined calculations are included : additive, multiplicative and dot product based.
+ support attention. Three well defined calculations are included : additive, multiplicative and dot product based.
+ support multihead attention
+ support customized calculations of similarity between Key and Query
+ support customized calculations of Value

## Example

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

## Install

`python -m pip install keras_attention_block`

## Documentation

Documentation on github page <https://nlp-deeplearning-club.github.io/keras_attention_block/>

## TODO

+ 3D attention
