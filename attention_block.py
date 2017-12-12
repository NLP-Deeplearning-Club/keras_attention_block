from keras import backend as K
from keras.layers.core import Lambda, Dense, Dropout, Masking, RepeatVector
from keras.layers import merge
from keras.layers.convolutional import Conv1D


def attention_block(Query, Key, Value, *,
                    simi_func="dot_product",
                    scale=True,
                    key_masking=False,
                    query_masking=False,
                    causality=False,
                    drop_out=False):
    """
    Query batch_size*dim
    Key batch_size*time_steps*dim
    Value batch_size*time_steps*dim
    simi_func :- dot_product,additive
    """
    if K.backend() != 'tensorflow':
        raise RuntimeError('attention block are only available '
                           'with the TensorFlow backend.')
    dim = int(Key.shape[-1])
    time_steps = int(Key.shape[-2])
    if simi_func == 'dot_product':
        K_transport = Permute((2, 1))(Key)
        # K_transport = Reshape((dim, time_steps))(a)
        simi_func_result = merge([Query, K_transport],
                                 name='attention_weight_dot_product', mode='mul')
        # Scale
        if scale:
            simi_func_result = Lambda(
                lambda x: x / (k_dim ** 0.5),
                name='attention_weight_dot_product_scale')(simi_func_result)
    elif simi_func == 'additive':

        w2_q = Dense(dim, use_bias=False)(Query)
        re_w2_q = RepeatVector(time_steps)(w2_q)

        w1_k = Conv1D(time_steps, 1, padding="same", use_bias=False)(Key)

        w = merge([w1_k, re_w2_q], name='additive_attention_w', mode='add')
        tanh_w = Lambda(lambda x: K.tanh(x))(w)
        simi_func_result = Dense(
            time_steps, use_bias=False)(tanh_w)
    else:
        raise RuntimeError(
            'unknown similarity function: {}'.format("simi_func"))
    # Key Masking

    if key_masking:
        simi_func_result = Masking(0)(simi_func_result)

    if causality:
        pass

    soft_max_result = Dense(time_steps,
                            activation='softmax',
                            name='attention_softmax')(simi_func_result)
    if drop_out and 0 < drop_out < 1:
        soft_max_result = Dropout(drop_out,
                                  name='attention_dropout')(soft_max_result)

    outputs = merge([Value, soft_max_result], name='attention_mul', mode='mul')
    return outputs
