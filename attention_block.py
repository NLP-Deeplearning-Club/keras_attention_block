from keras import backend as K
from keras.layers.core import Lambda, Dense, Dropout, Masking
from keras.layers import merge


def attention_block(Query, Key, Value, *,
                    simi_func="dot_product",
                    scale=True,
                    key_masking=True,
                    query_masking=True,
                    causality=False,
                    drop_out=False):
    if K.backend() != 'tensorflow':
        raise RuntimeError('attention block are only available '
                           'with the TensorFlow backend.')
    num_units = int(Query.shape[-1])
    k_num_units = int(Key.shape[-1])
    num_vector = int(Key.shape[-2])
    if simi_func == 'dot_product':
        K_transport = Lambda(lambda x: K.transpose(x),
                             name='attention_Key_transport')(Key)
        simi_func_result = merge([Query, K_transport],
                                 name='attention_weight_dot_product', mode='mul')
        # Scale
        if scale:
            simi_func_result = Lambda(
                lambda x: x / (k_num_units ** 0.5),
                name='attention_weight_dot_product_scale')(simi_func_result)
    else:
        raise RuntimeError(
            'unknown similarity function: {}'.format("simi_func"))
    # Key Masking

    if key_masking:
        simi_func_result = Masking(0)(simi_func_result)

    if causality:
        pass

    soft_max_result = Dense(num_vector,
                            activation='softmax',
                            name='attention_softmax')(simi_func_result)
    if drop_out and 0 < drop_out < 1:
        soft_max_result = Dropout(drop_out,
                                  name='attention_dropout')(soft_max_result)

    outputs = merge([Value, soft_max_result], name='attention_mul', mode='mul')
    return outputs
