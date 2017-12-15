from keras import backend as K
from keras.layers.core import Lambda, Dense, Dropout, Masking
from keras.layers import merge
from .layers.additive_similarity_layer import AdditiveSimilarity3DLayer


class Attention3DBlock:

    def __init__(self, simi_func="dot_product",
                 scale=True,
                 key_masking=True,
                 query_masking=True,
                 causality=False,
                 drop_out=False):
        self.simi_func = simi_func
        self.scale = scale
        self.key_masking = key_masking
        self.query_masking = query_masking
        self.causality = causality
        self.drop_out = drop_out

    def __call__(self, Key, Value, Query):
        self_name = self.__class__.__name__
        if K.backend() != 'tensorflow':
            raise RuntimeError('attention block are only available '
                               'with the TensorFlow backend.')
        num_units = int(Key.shape[-1])
        num_vector = int(Key.shape[-2])
        if self.simi_func == 'dot_product':
            K_transport = Lambda(lambda x: K.transpose(x),
                                 name=self_name + '_Key_transport')(Key)
            simi_func_result = merge([Query, K_transport],
                                     name=self_name + '_weight_',
                                     mode='mul')
            # Scale
            if self.scale:
                simi_func_result = Lambda(
                    lambda x: x / (num_units ** 0.5),
                    name=self_name + '_weight_scale')(
                        simi_func_result)
        elif self.simi_func == 'Additive':
            simi_func_result = AdditiveSimilarity3DLayer()([Key, Query])
        else:
            raise RuntimeError(
                'unknown similarity function: {}'.format(self.simi_func))
        # Key Masking

        if self.key_masking:
            simi_func_result = Masking(0)(simi_func_result)

        if self.causality:
            pass

        soft_max_result = Dense(num_vector,
                                activation='softmax',
                                name=self_name + '_softmax')(simi_func_result)
        if self.query_masking:
            pass
        if self.drop_out and 0 < self.drop_out < 1:
            soft_max_result = Dropout(
                self.drop_out,
                name=self_name + '_dropout')(soft_max_result)

        outputs = merge([Value, soft_max_result],
                        name=self_name, mode='mul')
        return outputs
