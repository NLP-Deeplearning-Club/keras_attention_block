from keras import backend as K


class MergfuncMixin:
    def batch_dot_merg(self, x, y):
        return K.batch_dot(x, y)

    def batch_mul_merg(self, x, y):
        x_t = K.permute_dimensions(x, (0, 2, 1))
        result = x_t * y
        return result

    def batch_add_merg(self, x, y):
        x_t = K.permute_dimensions(x, (0, 2, 1))
        return x_t + y
