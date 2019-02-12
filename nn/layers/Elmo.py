import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.engine import Layer
from keras.utils.generic_utils import to_list


class ElmoEmbeddingLayer(Layer):
    def __init__(self, input_length=None, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        self.input_length = input_length
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default',
                           )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        if self.input_length is None:
            return input_shape + (self.dimensions,)
        else:
            # input_length can be tuple if input is 3D or higher
            in_lens = to_list(self.input_length, allow_tuple=True)
            if len(in_lens) != len(input_shape) - 1:
                raise ValueError(
                    '"input_length" is %s, but received input has shape %s' %
                    (str(self.input_length), str(input_shape)))
            else:
                for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:])):
                    if s1 is not None and s2 is not None and s1 != s2:
                        raise ValueError(
                            '"input_length" is %s, but received input has shape %s' %
                            (str(self.input_length), str(input_shape)))
                    elif s1 is None:
                        in_lens[i] = s2
            return (input_shape[0],) + tuple(in_lens) + (self.dimensions,)
