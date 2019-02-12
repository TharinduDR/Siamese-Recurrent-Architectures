import keras.backend as K


def exponent_neg_manhattan_distance(left, right):
    """ Helper function for the similarity estimate of the LSTMs outputs"""
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


def exponent_neg_euclidean_distance(left, right):
    """ Helper function for the similarity estimate of the LSTMs outputs"""

    return K.sqrt(K.sum(K.square(left - right), axis=-1, keepdims=True))
