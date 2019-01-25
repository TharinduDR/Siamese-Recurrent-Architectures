import keras.backend as K


def exponent_neg_manhattan_distance(left, right):
    """ Helper function for the similarity estimate of the LSTMs outputs"""
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

def exponent_neg_euclidian_distance(left, right):
    """ Helper function for the similarity estimate of the LSTMs outputs"""
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))
