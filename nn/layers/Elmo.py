import tensorflow as tf
from keras import backend
import tensorflow_hub as hub

sess = tf.Session()
backend.set_session(sess)
elmo_model = hub.Module("https://alpha.tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


def ElmoEmbedding(x):
    y = elmo_model(tf.squeeze(x), signature="default", as_dict=True)["elmo"]
    return y

