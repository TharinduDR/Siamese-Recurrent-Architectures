import gensim
from gensim.scripts.glove2word2vec import glove2word2vec


def load_word2vec(path):
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    return word2vec


def load_glove(path):
    tmp_file = "/tmp/glove.840B.300d.w2v.txt"
    glove2word2vec(path, tmp_file)
    glove = gensim.models.KeyedVectors.load_word2vec_format(tmp_file)
    return glove


def load_fasttext(path):
    fasttext = gensim.models.KeyedVectors.load_word2vec_format(path, binary=False)
    return fasttext


def load_para(path):
    tmp_file = "/tmp/paragram_300_sl999.txt"
    glove2word2vec(path, tmp_file)
    para = gensim.models.KeyedVectors.load_word2vec_format(tmp_file, unicode_errors="ignore")
    return para


def load_arabic_word2vec(path):
    model = gensim.models.Word2Vec.load(path)
    return model
