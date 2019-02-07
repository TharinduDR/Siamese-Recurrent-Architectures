from enum import Enum

from flair.data import Sentence


class Language(Enum):
    ENGLISH = 'english'
    ARABIC = 'arabic'
    SPANISH = "Spanish"


from flair.embeddings import BertEmbeddings

# init embedding
embedding = BertEmbeddings("bert-base-uncased")

# create a sentence
sentence = Sentence('The grass is green .')
