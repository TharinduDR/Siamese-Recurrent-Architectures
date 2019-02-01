import numpy as np

from preprocessing.cleaning import text_to_word_list
from utility.commons.decorators import deprecated


@deprecated("Use method merge embeddings instead")
def prepare_embeddings(model, datasets, question_cols):
    vocabulary = dict()
    inverse_vocabulary = ['<unk>']
    questions_cols = question_cols

    # Iterate over the questions only of both training and test datasets
    for dataset in datasets:
        for index, row in dataset.iterrows():

            # Iterate through the text of both questions of the row
            for question in questions_cols:

                q2n = []  # q2n -> question numbers representation
                for word in text_to_word_list(row[question]):

                    # Check for unwanted words
                    if word not in model.vocab:
                        continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        q2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        q2n.append(vocabulary[word])

                # Replace questions as word to question as number representationindex, question, q2n
                dataset.set_value(index, question, q2n)

    embedding_dim = 300
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabulary.items():
        if word in model.vocab:
            embeddings[index] = model.word_vec(word)

    return embeddings


def merge_embeddings(models, datasets, question_cols):
    vocabulary = dict()
    inverse_vocabulary = ['<unk>']
    questions_cols = question_cols

    # Iterate over the questions only of both training and test datasets
    for dataset in datasets:
        for index, row in dataset.iterrows():

            # Iterate through the text of both questions of the row
            for question in questions_cols:

                q2n = []  # q2n -> question numbers representation
                for word in text_to_word_list(row[question]):

                    word_exists = True
                    for model in models:
                        if word not in model.vocab:
                            word_exists = False
                    if not word_exists:
                        continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        q2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        q2n.append(vocabulary[word])

                # Replace questions as word to question as number representationindex, question, q2n
                dataset.set_value(index, question, q2n)

    embedding_dim = 0
    for model in models:
        embedding_dim = embedding_dim + model.vector_size
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabulary.items():

        counter = 0
        for model in models:
            if word in model.vocab:
                if counter == 0:
                    embeddings[index] = model.word_vec(word)
                else:
                    embeddings[index] = np.concatenate((embeddings[index], model.word_vec(word)), axis=0)
            counter += 1
    return embeddings, embedding_dim
