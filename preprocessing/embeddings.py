from enum import Enum

import numpy as np

from preprocessing.cleaning import text_to_word_list, text_to_arabic_word_list


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

                    # # Check for unwanted words
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

    embedding_dim = model.vector_size
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabulary.items():
        if word in model.vocab:
            embeddings[index] = model.word_vec(word)

    return embeddings, embedding_dim


def prepare_embeddings_elmo(model, datasets, question_cols):
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

    embedding_dim = model.vector_size
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabulary.items():
        if word in model.vocab:
            embeddings[index] = model.word_vec(word)

    return embeddings, embedding_dim, inverse_vocabulary


def prepare_spanish_embeddings(model, datasets, question_cols):
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

    embedding_dim = model.vector_size
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabulary.items():
        if word in model.vocab:
            embeddings[index] = model.word_vec(word)

    return embeddings, embedding_dim


def prepare_arabic_embeddings(model, datasets, question_cols):
    vocabulary = dict()
    inverse_vocabulary = ['<unk>']
    questions_cols = question_cols

    # Iterate over the questions only of both training and test datasets
    for dataset in datasets:
        for index, row in dataset.iterrows():

            # Iterate through the text of both questions of the row
            for question in questions_cols:

                q2n = []  # q2n -> question numbers representation
                for word in text_to_arabic_word_list(row[question]):

                    # Check for unwanted words
                    if word not in model.wv.vocab:
                        continue

                    if word not in vocabulary:
                        vocabulary[word] = len(inverse_vocabulary)
                        q2n.append(len(inverse_vocabulary))
                        inverse_vocabulary.append(word)
                    else:
                        q2n.append(vocabulary[word])

                # Replace questions as word to question as number representationindex, question, q2n
                dataset.set_value(index, question, q2n)

    embedding_dim = model.vector_size
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabulary.items():
        if word in model.wv.vocab:
            embeddings[index] = model.wv[word]

    return embeddings, embedding_dim


def merge_embeddings(models, datasets, question_cols, merge_operation):
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

                # Replace questions as word to question as number representation index, question, q2n
                dataset.set_value(index, question, q2n)

    embedding_dim = 0

    if merge_operation is MergeOperations.CONCATENATE:
        for model in models:
            embedding_dim = embedding_dim + model.vector_size

    if merge_operation is MergeOperations.AVERAGE:
        embedding_dim = models[0].vector_size

    embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix

    if merge_operation is MergeOperations.CONCATENATE:
        for word, index in vocabulary.items():
            counter = 0
            temp_embedding = []
            for model in models:
                if word in model.vocab:
                    temp_embedding = temp_embedding + model.word_vec(word).tolist()

                counter += 1
            embeddings[index] = np.asarray(temp_embedding, dtype=np.float32)
        return embeddings, embedding_dim

    if merge_operation is MergeOperations.AVERAGE:
        for word, index in vocabulary.items():
            counter = 0
            temp_embedding = []
            for model in models:
                if word in model.vocab:
                    temp_embedding.append(model.word_vec(word).tolist())

                counter += 1
            embeddings[index] = np.asarray(np.mean(temp_embedding, axis=0), dtype=np.float32)
        return embeddings, embedding_dim


class MergeOperations(Enum):
    AVERAGE = 'average'
    CONCATENATE = 'concatenate'
