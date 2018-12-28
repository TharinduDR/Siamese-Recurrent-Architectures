import numpy as np
from preprocessing.cleaning import text_to_word_list


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