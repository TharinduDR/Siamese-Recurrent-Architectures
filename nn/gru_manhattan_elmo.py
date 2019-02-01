import itertools

from keras import Input, Model
from keras.layers import GRU, Lambda
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from nn.layers.Elmo import ElmoEmbeddingLayer
from nn.util.distances import exponent_neg_manhattan_distance


def run_gru_elmo_benchmark(train_df, test_df, sent_cols, sim_col, validation_portion=0.1, n_hidden=100,
                           embedding_dim=300,
                           batch_size=64, n_epoch=500, optimizer=None, save_weights=None, load_weights=None,
                           max_seq_length=None, model=None):
    datasets = [train_df, test_df]

    if max_seq_length is None:
        max_seq_length = max(train_df.sent_1.map(lambda x: len(x)).max(),
                             train_df.sent_2.map(lambda x: len(x)).max(),
                             test_df.sent_1.map(lambda x: len(x)).max(),
                             test_df.sent_2.map(lambda x: len(x)).max())

    # Split to train validation
    validation_size = int(validation_portion * len(train_df))
    training_size = len(train_df) - validation_size

    X = train_df[sent_cols]
    Y = train_df[sim_col]

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

    # Split to dicts
    X_train = {'left': X_train.sent_1, 'right': X_train.sent_2}
    X_validation = {'left': X_validation.sent_1, 'right': X_validation.sent_2}
    X_test = {'left': test_df.sent_1, 'right': test_df.sent_2}

    # Convert labels to their numpy representations
    Y_train = Y_train.values
    Y_validation = Y_validation.values

    # Zero padding
    # for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    #     dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

    # # The visible layer
    # left_input = Input(shape=(max_seq_length,), dtype='int32')
    # right_input = Input(shape=(max_seq_length,), dtype='int32')

    left_input_text = Input(shape=(1,), dtype="string")
    right_input_text = Input(shape=(1,), dtype="string")

    # embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length,
    #                             trainable=False)

    embedding_layer = ElmoEmbeddingLayer(embedding_dim, input_length=1)

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input_text)
    encoded_right = embedding_layer(right_input_text)

    # Since this is a siamese network, both sides share the same LSTM
    shared_gru = GRU(n_hidden, name='gru')
    # shared_gru = GRU(n_hidden, name='gru', return_sequences=True,
    #                  kernel_initializer=glorot_normal(seed=12300),
    #                  recurrent_initializer=orthogonal(gain=1.0, seed=10000))

    left_output = shared_gru(encoded_left)
    right_output = shared_gru(encoded_right)

    # Calculates the distance as defined by the MaLSTM model
    magru_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                            output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

    # Pack it all up into a model
    magru = Model([left_input_text, right_input_text], [magru_distance])

    optimizer = optimizer

    if load_weights is not None:
        magru.load_weights(load_weights, by_name=True)

    magru.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    magru_trained = magru.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
                              verbose=0,
                              validation_data=([X_validation['left'], X_validation['right']], Y_validation))

    if save_weights is not None:
        magru.save_weights(save_weights)

    for dataset, side in itertools.product([X_test], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

    sims = magru.predict([X_test['left'], X_test['right']], batch_size=batch_size)
    formatted_sims = []

    for sim in sims:
        formatted_sims.append(sim[0])

    return formatted_sims, magru_trained
