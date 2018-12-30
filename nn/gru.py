import itertools

from keras import Input, Model
from keras.layers import Embedding, GRU, Lambda
from keras.optimizers import Adadelta
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from nn.util.distances import exponent_neg_manhattan_distance
from preprocessing.embeddings import prepare_embeddings


def run_gru_benchmark(train_df, test_df, questions_cols, validation_portion, n_hidden, embedding_dim, gradient_clipping_norm, batch_size, n_epoch, model):

    datasets = [train_df, test_df]
    embeddings = prepare_embeddings(datasets=datasets, question_cols=questions_cols, model=model)

    max_seq_length = max(train_df.sent_1.map(lambda x: len(x)).max(),
                         train_df.sent_2.map(lambda x: len(x)).max(),
                         test_df.sent_1.map(lambda x: len(x)).max(),
                         test_df.sent_2.map(lambda x: len(x)).max())

    # Split to train validation
    validation_size = validation_portion * len(train_df)
    training_size = len(train_df) - validation_size

    X = train_df[questions_cols]
    Y = train_df['sim']

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

    # Split to dicts
    X_train = {'left': X_train.sent_1, 'right': X_train.sent_2}
    X_validation = {'left': X_validation.sent_1, 'right': X_validation.sent_2}
    X_test = {'left': test_df.sent_1, 'right': test_df.sent_2}

    # Convert labels to their numpy representations
    Y_train = Y_train.values
    Y_validation = Y_validation.values

    # Zero padding
    for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length,
                                trainable=False)

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # Since this is a siamese network, both sides share the same LSTM
    shared_lstm = GRU(n_hidden, name='gru')

    left_output = shared_lstm(encoded_left)
    right_output = shared_lstm(encoded_right)

    # Calculates the distance as defined by the MaLSTM model
    malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                             output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

    # Pack it all up into a model
    malstm = Model([left_input, right_input], [malstm_distance])

    # Adadelta optimizer, with gradient clipping by norm
    optimizer = Adadelta(clipnorm=gradient_clipping_norm)
    # optimizer = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    #malstm.load_weights('gru_weights.h5', by_name=True)
    malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch,
                                validation_data=([X_validation['left'], X_validation['right']], Y_validation))



