# models.py

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (
    Dense, LSTM, GRU, Dropout, Input, Conv1D, MaxPooling1D,
    Flatten, TimeDistributed, Bidirectional, LayerNormalization,
    SimpleRNN, Attention, BatchNormalization, Activation,
    RepeatVector, Reshape, Embedding, MultiHeadAttention, Add, Concatenate, GlobalAveragePooling1D
)
import keras
import numpy as np
import losses  # To access loss functions and metrics if needed
from keras import backend as K

def get_model(config, number_of_features):
    model_type = config.get('model_type', 'LSTM').upper()
    model_params = config.get('model_params', {})
    n_steps_in = config['n_steps_in']
    n_steps_out = config['n_steps_out']

    if model_type == 'LSTM':
        model = build_lstm_model(n_steps_in, n_steps_out, number_of_features, model_params)
    elif model_type == 'GRU':
        model = build_gru_model(n_steps_in, n_steps_out, number_of_features, model_params)
    elif model_type == 'TRANSFORMER':
        # Retrieve num_time_features from model_params
        num_time_features = model_params.get('num_time_features')
        if num_time_features is None:
            raise ValueError("num_time_features must be specified in model_params for TRANSFORMER model.")
        model = build_transformer_model_with_temporal_embeddings(
            n_steps_in, n_steps_out, number_of_features, num_time_features, model_params
        )
    elif model_type == 'TCN':
        model = build_tcn_model(n_steps_in, n_steps_out, number_of_features, model_params)
    elif model_type == 'CNN_RNN':
        model = build_cnn_rnn_model(n_steps_in, n_steps_out, number_of_features, model_params)
    elif model_type == 'SEQ2SEQ':
        model = build_seq2seq_model(n_steps_in, n_steps_out, number_of_features, model_params)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return model

def build_lstm_model(n_steps_in, n_steps_out, number_of_features, model_params):
    units = model_params.get('units', 256)
    dropout = model_params.get('dropout', 0.4)
    activation = model_params.get('activation', 'tanh')

    model = Sequential()
    model.add(Input(shape=(n_steps_in, number_of_features)))
    model.add(LSTM(units, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(n_steps_out, dtype='float32'))
    return model

def build_gru_model(n_steps_in, n_steps_out, number_of_features, model_params):
    units = model_params.get('units', 256)
    dropout = model_params.get('dropout', 0.4)
    activation = model_params.get('activation', 'tanh')

    model = Sequential()
    model.add(Input(shape=(n_steps_in, number_of_features)))
    model.add(GRU(units, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(n_steps_out, dtype='float32'))
    return model

def positional_encoding(seq_len, d_model):
    """Generate a positional encoding matrix with sine and cosine functions."""
    angle_rads = np.arange(seq_len)[:, np.newaxis] / np.power(
        10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model)
    )

    # Apply sin to even indices (2i) and cos to odd indices (2i+1)
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(pos_encoding, dtype=tf.float32)  # Shape: (seq_len, d_model)

def temporal_encoding(time_features, time_feature_sizes):
    """
    Compute temporal embeddings from time features.

    Args:
        time_features: tensor of shape (batch_size, seq_len, num_time_features), containing the time features ti(k)
        time_feature_sizes: list or tensor of integers Ni, the total number of values for each time feature i

    Returns:
        temporal_embeddings: tensor of shape (batch_size, seq_len, num_time_features), containing the di(k)
    """
    time_feature_sizes = tf.constant(time_feature_sizes, dtype=tf.float32)
    time_feature_sizes = tf.reshape(time_feature_sizes, (1, 1, -1))  # Shape: (1, 1, num_time_features)
    di = (time_features / time_feature_sizes) - 0.5
    return di

def transformer_encoder(inputs, head_size, num_heads, ff_dim, hidden_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    res = LayerNormalization(epsilon=1e-6)(x)  # Add and Norm

    # Feed Forward Part
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=hidden_dim, kernel_size=1)(x)
    x = Add()([x, res])
    x = LayerNormalization(epsilon=1e-6)(x)  # Add and Norm
    return x

def build_transformer_model_with_temporal_embeddings(
    n_steps_in, n_steps_out, number_of_features, num_time_features, model_params
):
    num_heads = model_params.get('num_heads', 8)  # Tunable
    head_size = model_params.get('head_size', 16)  # Tunable
    ff_dim = model_params.get('ff_dim', 128)
    dropout_rate = model_params.get('dropout_rate', 0.3)
    num_transformer_blocks = model_params.get('num_transformer_blocks', 6)  # Tunable
    mlp_units = model_params.get('mlp_units', [128])  # Tunable
    mlp_dropout_rate = model_params.get('mlp_dropout_rate', 0.3)
    hidden_dim = model_params.get('hidden_dim', 128)  # Dimension of model hidden layer

    # Inputs
    inputs = keras.Input(shape=(n_steps_in, number_of_features))
    time_inputs = keras.Input(shape=(n_steps_in, num_time_features))

    # Generate positional encoding
    pos_encoding = positional_encoding(n_steps_in, hidden_dim)  # Shape: (n_steps_in, hidden_dim)
    pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

    # Compute temporal embeddings
    time_feature_sizes = model_params.get('time_feature_sizes', [2022, 12, 31, 7, 366])
    temporal_embeddings = temporal_encoding(time_inputs, time_feature_sizes)  # Shape: (batch_size, n_steps_in, num_time_features)

    # Project inputs and temporal embeddings to hidden_dim
    EX = Dense(hidden_dim)(inputs)  # Shape: (batch_size, n_steps_in, hidden_dim)
    ET = Dense(hidden_dim)(temporal_embeddings)  # Shape: (batch_size, n_steps_in, hidden_dim)

    # Sum them with positional encoding (broadcasting will handle the addition)
    x = EX + ET + pos_encoding  # Broadcasting adds pos_encoding to each example in the batch

    # Transformer Blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, hidden_dim, dropout_rate)

    # Pooling and Output Layers
    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout_rate)(x)
    outputs = Dense(n_steps_out)(x)

    # Model with two inputs: features and time features
    return keras.Model(inputs=[inputs, time_inputs], outputs=outputs)

def build_tcn_model(n_steps_in, n_steps_out, number_of_features, model_params):
    # You need to install keras-tcn: pip install keras-tcn
    from tcn import TCN

    units = model_params.get('units', 64)
    dropout = model_params.get('dropout', 0.2)
    kernel_size = model_params.get('kernel_size', 3)
    dilations = model_params.get('dilations', [1, 2, 4, 8])
    nb_stacks = model_params.get('nb_stacks', 1)
    use_skip_connections = model_params.get('use_skip_connections', True)

    inputs = Input(shape=(n_steps_in, number_of_features))
    x = TCN(nb_filters=units,
            kernel_size=kernel_size,
            dilations=dilations,
            nb_stacks=nb_stacks,
            use_skip_connections=use_skip_connections,
            dropout_rate=dropout)(inputs)
    outputs = Dense(n_steps_out, dtype='float32')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_cnn_rnn_model(n_steps_in, n_steps_out, number_of_features, model_params):
    conv_filters = model_params.get('conv_filters', 64)
    kernel_size = model_params.get('kernel_size', 3)
    pool_size = model_params.get('pool_size', 2)
    rnn_units = model_params.get('rnn_units', 128)
    dropout = model_params.get('dropout', 0.2)

    inputs = Input(shape=(n_steps_in, number_of_features))
    x = Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = LSTM(rnn_units)(x)
    x = Dropout(dropout)(x)
    outputs = Dense(n_steps_out, dtype='float32')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def build_seq2seq_model(n_steps_in, n_steps_out, number_of_features, model_params):
    encoder_units = model_params.get('encoder_units', 128)
    decoder_units = model_params.get('decoder_units', 128)
    dropout = model_params.get('dropout', 0.2)
    activation = model_params.get('activation', 'tanh')

    # Encoder
    encoder_inputs = Input(shape=(n_steps_in, number_of_features))
    encoder = LSTM(encoder_units, activation=activation, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = RepeatVector(n_steps_out)(encoder_outputs)
    decoder_lstm = LSTM(decoder_units, activation=activation, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = TimeDistributed(Dense(1))(decoder_outputs)
    decoder_outputs = tf.keras.layers.Flatten()(decoder_outputs)
    outputs = Dense(n_steps_out, dtype='float32')(decoder_outputs)

    model = Model(inputs=encoder_inputs, outputs=outputs)
    return model
