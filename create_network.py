import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, BatchNormalization, Dropout, Activation,
                                     Convolution2D, Dense, MaxPooling2D, Concatenate, Flatten,
                                     UpSampling2D)

from tensorflow.keras.models import Model, Sequential
import pandas as pd

# Sequential model just an encoder/decoder without the umap style connections
def create_sequential(input_shape,
                      nclasses,
                      conv_layers,
                      lambda_regularization=None,
                      lrate=0.0001):

    if lambda_regularization is not None:
        lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)

    # TODO: Fix shape being fed in
    input_tensor = Input(shape=input_shape, name="input")


    for i in range(len(conv_layers)):
        layer = Convolution2D(filters=conv_layers[i]['filters'], kernel_size=conv_layers[i]['kernel_size']
                       , activation='relu', strides=1, padding='same')(layer)
    """
    layer = MaxPooling2D(pool_size=conv_layers[i]['pool_size'], strides=conv_layers[i]['strides'], padding="same",
                      data_format="channels_last",)(layer)
    """

    # Your network output should be shape (examples, rows, cols, class),
    # where the sum of all class outputs for a single pixel is 1
    # (i.e., we are using a softmax across the last dimension of your output).

    # Output is
    output_tensor = Convolution2D(nclasses, kernel_size=(1,1),
                                  activation='softmax',  padding='same')(layer)

    # The optimizer determines how the gradient descent is to be done
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999,
                                   epsilon=None, decay=0.0, amsgrad=False)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['sparse_categorical_accuracy'])


    return model

def create_Unet(input_shape,
                      nclasses,
                      conv_layers,
                      lambda_regularization=None,
                      lrate=0.0001):
    if lambda_regularization is not None:
        lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)

    # TODO: Fix shape being fed in
    input_tensor = Input(shape=input_shape, name="input")

    # 128x128
    layer = Convolution2D(filters=conv_layers[i]['filters'], kernel_size=conv_layers[i]['kernel_size']
                          , activation='relu', strides=1, padding='same')(layer)

    # 64x64
    layer = MaxPooling2D(pool_size=conv_layers[i]['pool_size'], strides=conv_layers[i]['strides'], padding="same",
                      data_format="channels_last",)(layer)

    layer = Convolution2D(filters=conv_layers[i]['filters'], kernel_size=conv_layers[i]['kernel_size']
                          , activation='relu', strides=1, padding='same')(layer)

    layer = Convolution2D(filters=conv_layers[i]['filters'], kernel_size=conv_layers[i]['kernel_size']
                          , activation='relu', strides=1, padding='same')(layer)

    # 32x32
    layer = MaxPooling2D(pool_size=conv_layers[i]['pool_size'], strides=conv_layers[i]['strides'], padding="same",
                         data_format="channels_last", )(layer)

    layer = Convolution2D(filters=conv_layers[i]['filters'], kernel_size=conv_layers[i]['kernel_size']
                          , activation='relu', strides=1, padding='same')(layer)

    layer = Convolution2D(filters=conv_layers[i]['filters'], kernel_size=conv_layers[i]['kernel_size']
                          , activation='relu', strides=1, padding='same')(layer)
    #######
    layer = UpSampling2D(size=2)(layer)

    #64x64
    layer = Convolution2D(filters=conv_layers[i]['filters'], kernel_size=conv_layers[i]['kernel_size']
                          , activation='relu', strides=1, padding='same')(layer)

    layer = Convolution2D(filters=conv_layers[i]['filters'], kernel_size=conv_layers[i]['kernel_size']
                          , activation='relu', strides=1, padding='same')(layer)


    layer = UpSampling2D(size=2)(layer)
    # 128x128
    layer = Convolution2D(filters=conv_layers[i]['filters'], kernel_size=conv_layers[i]['kernel_size']
                          , activation='relu', strides=1, padding='same')(layer)

    layer = Convolution2D(filters=conv_layers[i]['filters'], kernel_size=conv_layers[i]['kernel_size']
                          , activation='relu', strides=1, padding='same')(layer)

    # Output is
    output_tensor = Convolution2D(nclasses, kernel_size=(1, 1),
                                  activation='softmax', padding='same')(layer)

    # The optimizer determines how the gradient descent is to be done
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999,
                                   epsilon=None, decay=0.0, amsgrad=False)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['sparse_categorical_accuracy'])

    return model

# An attempt at declaring the order of layers at the command line.
# Takes in the layer type and the parameters associated with the layer, and outputs the layer
#TODO: Figure out how to ensure the parameters match the given layer.
# Maybe a stack of tuples?

def makeLayer(type, parameters):
    if type == 'LSTM':
        layer = (LSTM(parameters['n_neurons'],
                      activation=parameters['tanh'],
                      use_bias=True,
                      return_sequences=False,  # Produce entire sequence of outputs
                      kernel_initializer='random_uniform',
                      kernel_regularizer=lambda_regularization,
                      unroll=False))

    if type == 'RNN':
        layer = (SimpleRNN(parameters['n_neurons'],
                           activation=parameters['tanh'],
                           use_bias=True,
                           return_sequences=False,  # Produce entire sequence of outputs
                           kernel_initializer='random_uniform',
                           kernel_regularizer=lambda_regularization,
                           unroll=False))

    if type == 'CNN':
        layer = (Conv1D(filters=64, kernel_size=25, activation='relu'))

    # Not used in this assignment, but can be potentially recycled
    if type == 'MP':
        layer = MaxPooling2D(pool_size=(3,3),
                              strides=(1,1),
                              padding='same')

    if type == 'Dense':
        layer = (Dense(parameters['n_neurons'],
                           activation=parameters['tanh'],
                           use_bias=True,
                           return_sequences=False,  # Produce entire sequence of outputs
                           kernel_initializer='random_uniform',
                           kernel_regularizer=lambda_regularization,
                           unroll=False))


    return layer

