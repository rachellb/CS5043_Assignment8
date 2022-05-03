import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, BatchNormalization, Dropout, Activation,
                                     Convolution2D, Dense, MaxPooling2D, Concatenate, Flatten,
                                     UpSampling2D)

from tensorflow.keras.models import Model, Sequential
import pandas as pd

# Sequential model just an encoder/decoder without the umap style connections
def create_sequential(input_shape,
                      nclasses,
                      filters,
                      pool_size,
                      kernel_size,
                      lambda_regularization=None,
                      lrate=0.0001):

    if lambda_regularization is not None:
        lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)

    # TODO: Fix shape being fed in
    input_tensor = Input(shape=input_shape, name="input")
    layer = input_tensor

    # Essentially stays the same shape I think
    for i in range(len(filters)):
        layer = Convolution2D(filters=filters[i], kernel_size=kernel_size, # TODO: figure out if kernel size automatically fills in second argument
                              activation='relu', strides=1, padding='same')(layer)

    # Your network output should be shape (examples, rows, cols, class),
    # where the sum of all class outputs for a single pixel is 1
    # (i.e., we are using a softmax across the last dimension of your output).

    # Output is probability distribution over classes (per pixel)
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
                filters,
                pool_size,
                kernel_size,
                lambda_regularization=None,
                lrate=0.0001):


    if lambda_regularization is not None:
        lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)

    # Create a stack for skip connections
    tensor_stack = []

    input_tensor = Input(shape=input_shape, name="input")

    layer = input_tensor

    layer = Convolution2D(filters=filters[0], kernel_size=kernel_size
                          , activation='relu', strides=1, padding='same')(layer)
    layer = Convolution2D(filters=filters[1], kernel_size=kernel_size
                          , activation='relu', strides=1, padding='same')(layer)

    #Step down
    for i in range(2, len(filters)//2): #Start counting at 2 and go up ot length of filters divided by 2
        # Save the previous layer for skip connections
        tensor_stack.append(layer)

        # Cut the resolution by half (if pool size=2)
        layer = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding="same",
                             data_format="channels_last", )(layer)

        layer = Convolution2D(filters=filters[i*2], kernel_size=kernel_size
                              , activation='relu', strides=1, padding='same')(layer)
        layer = Convolution2D(filters=filters[i*2+1], kernel_size=kernel_size
                              , activation='relu', strides=1, padding='same')(layer)

    # Step up
    for i in reverse(range(2, len(filters)//2)):

        # Increase the resolution up a level (by 2 if pool=2)
        layer = UpSampling2D(size=pool_size)(layer)

        # Add the skipped connection back in
        layer = Concatenate([layer, tensor_stack.pop()])
        layer = Convolution2D(filters=filters[i*2], kernel_size=kernel_size,
                              activation='relu', strides=1, padding='same')(layer)

        layer = Convolution2D(filters=filters[i*2], kernel_size=kernel_size,
                              activation='relu', strides=1, padding='same')(layer)

    # For symmetry of beginning
    layer = Convolution2D(filters=filters[0], kernel_size=kernel_size,
                          activation='relu', strides=1, padding='same')(layer)
    layer = Convolution2D(filters=filters[1], kernel_size=kernel_size,
                          activation='relu', strides=1, padding='same')(layer)


    output_tensor = Convolution2D(nclasses, kernel_size=(1, 1),
                                  activation='softmax', padding='same')(layer)

    # The optimizer determines how the gradient descent is to be done
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999,
                                   epsilon=None, decay=0.0, amsgrad=False)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['sparse_categorical_accuracy'])

    return model




