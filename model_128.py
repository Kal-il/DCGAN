import tensorflow as tf
from tensorflow.keras import layers
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import config



# Inicializador de pesos
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=config.WEIGHT_INIT_STDDEV)
#----------------------------------------------------------------------------

def make_generator_model():
    """Cria o modelo do Gerador."""
    model = tf.keras.Sequential()
    model.add(
        tf.keras.Input(shape=(config.NOISE_DIM,))
    )
    model.add(
        layers.Dense(
            4*4*1024,
            use_bias=False,
            kernel_initializer=weight_initializer
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((4, 4, 1024)))

    model.add(
        layers.Conv2DTranspose(
            512,
            (5, 5),
            strides=(2, 2),
            padding='same',
            use_bias=False,
            kernel_initializer=weight_initializer
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU()) # -> saida 8x8x512

    model.add(
        layers.Conv2DTranspose(
            256,
            (5, 5),
            strides=(2, 2),
            padding='same',
            use_bias=False,
            kernel_initializer=weight_initializer
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU()) # -> saida 16x16x256

    model.add(
        layers.Conv2DTranspose(
            128,
            (5, 5),
            strides=(2, 2),
            padding='same',
            use_bias=False,
            kernel_initializer=weight_initializer
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU()) # -> saida 32x32x128

    model.add(
        layers.Conv2DTranspose(
            64,
            (5, 5),
            strides=(2, 2),
            padding='same',
            use_bias=False,
            kernel_initializer=weight_initializer
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU()) # -> saida 64x64x64

    model.add(
        layers.Conv2DTranspose(
            3,
            (5, 5),
            strides=(2, 2),
            padding='same',
            use_bias=False,
            activation='tanh',
            kernel_initializer=weight_initializer
        )
    ) # -> saida 128x128x3
    # assert model.output_shape == (None, 64, 64, 3)

    return model

#----------------------------------------------------------------------------

def make_discriminator_model():
    """Cria o modelo do Discriminador."""
    model = tf.keras.Sequential()
    # Camada 1 64x64x64
    model.add(
        tf.keras.Input(shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS))
    )
    model.add(
        layers.Conv2D(
            64,
            (5, 5),
            strides=(2, 2),
            padding='same',
            kernel_initializer=weight_initializer
        )
    )
    model.add(layers.LeakyReLU(alpha=config.LEAKY_RELU_SLOPE)) # slope of the leak was set to 0.2
    model.add(layers.Dropout(0.3))
    # Camada 2 32x32x128
    model.add(
        layers.Conv2D(
            128,
            (5, 5),
            strides=(2, 2),
            padding='same',
            kernel_initializer=weight_initializer
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=config.LEAKY_RELU_SLOPE))
    model.add(layers.Dropout(0.3))
    # Camada 3 reduz 16x16x256
    model.add(
        layers.Conv2D(
            256,
            (5, 5),
            strides=(2, 2),
            padding='same',
            kernel_initializer=weight_initializer
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=config.LEAKY_RELU_SLOPE))
    model.add(layers.Dropout(0.3))
    # Camada 4 reduz pra 8x8x512
    model.add(
        layers.Conv2D(
            512,
            (5, 5),
            strides=(2, 2),
            padding='same',
            kernel_initializer=weight_initializer
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=config.LEAKY_RELU_SLOPE))
    model.add(layers.Dropout(0.3))

    # Camada 5 reduz pra 4x4x1024
    model.add(
        layers.Conv2D(
            1024,
            (5, 5),
            strides=(2, 2),
            padding='same',
            kernel_initializer=weight_initializer
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=config.LEAKY_RELU_SLOPE))
    model.add(layers.Dropout(0.3))

    # Camada final
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            1,
            kernel_initializer=weight_initializer
        )
    )

    return model