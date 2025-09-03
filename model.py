import tensorflow as tf
from tensorflow.keras import layers

import config

# Inicializador de pesos
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=config.WEIGHT_INIT_STDDEV)
#----------------------------------------------------------------------------

def make_generator_model():
    """Cria o modelo do Gerador."""
    model = tf.keras.Sequential()
    model.add(
        layers.Dense(
            8*8*256,
            use_bias=False,
            input_shape=(100,),
            kernel_initializer=weight_initializer
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((8, 8, 256)))

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
    model.add(layers.ReLU())

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
    model.add(layers.ReLU())

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
    )
    # assert model.output_shape == (None, 64, 64, 3)

    return model

#----------------------------------------------------------------------------

def make_discriminator_model():
    """Cria o modelo do Discriminador."""
    model = tf.keras.Sequential()
    # Lembre-se de ajustar o input_shape para as dimensões da sua imagem de flor!
    model.add(
        layers.Conv2D(
            64,
            (5, 5),
            strides=(2, 2),
            padding='same',
            input_shape=[64, 64, 3],
            kernel_initializer=weight_initializer
        )
    )
    model.add(layers.LeakyReLU(alpha=config.LEAKY_RELU_ALPHA)) # slope of the leak was set to 0.2
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv2D(
            128,
            (5, 5),
            strides=(2, 2),
            padding='same',
            kernel_initializer=weight_initializer
        )
    )
    model.add(layers.LeakyReLU(alpha=config.LEAKY_RELU_ALPHA))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            1,
            kernel_initializer=weight_initializer
        )
    )

    return model