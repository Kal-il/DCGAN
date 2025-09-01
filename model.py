import tensorflow as tf
from tensorflow.keras import layers

def make_generator_model():
    """Cria o modelo do Gerador."""
    model = tf.keras.Sequential()
    # Lembre-se de ajustar as dimensões aqui para o seu dataset de flores!
    # Ex: Mudar de 7*7*256 para 4*4*256 se for gerar 64x64
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Lembre-se de ajustar o primeiro parâmetro (filtros) para 3 para imagens coloridas!
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    # assert model.output_shape == (None, 64, 64, 3)

    return model

def make_discriminator_model():
    """Cria o modelo do Discriminador."""
    model = tf.keras.Sequential()
    # Lembre-se de ajustar o input_shape para as dimensões da sua imagem de flor!
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model