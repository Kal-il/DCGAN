import tensorflow as tf

import config


def load_and_prepare_dataset():
    """Carrega as imagens do diretório, redimensiona e normaliza."""
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=config.DATA_DIR,
        labels='inferred',
        label_mode=None,
        image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        batch_size=config.BATCH_SIZE,
        validation_split=None
    )

    # funcao de normalizacao
    def normalize(image):
        # converte pra float32 e normaliza para [-1, 1]
        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5
        return image

    train_dataset = train_dataset.map(normalize) # Aplica a função de normalização a cada imagem no dataset

    return train_dataset.shuffle(config.BUFFER_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    # .shuffle() ➔ Garante que o modelo aprenda os padrões corretos (generalização).
    # .prefetch() ➔ Garante que o treinamento seja rápido e estável.
    # O prefetch carrega os próximos lotes enquanto o modelo está treinando no lote atual, reduzindo o tempo de espera.
