import os
import config
import tensorflow as tf

import model
from utils import generate_image

generator = model.make_generator_model()

output_dir = os.path.join("experiments", config.EXPERIMENT_NAME)
#pra gerar uma imagem de um expperimento especifico, mude o nome do EXPERIMENT_NAME no config.py
checkpoint_dir = os.path.join(output_dir, "checkpoints") # pra caso queira gerar uma imagem com o modelo treinado
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

checkpoint = tf.train.Checkpoint(generator=generator)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint).expect_partial()
    noise = tf.random.normal([1, config.NOISE_DIM])
    generated_images = generate_image(generator, noise)
else:
    print(f"ERRO: Nenhum checkpoint encontrado no diretório: '{checkpoint_dir}'")

