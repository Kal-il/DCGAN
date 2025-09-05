import os
import config
import tensorflow as tf

import model
import utils
from utils import generate_image

generator = model.make_generator_model()
checkpoint = tf.train.Checkpoint(generator=generator)

# caminho_do_checkpoint = "experiments/test_330_epochs_and_4x4_base_64x64/checkpoints/ckpt-30"
caminho_do_checkpoint = "experiments/test_2_1000_epochs_4x4_base_64x64_gen_lr_adjusted_to_0_00015/checkpoints/ckpt-96"
# caminho_do_checkpoint = "experiments/test_1000_epochs_4x4_base_64x64_gen_lr_adjusted/checkpoints/ckpt-77"

checkpoint.restore(caminho_do_checkpoint).expect_partial()

print(f"Pesos do checkpoint '{caminho_do_checkpoint}' restaurados com sucesso no gerador.")

noise = tf.random.normal([1, config.NOISE_DIM]) # Gera 1 imagem

generated_image = generator(noise, training=False)

utils.generate_image(generator, tf.random.normal([16, config.NOISE_DIM]))

# utils.create_evolution_gif(image_folder='experiments/test_1000_epochs_4x4_base_64x64_gen_lr_adjusted/images', output_path="evolucao_treinamento.gif")

