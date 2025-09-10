import os
import config
import tensorflow as tf

import model_128 as model
import utils
from utils import generate_image
import datetime


generator = model.make_generator_model()
checkpoint = tf.train.Checkpoint(generator=generator)

# caminho_do_checkpoint = "experiments/test_500_epochs/checkpoints/ckpt-43"
caminho_do_checkpoint = "experiments/test_colab_artigo/ckpt-60"

checkpoint.restore(caminho_do_checkpoint).expect_partial()

print(f"Pesos do checkpoint '{caminho_do_checkpoint}' restaurados com sucesso no gerador.")

noise = tf.random.normal([1, config.NOISE_DIM]) # Gera 1 imagem

# generated_image = generator(noise, training=False)

flag = 1

while flag != -1:
    print("\n\ngerar uma unicca imagem (1) ou um batch de imagens (0) ?\n\n")
    option = input()
    if option == 1 or option == '1':
        utils.generate_image(generator, tf.random.normal([16, config.NOISE_DIM]), f"image{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    elif option == 0 or option == '0':
        utils.generate_and_save_in_batches(
            model=generator, 
            num_images=6000, # 1638
            output_dir="fid_batches/test_colab_artigo_440",
            batch_size=32 # Um lote de 64 por vez é seguro
        )
        # utils.create_evolution_gif(image_folder='experiments/test_1000_epochs_4x4_base_64x64_gen_lr_adjusted/images', output_path="evolucao_treinamento.gif")
    else:
        flag = -1

