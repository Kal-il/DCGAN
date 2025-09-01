import tensorflow as tf
import time
import os
from IPython import display

# Importa nossos módulos organizados
import config
import dataset
import model
import utils

# --- 1. CARREGAR OS DADOS ---
print("Carregando e preparando o dataset de flores...")
train_dataset = dataset.load_and_prepare_dataset()
print("Dataset pronto.")

# --- 2. CONSTRUIR OS MODELOS ---
print("Construindo o Gerador e o Discriminador...")
generator = model.make_generator_model()
discriminator = model.make_discriminator_model()
print("Modelos construídos.")

# --- 3. FUNÇÕES DE PERDA E OTIMIZADORES ---
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# --- 4. CHECKPOINTS (SALVAR O PROGRESSO) ---
checkpoint_prefix = os.path.join(config.CHECKPOINT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# --- 5. LÓGICA DE TREINAMENTO (1 PASSO) ---
@tf.function
def train_step(images):
    noise = tf.random.normal([config.BATCH_SIZE, config.NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# --- 6. LOOP DE TREINAMENTO COMPLETO ---
def train(dataset, epochs):
  # Cria a semente de ruído fixa para acompanhar o progresso
  seed = tf.random.normal([config.NUM_EXAMPLES_TO_GENERATE, config.NOISE_DIM])

  for epoch in range(epochs):
    start = time.time()
    for image_batch in dataset:
      train_step(image_batch)

    display.clear_output(wait=True)
    utils.generate_and_save_images(generator, epoch + 1, seed)

    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    print ('Tempo para a época {} é {} seg'.format(epoch + 1, time.time()-start))

  display.clear_output(wait=True)
  utils.generate_and_save_images(generator, epochs, seed)

# --- 7. INICIAR O TREINAMENTO ---
print("\nIniciando o treinamento...")
train(train_dataset, config.EPOCHS)
print("Treinamento concluído.")