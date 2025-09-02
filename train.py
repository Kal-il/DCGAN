import tensorflow as tf
import time
import os
from IPython import display
import config
import dataset
import model
import utils

output_dir = os.path.join("experiments", config.EXPERIMENT_NAME)
checkpoint_dir = os.path.join(output_dir, "checkpoints")
log_file_path = os.path.join(output_dir, "training_log.csv")
os.makedirs(checkpoint_dir, exist_ok=True) # cria diretorio se nao existir


#carregar os dados
print("Carregando e preparando o dataset de flores...")
train_dataset = dataset.load_and_prepare_dataset()
print("Dataset pronto.")

# construir os modelos
print("Construindo o Gerador e o Discriminador...")
generator = model.make_generator_model()
discriminator = model.make_discriminator_model()
print("Modelos construídos.")

# funcoes de perda e otimizacoes
#---------------------------------------------------------------------------------

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
#---------------------------------------------------------------------------------

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# checkpoints
checkpoint_prefix = os.path.join(config.CHECKPOINT_DIR, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# treinamento, passo 1
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
    return gen_loss, disc_loss

# loop de treinamento
def train(dataset, epochs):
  # Cria a semente de ruído fixa para acompanhar o progresso
  seed = tf.random.normal([config.NUM_EXAMPLES_TO_GENERATE, config.NOISE_DIM])
  with open(log_file_path, "w") as log_file:
    log_file.write("epoch,gen_loss,disc_loss,time_sec\n")

    for epoch in range(epochs):
      start = time.time()
      
      epoch_gen_loss = []
      epoch_disc_loss = []
      
      
      for image_batch in dataset:
        gen_loss, disc_loss = train_step(image_batch)
        epoch_gen_loss.append(gen_loss)
        epoch_disc_loss.append(disc_loss)
      
      avg_gen_loss = tf.reduce_mean(epoch_gen_loss).numpy()
      avg_disc_loss = tf.reduce_mean(epoch_disc_loss).numpy()
      time_for_epoch = time.time() - start

      display.clear_output(wait=True)
      utils.generate_and_save_images(generator, epoch + 1, seed)

      if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
      print (f'tempo para a epoca {epoch + 1} é {time_for_epoch} seg')
      print(f"Loss Gerador: {avg_gen_loss}, Loss Discriminador: {avg_disc_loss}")
      log_file.write(f"{epoch + 1},{avg_gen_loss},{avg_disc_loss},{time_for_epoch}\n")


  display.clear_output(wait=True)
  utils.generate_and_save_images(generator, epochs, seed)
  

# inicia o treinamento pra
print("\nIniciando o treinamento...")
train(train_dataset, config.EPOCHS)
print("Treinamento concluído.")