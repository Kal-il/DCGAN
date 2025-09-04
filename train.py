import tensorflow as tf
import time
import os
from IPython import display
import config
import dataset
import model
import utils


# --- INÍCIO DO BLOCO DE CONFIGURAÇÃO DE MEMÓRIA DA GPU ---
# Coloque este bloco logo após suas importações

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Define um limite de memória virtual de 3GB (3 * 1024 = 3072 MB)
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "GPUs Físicas,", len(logical_gpus), "GPUs Lógicas")
  except RuntimeError as e:
    # A configuração de dispositivos virtuais deve ser feita antes da GPU ser inicializada
    print(e)
# --- FIM DO BLOCO DE CONFIGURAÇÃO ---


output_dir = os.path.join("experiments", config.EXPERIMENT_NAME)
checkpoint_dir = os.path.join(output_dir, "checkpoints")
log_file_path = os.path.join(output_dir, "training_log.csv")
epochs_images_dir = os.path.join(output_dir, "images")
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
    real_loss = cross_entropy(tf.ones_like(real_output), real_output) # compara quantos verdadeiros ele deu pras imagens reais
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output) # compara quantos falsos ele deu pras imagens falsas
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output) # quer que o gerador engane o discriminador, ou seja, que ele ache que as imagens falsas sao reais

#---------------------------------------------------------------------------------

# otimizadores

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_GENERATOR, beta_1=config.ADAM_BETA_1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_DISCRIMINATOR, beta_1=config.ADAM_BETA_1)

# checkpoints
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
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
      if (epoch + 1) % config.SAVE_IMAGES_AND_CHECKPOINTS_EVERY == 0 or epoch == 0:
        # Salva as imagens geradas a cada N épocas
        utils.generate_and_save_images(model=generator, epoch=epoch + 1, test_input=seed, directory=epochs_images_dir)

      if (epoch + 1) % config.SAVE_IMAGES_AND_CHECKPOINTS_EVERY == 0:
        # Salva o modelo a cada N épocas
        checkpoint.save(file_prefix = checkpoint_prefix)
      print (f'tempo para a epoca {epoch + 1} é {time_for_epoch} seg')
      print(f"Loss Gerador: {avg_gen_loss}, Loss Discriminador: {avg_disc_loss}")
      log_file.write(f"{epoch + 1},{avg_gen_loss},{avg_disc_loss},{time_for_epoch}\n")


  display.clear_output(wait=True)
  utils.generate_and_save_images(model=generator, epoch=epochs, test_input=seed, directory=epochs_images_dir)
  

# inicia o treinamento pra
print("\nIniciando o treinamento...")
train(train_dataset, config.EPOCHS)
print("Treinamento concluído.")