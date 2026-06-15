import tensorflow as tf
import time
import os
from IPython import display
from config import model_A_config_128 as config
import dataset
import model_128 as model
import utils

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

output_dir = os.path.join("experiments", config.EXPERIMENT_NAME)
checkpoint_dir = os.path.join(output_dir, "checkpoints")
log_file_path = os.path.join(output_dir, "training_log.csv")
epochs_images_dir = os.path.join(output_dir, "images")
os.makedirs(checkpoint_dir, exist_ok=True)


#carregar os dados
print("Carregando e preparando o dataset train .py")
train_dataset = dataset.load_and_prepare_dataset()
print("Dataset pronto.")

# construir os modelos
print("Construindo o Gerador e o Discriminador")
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
    #acuracia
    real_accuracy = tf.reduce_mean(tf.cast(real_output > 0.0, tf.float32))
    fake_accuracy = tf.reduce_mean(tf.cast(fake_output <= 0.0, tf.float32))
    disc_accuracy = (real_accuracy + fake_accuracy) / 2.0
    #calcula o gradiente baseado no valor da perda
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    #gurada valor do gradiente pra o log
    gen_grad_norm = tf.linalg.global_norm(gradients_of_generator)
    disc_grad_norm = tf.linalg.global_norm(gradients_of_discriminator)
    #aplica penalidades
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss, gen_grad_norm, disc_grad_norm, disc_accuracy

# loop de treinamento
def train(dataset, epochs):
  # Cria a semente de ruído fixa para acompanhar o progresso
  seed = tf.random.normal([config.NUM_EXAMPLES_TO_GENERATE, config.NOISE_DIM])
  with open(log_file_path, "w") as log_file:
    log_file.write("epoch,gen_loss,disc_loss,gen_grad_norm,disc_grad_norm,disc_accuracy,time_sec\n")

    for epoch in range(epochs):
      start = time.time()
      
      epoch_gen_loss = []
      epoch_disc_loss = []
      epoch_gen_grad_norm = []
      epoch_disc_grad_norm = []
      epoch_disc_accuracy = []

      
      for image_batch in dataset:
        gen_loss, disc_loss, g_norm, d_norm, d_acc = train_step(image_batch)

        epoch_gen_loss.append(gen_loss)
        epoch_disc_loss.append(disc_loss)
        epoch_gen_grad_norm.append(g_norm)
        epoch_disc_grad_norm.append(d_norm)
        epoch_disc_accuracy.append(d_acc)
      
      avg_gen_loss = tf.reduce_mean(epoch_gen_loss).numpy()
      avg_disc_loss = tf.reduce_mean(epoch_disc_loss).numpy()
      avg_gen_gradient = tf.reduce_mean(epoch_gen_grad_norm).numpy()
      avg_disc_gradient = tf.reduce_mean(epoch_disc_grad_norm).numpy()
      avg_disc_acc = tf.reduce_mean(epoch_disc_accuracy).numpy()

      time_for_epoch = time.time() - start

      display.clear_output(wait=True)
      if (epoch + 1) % config.SAVE_IMAGES_AND_CHECKPOINTS_EVERY == 0 or epoch == 0:
        # Salva as imagens geradas a cada N épocas
        utils.generate_and_save_images(model=generator, epoch=epoch + 1, test_input=seed, directory=epochs_images_dir)

      if (epoch + 1) % config.SAVE_IMAGES_AND_CHECKPOINTS_EVERY == 0:
        # Salva o modelo a cada N épocas
        checkpoint.save(file_prefix = checkpoint_prefix)
      print(f'Tempo para a época {epoch + 1}: {time_for_epoch:.2f} seg')
      print(f"Loss Gerador: {avg_gen_loss:.4f}, Loss Discriminador: {avg_disc_loss:.4f}")
      print(f"Norma Grad G: {avg_gen_gradient:.4f}, Norma Grad D: {avg_disc_gradient:.4f}")
      print(f"Acurácia Discriminador: {avg_disc_acc:.4f} ({(avg_disc_acc * 100):.1f}%)")
      log_file.write(f"{epoch + 1},{avg_gen_loss},{avg_disc_loss},{time_for_epoch}\n")


  display.clear_output(wait=True)
  utils.generate_and_save_images(model=generator, epoch=epochs, test_input=seed, directory=epochs_images_dir)
  

# inicia o treinamento pra
print("\nIniciando o treinamento")
train(train_dataset, config.EPOCHS)
print("Treinamento concluído.")