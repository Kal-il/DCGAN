# from PIL import Image

# import numpy as np
import matplotlib.pyplot as plt
import config
import os
import imageio
import glob
import tensorflow as tf

directory = os.path.join("fid_batches", config.EXPERIMENT_NAME)

def generate_and_save_images(model, epoch, test_input, directory):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    # fig.subplots_adjust(0,0,1,1)

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # Normaliza a imagem de [-1, 1] para [0, 1] para plotar
        plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5)
        plt.axis('off')

    filename ='image_at_epoch_{:04d}.png'.format(epoch)
    if not os.path.exists(directory):
        os.makedirs(directory)
    full_path = os.path.join(directory, filename)
    plt.savefig(full_path)
    plt.close(fig)

def generate_image(model, test_input, output_file_name):
    predictions = model(test_input, training=False)
    # Normaliza a imagem de [-1, 1] para [0, 1] para plotar
    img = predictions[0, :, :, :] * 0.5 + 0.5

    fig = plt.figure(figsize=plt.figaspect(img))

    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis('off')
    ax.imshow(img)
    filename = f'{output_file_name}.png'
    if not os.path.exists('single_images'):
        os.makedirs('single_images')
    full_path = os.path.join('single_images', filename)

    fig.savefig(full_path, bbox_inches='tight', pad_inches=0)

    plt.close(fig)

def generate_batch_of_images(model, vector_input):
    predictions = model(vector_input, training=False)
    for prediction in range(predictions.shape[0]):
        img = predictions[prediction, :, :, :] * 0.5 + 0.5

        fig = plt.figure(figsize=plt.figaspect(img))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(img)

        filename = f'image_{prediction}.png'
        if not os.path.exists(directory):
            os.makedirs(directory)
        full_path = os.path.join(directory, filename)

        fig.savefig(full_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
def generate_and_save_in_batches(model, num_images, output_dir, batch_size=64):
    """
    Gera um grande número de imagens em lotes para evitar erros de memória
    e as salva em um diretório.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Salvando {num_images} imagens em: {output_dir}")
    
    generated_count = 0
    while generated_count < num_images:
        # Define o tamanho do lote atual, cuidando para não exceder o total
        current_batch_size = min(batch_size, num_images - generated_count)
        if current_batch_size <= 0:
            break
            
        # --- A MUDANÇA CRUCIAL ESTÁ AQUI ---
        # O ruído é criado em um lote PEQUENO a cada iteração do loop
        noise = tf.random.normal([current_batch_size, config.NOISE_DIM])
        
        predictions = model(noise, training=False)
        
        # Salva cada imagem do lote
        for i in range(predictions.shape[0]):
            tf.keras.utils.save_img(
                os.path.join(output_dir, f'image_{generated_count + i:05d}.png'),
                predictions[i],
                scale=True # Converte de [-1, 1] para [0, 255]
            )
        
        generated_count += predictions.shape[0]
        print(f"  ... {generated_count}/{num_images} imagens salvas")

    print("Geração de imagens concluída.")
        


def create_evolution_gif(image_folder, output_path):
    """
    Encontra todas as imagens .png em uma pasta, as ordena e cria um GIF.
    """
    print("Criando o GIF de evolução do treinamento...")

    # Encontra todos os arquivos de imagem e os ordena
    # O padrão 'image_at_epoch_*.png' já permite ordenação alfabética correta
    filenames = glob.glob(os.path.join(image_folder, 'image_at_epoch_*.png'))
    filenames = sorted(filenames)

    if not filenames:
        print("Nenhuma imagem encontrada para criar o GIF.")
        return

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))

    # Salva as imagens como um GIF animado
    # fps (frames per second) controla a velocidade da animação
    imageio.mimsave(output_path, images, fps=5)
    print(f"GIF de evolução salvo em: {output_path}")