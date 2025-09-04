import matplotlib.pyplot as plt
import config
import os
import imageio
import glob

def generate_and_save_images(model, epoch, test_input, directory):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

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

def generate_image(model, test_input):
    predictions = model(test_input, training=False)
    # Normaliza a imagem de [-1, 1] para [0, 1] para plotar
    img = predictions[0, :, :, :] * 0.5 + 0.5
    plt.imshow(img)
    plt.axis('off')
    plt.show()


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