# plot_evolution.py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# --- CONFIGURAÇÕES ---

# 1. Defina o diretório onde as imagens do treinamento foram salvas.
#    Lembre-se que criamos uma pasta de experimento, ex: 'experiments/flores_v1_paper_params'
IMAGES_DIR = "experiments/reLU_on_generator"

# 2. Escolha as 5 épocas que você quer mostrar na figura final.
EPOCHS_TO_PLOT = [1, 25, 50, 75, 100]

# 3. Defina o nome do arquivo de saída.
OUTPUT_FILENAME = "training_evolution.png"

# --- LÓGICA DO SCRIPT ---

# Cria uma figura com 5 linhas e 1 coluna.
# O figsize controla o tamanho final da imagem (largura, altura em polegadas).
fig, axes = plt.subplots(len(EPOCHS_TO_PLOT), 1, figsize=(5, 25))

print(f"Gerando a imagem de evolução: {OUTPUT_FILENAME}")

# Itera através da lista de épocas e dos eixos do subplot.
for i, epoch in enumerate(EPOCHS_TO_PLOT):
    # Monta o caminho completo para cada imagem de época.
    # O {:04d} garante que o número da época tenha 4 dígitos (ex: 0001, 0025).
    filename = f'image_at_epoch_{epoch:04d}.png'
    image_path = os.path.join(IMAGES_DIR, filename)

    try:
        # Carrega a imagem do disco.
        img = mpimg.imread(image_path)

        # Seleciona o subplot atual (ax).
        ax = axes[i]

        # Mostra a imagem no subplot.
        ax.imshow(img)

        # Remove as bordas e os números dos eixos (x, y).
        ax.axis('off')

        # Adiciona um título à esquerda de cada imagem para identificar a época.
        ax.set_ylabel(f"Época {epoch}", size=12, labelpad=10)

    except FileNotFoundError:
        print(f"AVISO: Imagem para a época {epoch} não encontrada em '{image_path}'. Pulando.")

# Ajusta o espaçamento entre as imagens para quase zero.
# hspace é o espaço vertical.
plt.subplots_adjust(wspace=0, hspace=0.05)

# Salva a figura final no disco.
# dpi (dots per inch) controla a resolução da imagem salva.
plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')

print("Figura de evolução salva com sucesso!")

plt.close(fig)  # Fecha a figura para liberar memória.