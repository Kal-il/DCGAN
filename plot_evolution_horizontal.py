# plot_evolution_horizontal.py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# --- CONFIGURAÇÕES ---

# 1. Defina o diretório do seu experimento
IMAGES_DIR = "experiments/reLU_on_generator"  # ATENÇÃO: Mude para o nome do seu experimento

# 2. As 10 épocas que queremos plotar (de 10 a 100)
EPOCHS_TO_PLOT = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 3. Nome do arquivo de saída
OUTPUT_FILENAME = "training_evolution_horizontal.png"

# --- LÓGICA DO SCRIPT ---

# Cria uma figura com 1 LINHA e 10 COLUNAS.
# O figsize está mais largo do que alto para acomodar o layout horizontal.
fig, axes = plt.subplots(1, len(EPOCHS_TO_PLOT), figsize=(20, 2))

print(f"Gerando a imagem de evolução horizontal: {OUTPUT_FILENAME}")

# Itera através da lista de épocas e dos eixos do subplot.
for i, epoch in enumerate(EPOCHS_TO_PLOT):
    filename = f'image_at_epoch_{epoch:04d}.png'
    image_path = os.path.join(IMAGES_DIR, filename)

    try:
        img = mpimg.imread(image_path)
        ax = axes[i]  # Seleciona o subplot atual
        ax.imshow(img)
        ax.axis('off')  # Remove as bordas

        # Usa set_title para colocar o texto ACIMA de cada imagem
        ax.set_title(f"Época {epoch}", size=10)

    except FileNotFoundError:
        print(f"AVISO: Imagem para a época {epoch} não encontrada em '{image_path}'. Pulando.")

# Ajusta o espaçamento horizontal (width space) entre as imagens.
plt.subplots_adjust(wspace=0.05, hspace=0)

# Salva a figura final em alta resolução.
plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')

print("Figura de evolução horizontal salva com sucesso!")

plt.close(fig)