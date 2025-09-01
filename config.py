# Arquivo para constantes e hiperparâmetros

# --- Configurações do Treinamento ---
EPOCHS = 100 # Podemos começar com um valor maior para um dataset mais complexo
BATCH_SIZE = 128 # tamanho do batch
BUFFER_SIZE = 10000 # tamanho do dataset de treino

# --- Configurações do Modelo e da Imagem ---
IMAGE_HEIGHT = 64   # Altura da imagem de flor (ex: 64x64)
IMAGE_WIDTH = 64    # Largura da imagem de flor
IMAGE_CHANNELS = 3  # 3 para imagens coloridas (RGB)
NOISE_DIM = 100     # Tamanho do vetor de ruído (espaço latente)

# --- Configurações de Saída ---
NUM_EXAMPLES_TO_GENERATE = 16
CHECKPOINT_DIR = './training_checkpoints'

DATA_DIR = "102flowers/jpg"
