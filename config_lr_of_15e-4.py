# config.py

# Nome do experimento para logs e checkpoints
EXPERIMENT_NAME = "test_2_1000_epochs_4x4_base_64x64_gen_lr_adjusted" # para ver os experimentos ja feitos, veja a pasta experiments

# --- configurações do dataset e do treinamento ---
EPOCHS = 1000
BATCH_SIZE = 128  # Conforme a secão de treinamento (4) do artigo
BUFFER_SIZE = 6551 # tamanho do dataset de treino   (tamamnhop do dataset: 8189) (tamanho do tataset de validacao: 1638)

# --- configuracoes do otimizador Adam ---
LEARNING_RATE_GENERATOR = 0.0002 # Conforme a seção de treinamento (4) do artigo
# LEARNING_RATE_DISCRIMINATOR = 0.0002 # Conforme a seção de treinamento (4) do artigo
LEARNING_RATE_DISCRIMINATOR = 0.00015 # um ajuste menor para desacelarar o treinamento do discriminador só um pouco 
ADAM_BETA_1 = 0.5    # Conforme a seção de treinamento (4) do artigo

# --- configuracoes da arquitetura ---
LEAKY_RELU_SLOPE = 0.2  # Slope do LeakyReLU no Discriminador
WEIGHT_INIT_STDDEV = 0.02  # Desvio padrão para inicialização dos pesos
NOISE_DIM = 100     # Tamanho do vetor de ruído (espaço latente) o mesmo usado no artigo original

# --- Configurações do Modelo e da Imagem ---
IMAGE_HEIGHT = 64   # Altura da imagem de flor (ex: 64x64)
IMAGE_WIDTH = 64    # Largura da imagem de flor
IMAGE_CHANNELS = 3  # 3 para imagens coloridas (RGB)

# --- Configurações de Saída ---
NUM_EXAMPLES_TO_GENERATE = 16
SAVE_IMAGES_AND_CHECKPOINTS_EVERY = 10  # Salva imagens a cada N épocas

DATA_DIR = "102flowers/train"