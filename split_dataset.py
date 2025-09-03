# split_dataset.py

import os
import random
import shutil

print("Iniciando a separação do dataset...")

# --- CONFIGURAÇÕES ---
# Caminho para a pasta original com todas as imagens
SOURCE_DIR = "102flowers/jpg"

# Caminhos para as novas pastas de treino e validação
TRAIN_DIR = "102flowers/train"
VALIDATION_DIR = "102flowers/validation"

# Proporção da divisão (80% para treino, 20% para validação)
SPLIT_RATIO = 0.8

# --- LÓGICA DO SCRIPT ---

# 1. Criar os diretórios de destino se não existirem
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

# 2. Listar todos os arquivos de imagem no diretório de origem
all_files = [f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]
print(f"Total de imagens encontradas: {len(all_files)}")

# 3. Embaralhar a lista de arquivos para garantir aleatoriedade
random.shuffle(all_files)

# 4. Calcular o ponto de divisão
split_point = int(len(all_files) * SPLIT_RATIO)

# 5. Dividir a lista de arquivos em treino e validação
train_files = all_files[:split_point]
validation_files = all_files[split_point:]

print(f"Separando {len(train_files)} imagens para treino...")
print(f"Separando {len(validation_files)} imagens para validação...")

# 6. Mover os arquivos para as pastas corretas
for filename in train_files:
    source_path = os.path.join(SOURCE_DIR, filename)
    destination_path = os.path.join(TRAIN_DIR, filename)
    shutil.move(source_path, destination_path)

for filename in validation_files:
    source_path = os.path.join(SOURCE_DIR, filename)
    destination_path = os.path.join(VALIDATION_DIR, filename)
    shutil.move(source_path, destination_path)

print("\nSeparação concluída!")
print(f"Os arquivos de treino estão em: '{TRAIN_DIR}'")
print(f"Os arquivos de validação estão em: '{VALIDATION_DIR}'")
print(f"A pasta original '{SOURCE_DIR}' agora deve estar vazia.")