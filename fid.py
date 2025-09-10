import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
import os

# Garante que estamos usando o TensorFlow 2.x e desabilita logs verbosos
tf.get_logger().setLevel('ERROR')
print(f"Usando TensorFlow v{tf.__version__}")

# --- FUNÇÕES DE CÁLCULO DO FID (não precisam ser alteradas) ---

def calculate_fid(model, images1_ds, images2_ds):
    """Calcula o FID entre dois datasets de imagens."""
    print("  Calculando ativações para o primeiro conjunto (reais)...")
    act1 = model.predict(images1_ds)
    print("  Calculando ativações para o segundo conjunto (geradas)...")
    act2 = model.predict(images2_ds)
    
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def preprocess_images(dataset):
    """Preprocessa um dataset de imagens para o formato do InceptionV3."""
    return dataset.map(lambda images: tf.keras.applications.inception_v3.preprocess_input(images))

# --- BLOCO PRINCIPAL DE EXECUÇÃO ---

if __name__ == '__main__':
    # --- 1. CONFIGURAÇÃO (Defina seus caminhos e parâmetros aqui) ---
    
    # Caminho para a pasta com as imagens REAIS (seu conjunto de validação)
    real_imgs_path = "102flowers/train"

    # Dicionário com os caminhos para as pastas de imagens GERADAS
    generated_imgs_paths = {
        "Época 600": "fid_batches/test_colab_artigo_600",
    }
    
    # Parâmetros das imagens
    BATCH_SIZE = 32
    IMG_HEIGHT = 128 # Ajuste para 128 se estiver testando as imagens maiores
    IMG_WIDTH = 128  # Ajuste para 128 se estiver testando as imagens maiores
    IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

    # --- 2. PREPARAÇÃO (Carregar modelo e imagens reais) ---
    
    print("Carregando o modelo InceptionV3 pré-treinado...")
    inception_model = tf.keras.applications.InceptionV3(
        include_top=False, 
        pooling='avg', 
        input_shape=IMG_SHAPE
    )

    try:
        print(f"\nCarregando imagens reais de: {real_imgs_path}")
        real_images_ds = tf.keras.utils.image_dataset_from_directory(
            real_imgs_path, label_mode=None, image_size=IMG_SHAPE[:2], batch_size=BATCH_SIZE)
        
        real_images_ds_preprocessed = preprocess_images(real_images_ds)
        print("Imagens reais carregadas e pré-processadas.")
    except Exception as e:
        print(f"ERRO CRÍTICO ao carregar imagens reais de '{real_imgs_path}': {e}")
        exit()

    # --- 3. CÁLCULO DO FID EM LOTE ---

    results = {}
    for name, gen_path in generated_imgs_paths.items():
        print(f"\n--- Processando Experimento: {name} ---")
        
        try:
            print(f"Carregando imagens geradas de: {gen_path}")
            generated_images_ds = tf.keras.utils.image_dataset_from_directory(
                gen_path, label_mode=None, image_size=IMG_SHAPE[:2], batch_size=BATCH_SIZE)
            
            generated_images_ds_preprocessed = preprocess_images(generated_images_ds)
            print("Imagens geradas carregadas e pré-processadas.")

            print("Calculando a pontuação FID... (Isso pode levar alguns minutos)")
            fid_score = calculate_fid(inception_model, real_images_ds_preprocessed, generated_images_ds_preprocessed)
            results[name] = fid_score
            print(f"-> FID para '{name}': {fid_score:.3f}")

        except Exception as e:
            print(f"ERRO ao processar o diretório '{gen_path}': {e}")
            results[name] = "ERRO"

    # --- 4. EXIBIÇÃO DOS RESULTADOS FINAIS ---
    
    print("\n\n====================================")
    print("   RESULTADOS FINAIS DO FID")
    print("  (Quanto menor, melhor)")
    print("====================================")
    print(f"{'Experimento':<15} | {'Pontuação FID':<15}")
    print("-" * 32)
    for name, score in results.items():
        if isinstance(score, float):
            print(f"{name:<15} | {score:<15.2f}")
        else:
            print(f"{name:<15} | {score:<15}")
    print("------------------------------------")