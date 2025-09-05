# Guia Completo: TensorFlow-GAN para FID

# ============================================
# 1. INSTALAÇÃO
# ============================================
"""
pip install tensorflow-gan
pip install tensorflow
pip install numpy
pip install pillow
"""

import tensorflow as tf
import tensorflow_gan as tfgan
import numpy as np
from PIL import Image
import os

# ============================================
# 2. FUNÇÃO BÁSICA PARA CALCULAR FID
# ============================================

def calculate_fid_tfgan(real_images_path, generated_images_path, batch_size=50, num_inception_images=50000):
    """
    Calcula FID usando TensorFlow-GAN
    
    Args:
        real_images_path: pasta com imagens reais
        generated_images_path: pasta com imagens geradas
        batch_size: tamanho do batch para processar
        num_inception_images: número de imagens do Inception para estatísticas
    """
    
    # Função para carregar e preprocessar imagens
    def load_and_preprocess_images(image_paths):
        def preprocess_image(image_path):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.image.resize(image, [299, 299])  # Tamanho do Inception
            image = tf.cast(image, tf.float32)
            image = (image - 127.5) / 127.5  # Normalizar para [-1, 1]
            return image
        
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(preprocess_image)
        dataset = dataset.batch(batch_size)
        return dataset
    
    # Carregar paths das imagens
    real_paths = [os.path.join(real_images_path, f) for f in os.listdir(real_images_path)]
    gen_paths = [os.path.join(generated_images_path, f) for f in os.listdir(generated_images_path)]
    
    # Criar datasets
    real_dataset = load_and_preprocess_images(real_paths)
    gen_dataset = load_and_preprocess_images(gen_paths)
    
    # Calcular FID
    fid = tfgan.eval.frechet_inception_distance(
        real_dataset,
        gen_dataset,
        num_inception_images=num_inception_images
    )
    
    return fid

# ============================================
# 3. EXEMPLO DE USO DIRETO
# ============================================

def exemplo_uso_simples():
    # Caminhos para suas pastas
    pasta_reais = "path/to/real/images"
    pasta_geradas = "path/to/generated/images"
    
    # Calcular FID
    with tf.Session() as sess:
        fid_score = calculate_fid_tfgan(pasta_reais, pasta_geradas)
        fid_value = sess.run(fid_score)
        print(f"FID Score: {fid_value:.2f}")


def calculate_fid_from_arrays(real_images, generated_images):
    """
    Calcula FID a partir de arrays numpy
    
    Args:
        real_images: array numpy (N, H, W, 3) com imagens reais
        generated_images: array numpy (N, H, W, 3) com imagens geradas
    """
    
    # Converter para tensors do TensorFlow
    real_images_tf = tf.constant(real_images, dtype=tf.float32)
    gen_images_tf = tf.constant(generated_images, dtype=tf.float32)
    
    # Redimensionar para 299x299 se necessário
    if real_images.shape[1] != 299:
        real_images_tf = tf.image.resize(real_images_tf, [299, 299])
        gen_images_tf = tf.image.resize(gen_images_tf, [299, 299])
    
    # Normalizar para [-1, 1] se ainda não estiver
    real_images_tf = (real_images_tf - 127.5) / 127.5
    gen_images_tf = (gen_images_tf - 127.5) / 127.5
    
    # Calcular FID
    fid = tfgan.eval.frechet_inception_distance(
        real_images_tf,
        gen_images_tf
    )
    
    return fid


if __name__ == "__main__":
    print("Testando FID com dados sintéticos...")
    
    real_imgs = None
    gen_imgs = None
    
    # Calcular FID
    fid_score = calculate_fid_from_arrays(real_imgs, gen_imgs)
    
    if fid_score is not None:
        print(f"FID calculado: {fid_score:.2f}")
    else:
        print("Erro no cálculo do FID")

# ============================================
# DICAS DE USO:
# ============================================
"""
1. Suas imagens devem estar no formato (N, H, W, 3)
2. Valores de pixel podem ser [0, 255] ou [0, 1] - a função normaliza automaticamente
3. Recomendo pelo menos 2048 imagens para FID estável
4. Se der erro de memória, diminua o batch_size
5. Para comparar com literatura, use exatamente o mesmo número de amostras
"""