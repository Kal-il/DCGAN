import scipy.io
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# --- CONFIGURAÇÃO ---
# Caminho para o arquivo .mat com as distâncias
DISTANCES_FILE_PATH = 'datasplits/distancematrices102.mat'  # Ajuste o caminho se necessário



# Quantos "grupos de formato" você quer criar?
# Este é um hiperparâmetro que você pode ajustar. Vamos começar com 10.
NUM_SHAPE_CLUSTERS = 10

# --- LÓGICA PRINCIPAL ---
print(f"Carregando o arquivo de distâncias de: {DISTANCES_FILE_PATH}")
# Carrega o arquivo .mat. Ele vem como um dicionário.
mat_data = scipy.io.loadmat(DISTANCES_FILE_PATH)

print("Chaves disponíveis no arquivo .mat:", mat_data.keys())


"""
Dhsv: Distância baseada em Cor. (HSV - Hue, Saturation, Value é um espaço de cores). Esta matriz agrupa flores com cores parecidas.

Dsiftint: Distância baseada em Textura e Características Internas da flor (usando o algoritmo SIFT no interior da flor).

Dsiftbdy: Distância baseada no Contorno/Borda da flor (usando SIFT na borda, boundary).

Dhog: Distância baseada na Forma Geral e estrutura (usando a técnica HOG - Histogram of Oriented Gradients).
"""
distance_matrix = mat_data['Dhsv']
print(f"Matriz de distâncias carregada com sucesso. Formato: {distance_matrix.shape}")

# Inicializa o algoritmo de clusterização.
# Usamos 'precomputed' para dizer a ele que já temos a matriz de distâncias.
print(f"Iniciando a clusterização em {NUM_SHAPE_CLUSTERS} grupos...")
clustering = AgglomerativeClustering(
    n_clusters=NUM_SHAPE_CLUSTERS, 
    metric='precomputed', 
    linkage='average'
)

# Executa a clusterização
shape_labels = clustering.fit_predict(distance_matrix)
print("Clusterização concluída.")

# --- RESULTADO ---
# 'shape_labels' é agora um array onde cada elemento é o ID do cluster de formato
# para a imagem correspondente.
# Ex: shape_labels[0] é o cluster da imagem 1, shape_labels[1] é o da imagem 2, etc.

print("\nExemplo dos primeiros 20 resultados (Imagem -> Cluster de Formato):")
for i in range(20):
    # O dataset é indexado a partir de 1, então usamos i+1 para a imagem
    print(f"Imagem {i+1:04d} pertence ao cluster de formato -> {shape_labels[i]}")

# Agora você pode salvar esses labels para usar depois
np.save('shape_labels.npy', shape_labels)
print("\nOs labels de formato foram salvos no arquivo 'shape_labels.npy'")