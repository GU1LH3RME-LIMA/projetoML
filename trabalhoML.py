import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

dados = pd.read_csv("/content/dados.csv")
X = dados[['X', 'Y']]
#dados.info()
dados = dados.drop('fid', axis = 1)
#dados.info()
# Normalizando as coordenadas
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# Escolhendo o número de clusters (k)
k = 50  # ajuste conforme necessário

# Aplicando o K-means
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_X)

# Adicionando os rótulos de cluster ao seu conjunto de dados
dados['cluster_label'] = kmeans.labels_

# Visualização dos clusters
plt.scatter(scaled_X[:, 0], scaled_X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('Agrupamento de Pontos com K-means K='+str(k))
plt.xlabel('Coordenada X (normalizada)')
plt.ylabel('Coordenada Y (normalizada)')



plt.show()
