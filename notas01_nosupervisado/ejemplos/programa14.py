"""
https://rubialesalberto.medium.com/clustering-con-dbscan-y-hdbscan-con-python-y-sus-hiperpar%C3%A1metros-en-sklearn-8728283b96ac

Instalación de hdbscan: https://stackoverflow.com/questions/65831739/how-to-install-hdbscan-modula-python-3-7-windows-10
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from hdbscan import HDBSCAN

#Cargamos el dataframe
#_________________________________________________________________________________________


data = np.load('data/clusterable_data.npy')

df_cluster = pd.DataFrame(data)

df_cluster.columns = ['x', 'y']

#Se instancia el algoritmo
#_________________________________________________________________________________________
hdbscan = HDBSCAN(min_cluster_size=10,
                 min_samples=10)
                 
#Se entrena y se predice
#_________________________________________________________________________________________
preds_2 = hdbscan.fit_predict(df_cluster)


#Métricas. Como se observa se puede aplicar el método silhouette
#_________________________________________________________________________________________
silhouette_score(df_cluster, preds_2)
calinski_harabasz_score(df_cluster, preds_2)

#Graficamos
#_________________________________________________________________________________________
df_cluster.plot(kind='scatter', x='x', y='y', 
                c=preds_2, cmap='Accent_r', figsize=(16,10))



#####
from sklearn.cluster import KMeans

clusterer = KMeans(n_clusters=6, random_state=10)
cluster_labels = clusterer.fit_predict(df_cluster)

df_cluster.plot(kind='scatter', x='x', y='y', 
                c= cluster_labels, cmap='Accent_r', figsize=(16,10))
