#https://www.kaggle.com/izzettunc/introduction-to-time-series-clustering

# Native libraries
import os
import math
# Essential Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Preprocessing
from sklearn.preprocessing import MinMaxScaler
# Algorithms
from minisom import MiniSom
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA



#Se leen los datos que están .csv por separados en un determinado directorio

directory = 'data/retail-and-retailers-sales-time-series-collection/'

mySeries = []
namesofMySeries = []
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        df = pd.read_csv(directory+filename)
        df = df.loc[:,["date","value"]]
        # While we are at it I just filtered the columns that we will be working on
        df.set_index("date",inplace=True)
        # ,set the date columns as index
        df.sort_index(inplace=True)
        # and lastly, ordered the data according to our date index
        mySeries.append(df)
        namesofMySeries.append(filename[:-4])


#Se han leído unas 23 series temporales

#Visualización de datos

fig, axs = plt.subplots(6,4,figsize=(25,25))
fig.suptitle('Series')
for i in range(6):
    for j in range(4):
        if i*4+j+1>len(mySeries): # pass the others that we can't fill
            continue
        axs[i, j].plot(mySeries[i*4+j].values)
        axs[i, j].set_title(namesofMySeries[i*4+j])
plt.show()


#Pre-procesado

#Longitud de series
#En este caso se va a proceder a "rellenar" missings y hacer todas las series que
#empiecen y finalicen a la vez

series_lengths = {len(series) for series in mySeries}
print(series_lengths)
ind = 0
for series in mySeries:
    print("["+str(ind)+"] "+series.index[0]+" "+series.index[len(series)-1])
    ind+=1

max_len = max(series_lengths)
longest_series = None
for series in mySeries:
    if len(series) == max_len:
        longest_series = series


#Se re-indexan todas las series
problems_index = []

for i in range(len(mySeries)):
    if len(mySeries[i])!= max_len:
        problems_index.append(i)
        mySeries[i] = mySeries[i].reindex(longest_series.index)
for i in problems_index:
    mySeries[i].interpolate(limit_direction="both",inplace=True)

#Se comprueba que todas las series están completas
def nan_counter(list_of_series):
    nan_polluted_series_counter = 0
    for series in list_of_series:
        if series.isnull().sum().sum() > 0:
            nan_polluted_series_counter+=1
    print(nan_polluted_series_counter)

nan_counter(mySeries)

#Proceso de normalización (cada serie se normaliza usando sólo sus propios
#valores, no el de otras series)

for i in range(len(mySeries)):
    scaler = MinMaxScaler()
    mySeries[i] = MinMaxScaler().fit_transform(mySeries[i])
    mySeries[i]= mySeries[i].reshape(len(mySeries[i]))



#Reducción de la dimensionalidad con pca

pca = PCA(n_components=2)
mySeries_transformed = pca.fit_transform(mySeries)

plt.figure(figsize=(25,10))
plt.scatter(mySeries_transformed[:,0],mySeries_transformed[:,1], s=300)
plt.show()

#Se aplica un kmeans sobre las anteriores componente con k = 5
kmeans = KMeans(n_clusters=5, max_iter=5000)

labels = kmeans.fit_predict(mySeries_transformed)

#Representación de la asociación

plt.figure(figsize=(25,10))
plt.scatter(mySeries_transformed[:, 0], mySeries_transformed[:, 1], c=labels, s=300)
plt.show()

#El balanceo queda algo mejor
cluster_c = [len(labels[labels==i]) for i in range(5)]
cluster_n = ["cluster_"+str(i) for i in range(5)]
plt.figure(figsize=(15,5))
plt.title("Cluster Distribution for KMeans")
plt.bar(cluster_n,cluster_c)
plt.show()