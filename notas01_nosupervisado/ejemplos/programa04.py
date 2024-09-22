import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance


Clientes = [[25, 23000],
     [30, 28000],
     [45, 50000],
     [65, 30000],
     [70, 27000]]

X = pd.DataFrame(Clientes)


M_euclidea = euclidean_distances(X, X)

#Se normalizan variables
X_norm = pd.DataFrame(Clientes)

X_norm[0] = (X[0] - np.mean(X[0])) / np.std(X[0])
X_norm[1] = (X[1] - np.mean(X[1])) / np.std(X[1])

M_euclidea2 = euclidean_distances(X_norm, X_norm)


#Distancia Mahalanobis

cov = np.cov(X.values.T)
inv_covmat = np.linalg.inv(cov)

M_Mahalanobis = [[0 for x in range(5)] for y in range(5)] 

for i in range(5):
    for j in range(5):
        M_Mahalanobis[i][j] = distance.mahalanobis(X.iloc[i], 
                                                   X.iloc[j], 
                                                   inv_covmat)

M_Mahalanobis = pd.DataFrame(M_Mahalanobis)
