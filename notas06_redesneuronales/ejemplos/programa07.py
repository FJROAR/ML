#https://www.analyticsvidhya.com/blog/2021/09/beginners-guide-to-anomaly-detection-using-self-organizing-maps/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Minisom library and module is used for performing Self Organizing Maps
from minisom import MiniSom

data = pd.read_csv('data/Credit_Card_Applications.csv')

# Defining X variables for the input of SOM
X = data.iloc[:, 1:14].values
y = data.iloc[:, -1].values
# X variables:
pd.DataFrame(X)

#Se realiza un escalado de las variables
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
pd.DataFrame(X)


#Se establecen los parámetros del modelo
som_grid_rows = 10
som_grid_columns = 10
iterations = 20000
sigma = 1
learning_rate = 0.5

#Se define la red
som = MiniSom(x = som_grid_rows, 
              y = som_grid_columns, 
              input_len=13, 
              sigma=sigma, 
              learning_rate=learning_rate)

# Initializing the weights
som.random_weights_init(X)

#Entrenamiento
som.train_random(X, iterations)


#Distancias en la red
som.distance_map()


#Se representan en la red definida y se señalan los grupos con y sin fraude
#Red = Fraude; Green = No fraude

from pylab import plot, axis, show, pcolor, colorbar, bone
bone()
pcolor(som.distance_map().T)       # Distance map as background
colorbar()
show()


bone()
pcolor(som.distance_map().T)
colorbar() #gives legend
markers = ['o', 's']        # if the observation is fraud then red circular color 
                            #or else green square
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)

show()