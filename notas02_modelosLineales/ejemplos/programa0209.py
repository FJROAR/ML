import pandas as pd
import numpy as np

from sklearn.datasets import load_boston
data = load_boston()



df = pd.read_csv('data/Boston.csv')
df.columns


y = df['medv'].values


#Se consideran sólo 2 variables importantes (conocidas por un ejemplo anterior)
#como RM y AGE 

X = df[['rm', 'age']].values


from sklearn.model_selection import train_test_split
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state = 0)

#Se estandarizan las variables en el training y se pasa al test 

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train_norm = ss.fit_transform(X_train)
X_test_norm = ss.transform(X_test)



#Defino el algoritmo KNN utilizar
from sklearn.neighbors import KNeighborsRegressor

#Algoritmo que usará la distancia euclídea
#Recuérdes que es buena práctica normalizar las observaciones en algoritmos
#de este tipo


def calcula_knn(k):
    
    knn = KNeighborsRegressor(n_neighbors = k, metric = 'minkowski', p = 2)
    knn.fit(X_train_norm, y_train)
    pred = knn.predict(X_test_norm)
    recm = (np.mean((y_test - pred)**2))**0.5
    print(k, "-vecinos ecm: ", recm)
    return pred

#Modelo con k = 1, validación en test
predk1 = calcula_knn(1)


#Modelo con k = 2, validación en test
predk2 = calcula_knn(2)


#Modelo con k = 5, validación en test
predk5 = calcula_knn(5)

#Modelo con k = 10, validación en test
predk10 = calcula_knn(10)

#Modelo con k = 10, validación en test
predk25 = calcula_knn(25)


#Comparación con una regresión lineal

from sklearn import linear_model

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train_norm, y_train)

pred = regr.predict(X_test_norm)
recm = (np.mean((y_test - pred)**2))**0.5
print("Regresión Lineal: ", recm)

#Representación del problema training - test (normalizado)

X_train_norm = pd.DataFrame(X_train_norm)
X_train_norm.columns = ['rm', 'age']
X_test_norm = pd.DataFrame(X_test_norm)
X_test_norm.columns = ['rm', 'age']


X_train_norm['color'] = 'blue'
X_test_norm['color'] = 'red'

X_total = pd.concat([X_train_norm, X_test_norm])

import matplotlib.pyplot as plt 

plt.scatter(X_total['rm'],X_total['age'],c=X_total['color'])
plt.xlabel("rm")
plt.ylabel("age")
plt.title("Representación Training (azul) vs Test (rojo)")
plt.show()


#Sin estandarizar variables

def calcula_knns(k):
    
    knn = KNeighborsRegressor(n_neighbors = k, metric = 'minkowski', p = 2)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    recm = (np.mean((y_test - pred)**2))**0.5
    print(k, "-vecinos ecm: ", recm)
    return pred

#Modelo con k = 1, validación en test
predk1s = calcula_knns(1)


#Modelo con k = 2, validación en test
predk2s = calcula_knns(2)


#Modelo con k = 5, validación en test
predk5s = calcula_knns(5)

#Modelo con k = 10, validación en test
predk10s = calcula_knns(10)

#Modelo con k = 10, validación en test
predk25s = calcula_knns(25)


#Comparación con una regresión lineal

from sklearn import linear_model

regrs = linear_model.LinearRegression()

# Train the model using the training sets
regrs.fit(X_train, y_train)

preds = regrs.predict(X_test)
recm = (np.mean((y_test - preds)**2))**0.5
print("Regresión Lineal: ", recm)
