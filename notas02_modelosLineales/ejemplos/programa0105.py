#Fuente: http://ligdigonzalez.com/algoritmo-regresion-lineal-simple-machine-learning-practica-con-python/


#Se importan la librerias a utilizar
from sklearn import datasets
import numpy as np
import statsmodels.api as sm


#Importamos los datos de la misma librería de scikit-learn
boston = datasets.load_boston()

#Forma del dataset
print('Características del dataset:')
print(boston.DESCR)
print('Cantidad de datos:')
print(boston.data.shape)

#Información de las columnas
print('Nombres columnas:')
print(boston.feature_names)

#Se selecciona solamente la columna 5 del dataset
X1 = boston.data[:, np.newaxis, 5]


#Creamos una nueva collumna combinación lineal de la anterior
X2 = 1 + 3*X1

#Se añade una constante al modelo y se unifican las 2 columnas en la
#matriz de variables explicativas
X = np.concatenate((X1, X2), axis = 1)
X = sm.add_constant(X, prepend=False)

#Variable Esplicada
Y = boston.target



#Se define el algoritmo a utilizar
mod = sm.OLS(Y, X)
res = mod.fit()

print(res.summary())

#Python avisa de que puede exister multicolinealidad, aunque no soluciona el problema

#Nótese la salida que ofrece donde además se da información de la distribución
#de los errores en base al estadístico Jarque-bera

#Aún más grave cuando se usa sólo sklearn en el caso de la regresión logística

import numpy as np
from sklearn.linear_model import LogisticRegression
X = np.array([[0, 1], 
              [0, 1],
              [1, 0],
              [1, 0],
              [0, 1]]) 
Y = np.array([1,0,0,1,0])
logist = LogisticRegression(penalty = 'l1',
                            solver = 'liblinear').fit(X, Y)
logist.intercept_
logist.coef_


X = np.array([[0, 1], 
              [0, 1],
              [1, 0],
              [1, 0],
              [0, 1]]) 
Y = np.array([1,0,0,1,0])
logist = LogisticRegression(penalty = 'l2', 
                            solver = 'liblinear').fit(X, Y)
logist.intercept_
logist.coef_

#¡Lo estima como si fuera cualquier cosa, no da ningún aviso de error!

logist.predict_proba(X)