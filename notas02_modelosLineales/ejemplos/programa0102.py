import numpy as np

# Creación de un conjunto de datos para entrenamiento
trX = np.linspace(-2, 2, 101)
trY = 3 + 2 * trX + np.random.randn(*trX.shape) * 0.33

# Definición de los ajustes y parámetros iniciales
num_steps = 100
learningRate = 0.10
criteria = 1e-8
b_0 = 1
b_1 = 1


# Proceso iterativo
for step in range(0, num_steps):
    b_0_gradient = 0
    b_1_gradient = 0
    N = float(len(trX))
    for i in range(0, len(trX)):
        b_0_gradient -= (2/N) * (trY[i] - (b_0 + b_1 * trX[i]))
        b_1_gradient -= (2/N) * (trY[i] - (b_0 + b_1 * trX[i])) * trX[i]
        
        
    b_0 = b_0 - (learningRate * b_0_gradient)
    b_1 = b_1 - (learningRate * b_1_gradient)
    if max(abs(learningRate * b_0_gradient), abs(learningRate * b_1_gradient)) < criteria:
        break
    
# Impresión de los resultados
print("Los valores que se obtienen son:", b_0, b_1, "en ", step, " pasos")


#Se calcula las estimaciones con sklearn
from sklearn.linear_model import LinearRegression

linear = LinearRegression().fit(trX.reshape((len(trX),1)), trY.reshape((len(trY),1)))

linear.intercept_
linear.coef_

