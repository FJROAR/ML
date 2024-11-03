import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x = np.array(range(-500, 500))/100
y = x * np.sin(x)

# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# plot the function
plt.plot(x, y, 'g')

# show the plot
plt.show()

#Se inicia la búsqueda del mínimo en 1.5

inicio = 1.5
paso_descendente = 0.1

#Inicialización
t = inicio

#Gradiente descendente con 100 iteraciones
for i in range(100):
    t = t - paso_descendente * (np.sin(t) + t * np.cos(t))
    if t >= 5: t = 5
    if t <= -5: t = -5

print("Valor final: ", t)


#Se inicia la búsqueda del mínimo en 2.5

inicio = 2.5
paso_descendente = 0.1

#Inicialización
t = inicio

#Gradiente descendente con 100 iteraciones
for i in range(100):
    t = t - paso_descendente * (np.sin(t) + t * np.cos(t))
    if t >= 5: t = 5
    if t <= -5: t = -5
    

print("Valor final: ", t)