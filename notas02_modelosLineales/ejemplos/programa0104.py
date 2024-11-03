import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt


# Creación de un conjunto de datos para entrenamiento

def coste(w0, w1, alpha = 0-1):

    X = [1, 2, 4, 8, 9]
    Y = [3, 6, 7, 10, 12]
    C = 0

    for i in range(5):
        C += (w0 + w1*X[i] - Y[i])**2
        
    C = C + alpha * (w0**2 + w1**2)
    
    return C


w0 = np.arange(0, 6.0,0.01)
w1 = np.arange(0, 2, 1)
p1, p2 = np.meshgrid(w0, w1) # grid of point
C = coste(p1, p2, 0.1) # evaluation of the function on the grid

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(p1, p2, C, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
#El mínimo se sabe que se tiene en: coste(3.0079, 0.9567) 
#y vale 0.540945
