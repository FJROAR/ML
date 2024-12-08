import numpy as np
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt



#Valores, en este caso matriciale, que se desea distribuir, las 2 primeras 
#componente son las ubicaciones de las muestras y la tercera columan el valor de
#éstas

data = np.array(
    [
        [0.3, 1.2, 0.47],
        [1.9, 0.6, 0.56],
        [1.1, 3.2, 0.74],
        [3.3, 4.4, 1.47],
        [4.7, 3.8, 1.74],
    ]
)




#Rejilla 2 x 2 para representar

gridx = np.arange(0.0, 5.5, 0.5)
gridy = np.arange(0.0, 5.5, 0.5)


#La construcción de un krigeado ordinario, requiere un modelo de variograma,
#si no se dice nada, se supone uno de corte lineal

OK = OrdinaryKriging(
    data[:, 0],
    data[:, 1],
    data[:, 2],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
)


#Se crea la distribución de la media y la varianza a lo largo de la red
#bi-dimensional

z, ss = OK.execute("grid", gridx, gridy)


#Se representa el resultado

#kt.write_asc_grid(gridx, gridy, z, filename="output.asc")
plt.imshow(z)
plt.show()