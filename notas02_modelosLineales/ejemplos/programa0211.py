#https://www.kdnuggets.com/2020/07/guide-survival-analysis-python-part-3.html


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter

data = pd.read_csv('data/lung.csv')
data.head()

#Se eliminan filas con datos nulos
data = data.dropna(subset=['time', 'status', 'age', 'sex', 'ph.ecog',
                           'ph.karno', 'pat.karno', 'meal.cal', 'wt.loss'])
data.shape


#Se llama al método km

cph = CoxPHFitter()
cph.fit(data, 'time', event_col = "status")

#Se observan los resultados obtenidos
cph.print_summary()

cph.plot()
plt.show()

#Estimación e funciones de supervivencia para 5 personas
d_data = data.iloc[0:5, :]
cph.predict_survival_function(d_data).plot()
plt.show()

