#https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html

from lifelines.datasets import load_dd

data = load_dd()
data.head()

from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()


#Ajuste del modelo. Se indica qué variables contienen la información del tiempo
#y de la censura
T = data["duration"]
E = data["observed"]

kmf.fit(T, event_observed=E)

#Se dibuja la curva en su versión más básica

from matplotlib import pyplot as plt

kmf.plot_survival_function()
plt.title('Función de Supervivencia de Régimes Políticos')
plt.show()

#Tiempo medio del mandato
from lifelines.utils import median_survival_times
median_ci = median_survival_times(kmf.confidence_interval_)

print("Tiempo medio del mandato = ", kmf.median_survival_time_)
print("Intervalo de Confianza min", median_ci.iloc[0,0])
print("Intervalo de Confianza max", median_ci.iloc[0,1])


import numpy as np

dem = (data["democracy"] == "Democracy")

ax = plt.subplot(111)

t = np.linspace(0, 50, 51)
kmf.fit(T[dem], event_observed=E[dem], timeline=t, label="Democratic Regimes")
ax = kmf.plot_survival_function(ax=ax)

kmf.fit(T[~dem], event_observed=E[~dem], timeline=t, label="Non-democratic Regimes")
ax = kmf.plot_survival_function(ax=ax)

plt.title("Lifespans of different global regimes");

plt.show()

from lifelines.statistics import logrank_test

results = logrank_test(T[dem], T[~dem], E[dem], E[~dem], alpha=.99)

results.print_summary()