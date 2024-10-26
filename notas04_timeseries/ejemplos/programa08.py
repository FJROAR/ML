import numpy as np
import pandas as pd
import numpy as np 
from scipy import stats 
import seaborn as sns 
import matplotlib.pyplot as plt 


df = pd.read_csv('data/pernoctaciones.csv', sep = ";")

np.std(df['pernoctaciones']) / np.mean(df['pernoctaciones'])


fitted_data, fitted_lambda = stats.boxcox(df['pernoctaciones']) 

fig, ax = plt.subplots(1, 2)

sns.distplot(df['pernoctaciones'], kde = True, 
            kde_kws = {'shade': True, 'linewidth': 2},  
            label = "Non-Normal", color ="green", ax = ax[0]) 
  
sns.distplot(fitted_data, kde = True, 
            kde_kws = {'shade': True, 'linewidth': 2},  
            label = "Normal", color ="red", ax = ax[1]) 
  
plt.legend(loc = "upper right") 
  
fig.set_figheight(5) 
fig.set_figwidth(10) 
  
print(f"Lambda value used for Transformation: {fitted_lambda}") 

#Transformación manual
serie_ajust = (df['pernoctaciones']**(-0.0018700305497903744) - 1)/-0.0018700305497903744

np.std(serie_ajust) / np.mean(serie_ajust)

#Estimación de la constante Box-Cox según el método de máxima normalidad
#dato en https://arxiv.org/ftp/arxiv/papers/1401/1401.3812.pdf

lambda_est = np.arange(-1000, 1000) / 1000

v_test = []

for i in lambda_est:
    
    if (i == 0):
        s_transf = np.log(df['pernoctaciones'])
        v_test.append(stats.shapiro(s_transf).pvalue)
    else:
        s_transf = (df['pernoctaciones']**(i) - 1)/i
        v_test.append(stats.shapiro(s_transf).pvalue)
    
table_res = pd.DataFrame({'bc_constant': lambda_est,
                          'test_shapiro': v_test})



#Representación de la elección

transformada = (df['pernoctaciones']**(0.017) - 1)/0.017


sns.distplot(pd.Series(transformada))
plt.legend(loc = "upper right") 
fig.set_figheight(5) 

#np.std(transformada) / np.mean(transformada)