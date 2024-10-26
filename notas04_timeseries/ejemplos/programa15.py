import pandas as pd
import numpy as np
from scipy.stats import boxcox
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('data/pernoctaciones.csv', sep = ';')

df_test = df[0: 15].sort_values(by = ['fch']).reset_index(drop = True)
df_train = (df[15: len(df['pernoctaciones'])]).sort_values(by = ['fch']).reset_index(drop = True)

df_train = df_train.sort_values(by = ['fch'])

df_train_t, best_lambda = boxcox(df_train['pernoctaciones'])

print('Transformada de box-cox: ', best_lambda)

#Se crea la tabla para predicción a partir de la serie transformada

df_train_t = pd.DataFrame({'fch': df_train['fch'],
                           'value': df_train_t})


#Se consideran un número de retardos r = 6 y un horizonte temporal igual a
#h = 1

h = 1
n_col = 6 + h
n_row = len(df_train_t['value']) - n_col + 1


matrix = np.zeros((n_row, n_col))

for i in range(n_row):
    for j in range(n_col):
        matrix[i, j] = df_train_t['value'][i + j]


df_train_t = pd.DataFrame(matrix)
df_train_t.columns = ['var_6', 'var_5', 'var_4', 'var_3', 'var_2', 'var_1',
                      'target']


#Para ver si el modelo, al menos a instante h = 1 va a ser o no adecuado, al tener
#este nuevo dataset incorporada la estructura temporal en sus registros, es factible
#hacer la típica división training - test que ya se hacía en los modelos habituales
#de ML


X_train2, X_test2, y_train2, y_test2 = train_test_split(
                                        df_train_t.drop('target', axis = 'columns'),
                                        df_train_t['target'],
                                        train_size   = 0.7,
                                        random_state = 1234,
                                        shuffle      = True
                                    )      

#Se comienza el entrenamiento y se pone como ejemplo a un random_forest
#por defecto. Por supuesto esto se puede enriquecer mucho más con un grid-search
#adecuado

rf = RandomForestRegressor(random_state=0)

#Evaluación sencilla del modelo en el test

def rf_method(model, x_train, y_train, x_test, y_test):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    ResidualSquare = (result - y_test)**2     #Calcular el cuadrado residual
    RSS = sum(ResidualSquare)   #Calcular la suma de cuadrados residual
    MSE = np.mean(ResidualSquare)       #Calcular el error cuadrático medio
    num_regress = len(result)   # Número de muestras de regresión
    print(f'n={num_regress}')
    print(f'R^2={score}')
    print(f'MSE={MSE}')
    print(f'RSS={RSS}')
    
    ############ Dibujar un gráfico de líneas ##########
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('RandomForestRegression R^2: %f'%score)
    plt.legend()        # Mostrar la muestra
    plt.show()
    return result

_ = rf_method(rf, X_train2, y_train2, X_test2, y_test2)


#Se re-entrena el modelo con todos los datos dado que parece aceptable

rf = RandomForestRegressor(random_state=0)
rf.fit(df_train_t.drop('target', 
                       axis = 'columns'),
       df_train_t['target'])



#Se comprueba ahora el modelo en la realidad del test de la serie temporal

#Se toma como primer input los últimos 6 datos de la serie temporal (los más)
#actuales posibles


pred1 = df_train_t.loc[len(df_train_t['target']) - 1].to_frame().T
pred1 = pred1.drop('var_6', axis = 1)
pred1.columns = ['var_6', 'var_5', 'var_4', 'var_3', 'var_2', 'var_1']
res1 = rf.predict(pred1)

res1 = np.exp(np.log(best_lambda * res1 + 1) / best_lambda)

eam1 = abs(res1 - df_test[df_test['fch'] == '2019M01']['pernoctaciones'])/df_test[df_test['fch'] == '2019M01']['pernoctaciones']
eam1

pred2 = pred1[['var_5', 'var_4', 'var_3', 'var_2', 'var_1']]
pred2['target'] = rf.predict(pred1) 
pred2.columns = ['var_6', 'var_5', 'var_4', 'var_3', 'var_2', 'var_1']
res2 = rf.predict(pred2)
res2 = np.exp(np.log(best_lambda * res2 + 1) / best_lambda)

eam2 = abs(res2 - df_test[df_test['fch'] == '2019M02']['pernoctaciones'])/df_test[df_test['fch'] == '2019M02']['pernoctaciones']
eam2

pred3 = pred2[['var_5', 'var_4', 'var_3', 'var_2', 'var_1']]
pred3['target'] = rf.predict(pred2) 
pred3.columns = ['var_6', 'var_5', 'var_4', 'var_3', 'var_2', 'var_1']
res3 = rf.predict(pred3)
res3 = np.exp(np.log(best_lambda * res3 + 1) / best_lambda)

eam3 = abs(res3 - df_test[df_test['fch'] == '2019M03']['pernoctaciones'])/df_test[df_test['fch'] == '2019M03']['pernoctaciones']
eam3


pred4 = pred3[['var_5', 'var_4', 'var_3', 'var_2', 'var_1']]
pred4['target'] = rf.predict(pred3) 
pred4.columns = ['var_6', 'var_5', 'var_4', 'var_3', 'var_2', 'var_1']
res4 = rf.predict(pred4)
res4 = np.exp(np.log(best_lambda * res4 + 1) / best_lambda)

eam4 = abs(res4 - df_test[df_test['fch'] == '2019M04']['pernoctaciones'])/df_test[df_test['fch'] == '2019M04']['pernoctaciones']
eam4










