import pandas as pd
#import numpy as np

dataset = pd.read_csv('data/Boston.csv', sep = ',')
dataset.head()
dataset.columns

X = dataset[['CRIM', 
             'ZN', 
             'INDUS', 
             'CHAS', 
             'NOX', 
             'RM', 
             'AGE', 
             'DIS', 
             'RAD', 
             'TAX',
             'PTRATIO', 
             'B', 
             'LSTAT']].values

y = dataset[['MEDV']].values.ravel()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.25,
                                                    random_state= 155)


from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes= (10, 10), 
                   max_iter = 3000,
                   activation = 'relu',
                   solver='adam',
                   random_state=123)

mlp.fit(X_train, y_train) 

mlp.score(X_test,y_test)


#Caso regresión lineal

from sklearn.linear_model import LinearRegression

rl = LinearRegression()

rl.fit(X_train, y_train)

rl.score(X_test, y_test)