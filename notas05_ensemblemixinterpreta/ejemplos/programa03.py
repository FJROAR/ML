#https://www.datasklr.com/tree-based-methods-for-regression/regression-trees

import pandas as pd

from sklearn.tree import DecisionTreeRegressor  
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import cross_val_score  
from sklearn.metrics import mean_squared_error

#from sklearn.datasets import load_boston
#boston= load_boston()

from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True)

housing_features_df = pd.DataFrame(data = housing.data,
                                  columns = housing.feature_names)


#Se considera unas pocas variables numéricas (las de tipo char habría que
#trabajarlas aparte)

variables = ["MSSubClass", "LotFrontage", "LotArea", "OverallQual", "YearBuilt",
             "YearRemodAdd", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
             "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
             "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
             "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", 
             "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
             "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
             "MoSold", "YrSold"]

housing_features_df = housing_features_df[variables]


housing_features_df.columns

#Los missing se ponen a 0 para que no causen problemas Ojo con esto!!!!!!!!!

housing_features_df = housing_features_df.fillna(0)

#Se aisla la target
housing_target_df = pd.DataFrame(data=housing.target,
                                 columns=['SalePrice'])

#Se crea los df de entrenamiento y validación en este caso al 80 - 20
X_train, X_test, Y_train, Y_test = train_test_split(housing_features_df, 
                                                    housing_target_df, 
                                                    test_size = 0.20, 
                                                    random_state = 155)
X_train.head(3)


#Se ajusta un primer modelo con toda la información posible

model1 = DecisionTreeRegressor(random_state = 155)

model1.fit(X_train, 
           Y_train)

predicted = model1.predict(X_test)


#Predict the response for test dataset
y_pred = model1.predict(X_test)
y_pred



import matplotlib.pyplot as plt
from sklearn.tree import plot_tree




plt.figure(figsize=(20,10))
plot_tree(model1, feature_names=variables, filled=True, rounded=True)
plt.show()

# create a regressor object 
model2 = DecisionTreeRegressor(random_state = 155, 
                                  max_depth = 4,
                                  min_samples_leaf = 25)  

# fit the regressor with X_tra and y_ar data 
model2.fit(X_train, 
           Y_train)

plt.figure(figsize=(20,10))
plot_tree(model2, 
          feature_names = variables, 
          filled = True, 
          rounded = True)
plt.show()
