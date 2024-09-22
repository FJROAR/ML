#https://medium.com/@brijesh_soni/stacking-to-improve-model-performance-a-comprehensive-guide-on-ensemble-learning-in-python-9ed53c93ce28

import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load the Boston Housing dataset
boston = fetch_openml(name='boston')


#Primera partición de datos
X_train1, X_val1, y_train1, y_val1 = train_test_split(boston.data, 
                                                  boston.target, 
                                                  test_size=0.1, 
                                                  random_state=155)


# Segunda partición de datos
X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train1, 
                                                  y_train1, 
                                                  test_size=0.2, 
                                                  random_state=42)

#Se entrenan los modelos base

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Train the base models
dt = DecisionTreeRegressor(random_state=155)
dt.fit(X_train2, y_train2)

rf = RandomForestRegressor(random_state=155)
rf.fit(X_train2, y_train2)

gb = GradientBoostingRegressor(random_state=155)
gb.fit(X_train2, y_train2)


#Se predice con los anteriores modelos
dt_pred = dt.predict(X_val2)
rf_pred = rf.predict(X_val2)
gb_pred = gb.predict(X_val2)


#Se entrena el metamodelo

from sklearn.linear_model import LinearRegression

# Combine the predictions of the base models into a single feature matrix
X_val_meta = np.column_stack((dt_pred, rf_pred, gb_pred))

# Train the meta-model on the combined feature matrix and the target values
meta_model = LinearRegression()
meta_model.fit(X_val_meta, y_val2)

#Se testea el resultado
# Make predictions on new data
X_new = X_val1
dt_pred_new = dt.predict(X_new)
rf_pred_new = rf.predict(X_new)
gb_pred_new = gb.predict(X_new)

# Combine the predictions of the base models into a single feature matrix
X_new_meta = np.column_stack((dt_pred_new, rf_pred_new, gb_pred_new))

# Make a prediction using the meta-model
y_new_pred = meta_model.predict(X_new_meta)


eamr = np.mean(np.abs(y_new_pred - y_val1)) / np.mean(y_val1)


#Comparación con un random forest (tuve algún problema con la regresión)

rf = RandomForestRegressor()
rf.fit(X_train1, y_train1)

y_rf_pred = rf.predict(X_val1)

eamr2 = np.mean(np.abs(y_rf_pred - y_val1)) / np.mean(y_val1)
