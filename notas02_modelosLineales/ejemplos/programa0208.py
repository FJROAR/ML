#https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch06_Analyzing_Bike_Sharing_Trends/decision_tree_regression.py


# data manuipulation
import numpy as np
import pandas as pd

# modeling utilities
#import pydotplus 
from sklearn.tree import _tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# plotting libraries
import seaborn as sn
import matplotlib.pyplot as plt


sn.set_style('whitegrid')
sn.set_context('talk')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)


#Se lee el dataset

hour_df = pd.read_csv('data/hour.csv')
print("Shape of dataset::{}".format(hour_df.shape))


#Manipulación de variables: Tratamiento de fechas

hour_df.rename(columns={'instant':'rec_id',
                        'dteday':'datetime',
                        'holiday':'is_holiday',
                        'workingday':'is_workingday',
                        'weathersit':'weather_condition',
                        'hum':'humidity',
                        'mnth':'month',
                        'cnt':'total_count',
                        'hr':'hour',
                        'yr':'year'},inplace=True)


# date time conversion
hour_df['datetime'] = pd.to_datetime(hour_df.datetime)

# categorical variables
hour_df['season'] = hour_df.season.astype('category')
hour_df['is_holiday'] = hour_df.is_holiday.astype('category')
hour_df['weekday'] = hour_df.weekday.astype('category')
hour_df['weather_condition'] = hour_df.weather_condition.astype('category')
hour_df['is_workingday'] = hour_df.is_workingday.astype('category')
hour_df['month'] = hour_df.month.astype('category')
hour_df['year'] = hour_df.year.astype('category')
hour_df['hour'] = hour_df.hour.astype('category')


#Encoding de variables categóricas (estas funciones pueden ir por fuera)

#Func 1

def fit_transform_ohe(df,col_name):
    """This function performs one hot encoding for the specified
        column.
    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        col_name: the column to be one hot encoded
    Returns:
        tuple: label_encoder, one_hot_encoder, transformed column as pandas Series
    """
    # label encode the column
    le = preprocessing.LabelEncoder()
    le_labels = le.fit_transform(df[col_name])
    df[col_name+'_label'] = le_labels
    
    # one hot encoding
    ohe = preprocessing.OneHotEncoder()
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    
    return le,ohe,features_df


def transform_ohe(df,le,ohe,col_name):
    """This function performs one hot encoding for the specified
        column using the specified encoder objects.
    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        le(Label Encoder): the label encoder object used to fit label encoding
        ohe(One Hot Encoder): the onen hot encoder object used to fit one hot encoding
        col_name: the column to be one hot encoded
    Returns:
        tuple: transformed column as pandas Series
    """
    # label encode
    col_labels = le.transform(df[col_name])
    df[col_name+'_label'] = col_labels
    
    # ohe 
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    
    return features_df


#Separación training - test

X, X_test, y, y_test = train_test_split(hour_df.iloc[:,0:-3], hour_df.iloc[:,-1], 
                                                    test_size=0.33, random_state=42)

X.reset_index(inplace=True)
y = y.reset_index()

X_test.reset_index(inplace=True)
y_test = y_test.reset_index()

print("Training set::{}{}".format(X.shape,y.shape))
print("Testing set::{}".format(X_test.shape))


#Tratamiento desde el training de las categórica. Best practice!!!


cat_attr_list = ['season','is_holiday',
                 'weather_condition','is_workingday',
                 'hour','weekday','month','year']
numeric_feature_cols = ['temp','humidity','windspeed','hour','weekday','month','year']
subset_cat_features =  ['season','is_holiday','weather_condition','is_workingday']


encoded_attr_list = []


for col in cat_attr_list:
    return_obj = fit_transform_ohe(X,col)
    encoded_attr_list.append({'label_enc':return_obj[0],
                              'ohe_enc':return_obj[1],
                              'feature_df':return_obj[2],
                              'col_name':col})


feature_df_list = [X[numeric_feature_cols]]

feature_df_list.extend([enc['feature_df']                         
                        for enc in encoded_attr_list
                         if enc['col_name'] in subset_cat_features])


train_df_new = pd.concat(feature_df_list, axis=1)
print("Shape::{}".format(train_df_new.shape))

#Se entrena el modelo (se preparan la variable explicada y las explicativas)

X = train_df_new
y= y.total_count.values.reshape(-1,1)


#Función para dibujar las reglas de un árbol
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature] 
    print ("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print ("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)



#Se toma un dataset de una única variable (por ejemplo temp)
X2 = X[['temp']]

#Se ajusta un árbol para un único corte
dtr2 = DecisionTreeRegressor(max_depth=1)
dtr2.fit(X2, y)

tree_to_code(dtr2, X2.columns)

corte = dtr2.tree_.threshold[0]

#Comprobación de los valores medios de la variable y

np.mean(y[X2['temp'] <= corte])
np.mean(y[X2['temp'] > corte])

abs(np.mean(y[X2['temp'] <= corte]) - np.mean(y[X2['temp'] > corte]))

len(y[X2['temp'] <= corte])
len(y[X2['temp'] > corte])


#¿Qué ocurre si se cambia el punto de corte?

corte2 = 0.5
np.mean(y[X2['temp'] <= corte2])
np.mean(y[X2['temp'] > corte2])

abs(np.mean(y[X2['temp'] <= corte2]) - np.mean(y[X2['temp'] > corte2]))

len(y[X2['temp'] <= corte2])
len(y[X2['temp'] > corte2])



corte2 = 0.2
np.mean(y[X2['temp'] <= corte2])
np.mean(y[X2['temp'] > corte2])

abs(np.mean(y[X2['temp'] <= corte2]) - np.mean(y[X2['temp'] > corte2]))

len(y[X2['temp'] <= corte2])
len(y[X2['temp'] > corte2])


#Función de Error con una única variable


def errores(corte):
    error = np.sum((y[X2['temp'] <= corte] - 
                    np.mean(y[X2['temp'] <= corte]))**2) + np.sum((y[X2['temp'] > corte] - 
                                                                   np.mean(y[X2['temp'] > corte]))**2)
    return error


errores(0.5)
errores(0.48999999463558197)
errores(0.2)

valor_x = np.array(np.arange(0, 1000, 1)) / 1000

valor_y = [errores(s) for s in valor_x]

plt.plot(valor_x, valor_y)

plt.show()


################################################################################
#PARTE II Construcción de un Regression Tree al completo
################################################################################
dtr = DecisionTreeRegressor()

param_grid = {"min_samples_split": [10, 20, 40],
              "max_depth": [2, 6],
              "min_samples_leaf": [20, 40],
              "max_leaf_nodes": [20, 100, 500, 800],
              }

grid_cv_dtr = GridSearchCV(dtr, param_grid, cv=4)

grid_cv_dtr.fit(X,y)

print("R-Squared::{}".format(grid_cv_dtr.best_score_))
print("Best Hyperparameters::\n{}".format(grid_cv_dtr.best_params_))


df = pd.DataFrame(data=grid_cv_dtr.cv_results_)
df.head()

fig,ax = plt.subplots()
sn.pointplot(data=df[['mean_test_score',
                           'param_max_leaf_nodes',
                           'param_max_depth']],
             y='mean_test_score',x='param_max_depth',
             hue='param_max_leaf_nodes',ax=ax)
ax.set(title="Effect of Depth and Leaf Nodes on Model Performance")

plt.show()

predicted = grid_cv_dtr.best_estimator_.predict(X)
residuals = y.flatten()-predicted


r2_scores = cross_val_score(grid_cv_dtr.best_estimator_, X, y, cv=10)
mse_scores = cross_val_score(grid_cv_dtr.best_estimator_, X, y, cv=10,scoring='neg_mean_squared_error')

print("avg R-squared::{}".format(np.mean(r2_scores)))
print("MSE::{}".format(np.mean(mse_scores)))

best_dtr_model = grid_cv_dtr.best_estimator_



#Actuación en el test

#Se prepara el test como se hizo con el training
test_encoded_attr_list = []
for enc in encoded_attr_list:
    col_name = enc['col_name']
    le = enc['label_enc']
    ohe = enc['ohe_enc']
    test_encoded_attr_list.append({'feature_df':transform_ohe(X_test,le,ohe,col_name),
                                   'col_name':col_name})
    
    
test_feature_df_list = [X_test[numeric_feature_cols]]
test_feature_df_list.extend([enc['feature_df'] for enc in test_encoded_attr_list if enc['col_name'] in subset_cat_features])

test_df_new = pd.concat(test_feature_df_list, axis=1) 
print("Shape::{}".format(test_df_new.shape))

X_test = test_df_new
y_test = y_test.total_count.values.reshape(-1,1)

#Predicción en el test
y_pred = best_dtr_model.predict(X_test)
residuals = y_test.flatten() - y_pred

r2_score = best_dtr_model.score(X_test,y_test)
print("R-squared::{}".format(r2_score))
print("MSE: %.2f"
      % metrics.mean_squared_error(y_test, y_pred))



r2_score = grid_cv_dtr.best_estimator_.score(X_test,y_test)


#Reglas del árbol finalmente seleccionado
tree_to_code(best_dtr_model, X.columns)
