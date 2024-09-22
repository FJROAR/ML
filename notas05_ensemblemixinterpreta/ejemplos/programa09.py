#Llamada a librerías

import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier


#Llamada a los conjuntos de datos
training_data = pd.read_csv("data/titanic/train.csv")
testing_data = pd.read_csv("data/titanic/test.csv")

#Tratamiento de variables. Análisis de valores nulos
def get_nulls(training, testing):
    print("Training Data:")
    print(pd.isnull(training).sum())
    print("Testing Data:")
    print(pd.isnull(testing).sum())

get_nulls(training_data, testing_data)

#Hay algunos valores nulos que van a tener que se tratados, en algunos casos, 
#como Cabin, la variable será directamente eliminada y la de Ticket
#por otro lado la edad será imputada en sus valores nulos con la mediana (al haber
#cierta asimetría en los datos)


training_data.drop(labels=['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
testing_data.drop(labels=['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

training_data["Age"].fillna(training_data["Age"].median(), inplace=True)
testing_data["Age"].fillna(testing_data["Age"].median(), inplace=True)
training_data["Embarked"].fillna("S", inplace=True)
testing_data["Fare"].fillna(testing_data["Fare"].median(), inplace=True)


#Se comprueba que el dataset está bien
get_nulls(training_data, testing_data)


#En este ejemplo se hace un escalado de las observaciones

encoder_1 = LabelEncoder()
# Fit the encoder on the data
encoder_1.fit(training_data["Sex"])

# Transform and replace training data
training_sex_encoded = encoder_1.transform(training_data["Sex"])
training_data["Sex"] = training_sex_encoded
test_sex_encoded = encoder_1.transform(testing_data["Sex"])
testing_data["Sex"] = test_sex_encoded

encoder_2 = LabelEncoder()
encoder_2.fit(training_data["Embarked"])

training_embarked_encoded = encoder_2.transform(training_data["Embarked"])
training_data["Embarked"] = training_embarked_encoded
testing_embarked_encoded = encoder_2.transform(testing_data["Embarked"])
testing_data["Embarked"] = testing_embarked_encoded

# Any value we want to reshape needs be turned into array first
ages_train = np.array(training_data["Age"]).reshape(-1, 1)
fares_train = np.array(training_data["Fare"]).reshape(-1, 1)
ages_test = np.array(testing_data["Age"]).reshape(-1, 1)
fares_test = np.array(testing_data["Fare"]).reshape(-1, 1)

# Scaler takes arrays
scaler = StandardScaler()

training_data["Age"] = scaler.fit_transform(ages_train)
training_data["Fare"] = scaler.fit_transform(fares_train)
testing_data["Age"] = scaler.fit_transform(ages_test)
testing_data["Fare"] = scaler.fit_transform(fares_test)


#Construcción del tranining y test

X_features = training_data.drop(labels=['PassengerId', 'Survived'], axis=1)
y_labels = training_data['Survived']

print(X_features.head(5))

# Make the train/test data from validation

X_train, X_val, y_train, y_val = train_test_split(X_features, 
                                                  y_labels, 
                                                  test_size=0.1, 
                                                  random_state = 155)



#Se comienza a modelizar

#Ensembling promedio básico

LogReg_clf = LogisticRegression()
DTree_clf = DecisionTreeClassifier()
SVC_clf = SVC()

LogReg_clf.fit(X_train, y_train)
DTree_clf.fit(X_train, y_train)
SVC_clf.fit(X_train, y_train)

LogReg_pred = LogReg_clf.predict(X_val)
DTree_pred = DTree_clf.predict(X_val)
SVC_pred = SVC_clf.predict(X_val)

averaged_preds = (LogReg_pred + DTree_pred + SVC_pred)//3
acc = accuracy_score(y_val, averaged_preds)
print(acc)



#Voting classifiers / stacking

voting_clf = VotingClassifier(estimators=[('SVC', SVC_clf), 
                                          ('DTree', DTree_clf), 
                                          ('LogReg', LogReg_clf)], 
                              voting='hard')
voting_clf.fit(X_train, y_train)
preds = voting_clf.predict(X_val)
acc = accuracy_score(y_val, preds)
l_loss = log_loss(y_val, preds)
f1 = f1_score(y_val, preds)

print("Accuracy is: " + str(acc))
print("Log Loss is: " + str(l_loss))
print("F1 Score is: " + str(f1))


#Ejemplo Bagging

logreg_bagging_model = BaggingClassifier(base_estimator=LogReg_clf, 
                                         n_estimators=50,
                                         random_state=155)
dtree_bagging_model = BaggingClassifier(base_estimator=DTree_clf, 
                                        n_estimators=50,
                                        random_state=155)
random_forest = RandomForestClassifier(n_estimators=100,
                                       random_state=155)
extra_trees = ExtraTreesClassifier(n_estimators=100,
                                   random_state=155)

def bagging_ensemble(model):
    k_folds = KFold(n_splits=20, random_state=155, shuffle=True)
    results = cross_val_score(model, X_train, y_train, cv=k_folds)
    print(results.mean())

bagging_ensemble(logreg_bagging_model)
bagging_ensemble(dtree_bagging_model)
bagging_ensemble(random_forest)
bagging_ensemble(extra_trees)



#Ejemplo boosting 
k_folds = KFold(n_splits=20, random_state=155, shuffle=True)

num_estimators = [20, 40, 60, 80, 100]

for i in num_estimators:
    ada_boost = AdaBoostClassifier(n_estimators=i, random_state=155)
    results = cross_val_score(ada_boost, X_train, y_train, cv=k_folds)
    print("Results for {} estimators:".format(i))
    print(results.mean())