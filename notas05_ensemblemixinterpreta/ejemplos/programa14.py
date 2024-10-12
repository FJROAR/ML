#Llamadas a las librerías básica y se importa sólo una parte de ella
from sklearn.model_selection import train_test_split

#Llamadas a las librerías básica y se importa sólo una parte de ella
from sklearn.ensemble import RandomForestClassifier

#Se traem imps datos pre-existesntes
from sklearn.datasets import make_moons


#Se traen una parte de los datasets separando entre variables explicativas y explicadas
X, y = make_moons(n_samples = 100, noise = 0.25, random_state = 3)

#Se realiza la típica separación training-test para modelizar. Segmentando por la variable explicada
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, 
                                                    random_state = 42)

#Se ajusta un modelo con los datos del training
forest = RandomForestClassifier(n_estimators = 5, random_state = 2)
forest.fit(X_train, y_train)

#Se pueden observar los árboles seleccionados
forest.estimators_

#Análisis por áboles
from sklearn import tree

Modelo0 = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, 
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, 
             random_state=1872583848, splitter='best')

Modelo1 = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, 
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, 
             random_state=794921487, splitter='best')

Modelo2 = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, 
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, 
             random_state=111352301, splitter='best')

Modelo3 = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, 
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, 
             random_state=1853453896, splitter='best')

Modelo4 = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
             max_features='sqrt', max_leaf_nodes=None,
             min_impurity_decrease=0.0, 
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, 
             random_state=213298710, splitter='best')

Ajuste0 = Modelo0.fit(X_train, y_train)
Ajuste1 = Modelo1.fit(X_train, y_train)
Ajuste2 = Modelo2.fit(X_train, y_train)
Ajuste3 = Modelo3.fit(X_train, y_train)
Ajuste4 = Modelo4.fit(X_train, y_train)

p0 = Ajuste0.predict_proba([[1.8, 0.5]])
p1 = Ajuste1.predict_proba([[1.8, 0.5]])
p2 = Ajuste2.predict_proba([[1.8, 0.5]])
p3 = Ajuste3.predict_proba([[1.8, 0.5]])
p4 = Ajuste4.predict_proba([[1.8, 0.5]])

p = forest.predict_proba([[1.8, 0.5]])

p_comprueba = (1/5)*(p0 + p1 + p2 + p3 + p4)








#Se dibuja el rf

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mglearn
import matplotlib.pyplot as plt


X, y = make_moons(n_samples = 100, noise = 0.25, random_state = 3)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify = y,
                                                    random_state = 42)

forest = RandomForestClassifier(n_estimators = 5 ,random_state = 2)
forest.fit(X_train, y_train)
    
#Se dibujan cada uno de los árboles por separado que constituyen el rf
fig, axes = plt.subplots(2, 3, figsize = (20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax = ax)
    
#Se dibuja el random forest
mglearn.plots.plot_2d_separator(forest,
                                X_train, 
                                fill = True,
                                ax = axes[-1, -1],
                                alpha = 0.4)
axes[-1, -1].set_title("Random Fores")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)