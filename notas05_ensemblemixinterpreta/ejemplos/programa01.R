#Librerías para modelizar y dibujar Decision y Regression Trees
library(rpart)
library(rpart.plot)


#Librería que contiene el conjunto de datos
library (MASS)

head(Boston)
help(Boston)
#La semilla se elije para repetir la separación training-validation
set.seed (1)
train = sample (1: nrow(Boston ), nrow(Boston )/2)
test = -train

#Comprobación de las medias de respuesta explicada en train y test. No hecho en ISLH

mean(Boston$medv[train])
sd(Boston$medv[train])

mean(Boston$medv[test])
sd(Boston$medv[test])

hist(Boston$medv[train])
hist(Boston$medv[test])

#Se observa bastante parecido y cabría aplicar un anova para demostrar que no hay diferencia
#significativa


#Se construye un árbol de regresión Simple
tree_boston = rpart(medv ~.,Boston ,subset =train)

rpart.plot(tree_boston, extra = 1)