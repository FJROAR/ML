#Librer?as para modelizar y dibujar Decision y Regression Trees
library(rpart)
library(rpart.plot)


#Librería que contiene el conjunto de datos
library (MASS)

head(Boston)

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

#Se observa bastante parecido y cabr?a aplicar un anova para demostrar que no hay diferencia
#significativa


#Se construye un ?rbol de regresión Simple
tree_boston = rpart(medv~.,Boston ,subset =train)

rpart.plot(tree_boston, extra = 1)


#Se construye un árbol de regresión con modificaciones
tree_boston2 = rpart(medv~.,Boston ,subset =train,
                     control = rpart.control(minsplit = 50, minbucket = 25, cp = 0.05))

rpart.plot(tree_boston2, extra = 1)


#Retoque mejora del 0.02
tree_boston3 = rpart(medv~.,Boston ,subset =train,
                     control = rpart.control(minsplit = 50, minbucket = 25, cp = 0.02))
rpart.plot(tree_boston3, extra = 1)

#Validación

predicciones <- predict(tree_boston, Boston[test,])
Real <-Boston[test,"medv"]
sum((predicciones - Real)**2)**0.5

predicciones <- predict(tree_boston2, Boston[test,])
Real <-Boston[test,"medv"]
sum((predicciones - Real)**2)**0.5

predicciones <- predict(tree_boston3, Boston[test,])
Real <-Boston[test,"medv"]
sum((predicciones - Real)**2)**0.5


