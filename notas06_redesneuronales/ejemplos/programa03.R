library(neuralnet)

#Trabla de verdad de la función XOR
x1 = c(1, 1, 0, 0)
x2 = c(1, 0, 1, 0)
y = c(0, 1, 1, 0)

X = data.frame(x1, x2, y)

#Modelo neuronal de una única capa (entradas - salida)
set.seed(1234)
nn1 <- neuralnet(y ~ x1 + x2, 
                hidden = 0, 
                data = X)

plot(nn1)

#Resultado del aprendizaje

Pred = X[1,]
X[1,]
predict(nn1, Pred)

Pred = X[2,]
X[2,]
predict(nn1, Pred)

Pred = X[3,]
X[3,]
predict(nn1, Pred)

Pred = X[4,]
X[4,]
predict(nn1, Pred)

#Modelo neuronal de 3 capas (con una capa oculta de neuronas interconectadas)
set.seed(1234)
nn2 <- neuralnet(y ~ x1 + x2, 
                 hidden = 2, 
                 data = X)

plot(nn2)

#Resultado del aprendizaje

Pred = X[1,]
X[1,]
predict(nn2, Pred)

Pred = X[2,]
X[2,]
predict(nn2, Pred)

Pred = X[3,]
X[3,]
predict(nn2, Pred)

Pred = X[4,]
X[4,]
predict(nn2, Pred)



#Anexo: Entrenando con una capa la función AND

#Trabla de verdad de la función XOR
x1 = c(1, 1, 0, 0)
x2 = c(1, 0, 1, 0)
y = c(1, 0, 0, 0)

X2 = data.frame(x1, x2, y)

#Modelo neuronal de una única capa (entradas - salida)
set.seed(1234)
nn3 <- neuralnet(y ~ x1 + x2, 
                 hidden = 0, 
                 data = X2)

plot(nn3)

#Resultado del aprendizaje

Pred = X2[1,]
X2[1,]
predict(nn3, Pred)

Pred = X2[2,]
X2[2,]
predict(nn3, Pred)

Pred = X2[3,]
X2[3,]
predict(nn3, Pred)

Pred = X2[4,]
X2[4,]
predict(nn3, Pred)
