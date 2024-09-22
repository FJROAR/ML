#http://apuntes-r.blogspot.com.es/2014/09/predecir-perdida-de-clientes-con-arbol.html

library(C50)
library(rpart)
library(rpart.plot) 

# carga tablas
churn <- read.csv("data/Telco-Customer-Churn.csv")

#Se elimina la variable customerID
churn <- churn[,-1]

set.seed(1)
index<-sample(c(1:nrow(churn)), 0.7*nrow(churn))


#Da como resultados 2 conjuntos de datos de 20 variables:

churnTrain <- churn[index,]
churnTest <- churn[-index,]


#Modelización

ModeloArbol<-rpart(Churn ~ .,data=churnTrain,parms=list(split="information"))

#El árbol resultaba demasiado complejo: Poda

plotcp(ModeloArbol)
printcp(ModeloArbol)
rpart.plot(ModeloArbol)


#Se toma un árbol más simple

ModeloArbol2 <- rpart(Churn ~ .,data=churnTrain,parms=list(split="information"),
                      control = rpart.control(cp = 0.035, minsplit = 8))

rpart.plot(ModeloArbol2)


########################################################################
#############Fiabilidad por Validación Cruzada: MODELO 1################
########################################################################

set.seed(1)
Folds         <- 10            
datos <- churnTrain
datos$kfold   <- sample(1:Folds, nrow(datos), replace = T)

#An?lisis de la Validaci?n Cruzada

Iter   <- data.frame(iteracion = NULL, aciertos = NULL)
for (i in 1:Folds)
{
  Test          <- subset(datos, kfold  == i)
  Entrenamiento <- subset(datos, !kfold == i) 
  Modelo        <- ModeloArbol       
  Prediccion    <- predict(Modelo, Test, type = "class")  
  MC            <- table(Test[, "Churn"],Prediccion)           
  Aciertos      <- MC[1, 1] / (MC[1, 1] + MC[2, 1])
  Iter          <- rbind(Iter, data.frame(Iter = i, acierto = Aciertos))  
}

#Grafico

promedio  <- format(mean(Iter$acierto, na.rm=TRUE)*100,digits = 4)
plot(Iter,type = "b", main = "% Prediccion en Cada Iteracion",  
     cex.axis = .7,cex.lab = .7,cex.main = .8, 
     xlab ="No. de Iteraciones", ylab="% Prediccion")
abline(h = mean(Iter$acierto), col = "blue", lty = 2)
legend("topright", legend = paste("Eficiencia de Prediccion =", promedio, "%"),
       col = "blue", lty = 2, lwd = 1, cex=.7, bg=NULL)

promedio
sd(Iter$acierto)


########################################################################
#############Fiabilidad por Validación Cruzada: MODELO 2################
########################################################################

set.seed(1)
Folds         <- 10            
datos <- churnTrain
datos$kfold   <- sample(1:Folds, nrow(datos), replace = T)

#An?lisis de la Validaci?n Cruzada

Iter   <- data.frame(iteracion = NULL, aciertos = NULL)
for (i in 1:Folds)
{
  Test          <- subset(datos, kfold  == i)
  Entrenamiento <- subset(datos, !kfold == i) 
  Modelo        <- ModeloArbol2       
  Prediccion    <- predict(Modelo, Test, type = "class")  
  MC            <- table(Test[, "Churn"],Prediccion)           
  Aciertos      <- MC[1, 1] / (MC[1, 1] + MC[2, 1])
  Iter          <- rbind(Iter, data.frame(Iter = i, acierto = Aciertos))  
}

#Grafico

promedio  <- format(mean(Iter$acierto, na.rm=TRUE)*100,digits = 4)
plot(Iter,type = "b", main = "% Prediccion en Cada Iteracion",  
     cex.axis = .7,cex.lab = .7,cex.main = .8, 
     xlab ="No. de Iteraciones", ylab="% Prediccion")
abline(h = mean(Iter$acierto), col = "blue", lty = 2)
legend("topright", legend = paste("Eficiencia de Prediccion =", promedio, "%"),
       col = "blue", lty = 2, lwd = 1, cex=.7, bg=NULL)

promedio
sd(Iter$acierto)