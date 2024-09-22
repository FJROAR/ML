#https://www.r-bloggers.com/improve-predictive-performance-in-r-with-bagging/

#Se simulan datos donde combinaciones de las variables explicativas 
#permitirán predecir la variable explicada.

#Las variables explicativas son la variable explicada con una perturbación aleatoria

set.seed(10)
y<-c(1:1000)
x1<-c(1:1000)*runif(1000,min=0,max=2)
x2<-c(1:1000)*runif(1000,min=0,max=2)
x3<-c(1:1000)*runif(1000,min=0,max=2)



#Construcción de un conjunto test y validaci?n

all_data<-data.frame(y,x1,x2,x3)
positions <- sample(nrow(all_data),size=floor((nrow(all_data)/4)*3))
training<- all_data[positions,]
testing<- all_data[-positions,]


#Se ajusta un modelo de regresión lineal

lm_fit<-lm(y~x1+x2+x3,data=training)
predictions<-predict(lm_fit,newdata=testing)
error<-sqrt((sum((testing$y-predictions)^2))/nrow(testing))
error

#Aplicación de un bagging

bagging<-function(training,testing,length_divisor=4,iterations=1000)
{
  library(foreach)
  predictions<-foreach(m=1:iterations,.combine=cbind) %do% {
    training_positions <- sample(nrow(training), size=floor((nrow(training)/length_divisor)))
    train_pos<-1:nrow(training) %in% training_positions
    lm_fit<-lm(y~x1+x2+x3,data=training[train_pos,])
    predict(lm_fit,newdata=testing)
  }

  predictions<-rowMeans(predictions)
  error<-sqrt((sum((testing$y-predictions)^2))/nrow(testing))
  
  Result = list(predictions, error)
  
  return(Result)
}

bagging(training,testing,length_divisor=4,iterations=10)

bagging(training,testing,length_divisor=4,iterations=50)

bagging(training,testing,length_divisor=4,iterations=100)

bagging(training,testing,length_divisor=4,iterations=500)

bagging(training,testing,length_divisor=4,iterations=1000)

bagging(training,testing,length_divisor=4,iterations=5000)

bagging(training,testing,length_divisor=4,iterations=10000)

