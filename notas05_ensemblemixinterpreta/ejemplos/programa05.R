library (randomForest)
library(rpart)
library(rpart.plot) 
library(MASS)
library(TH.data)

data(GlaucomaM, package = "TH.data" )

set.seed(10)
train <- sample(1:nrow(GlaucomaM), size = 0.7 * nrow(GlaucomaM))

#Se observa que aproximadamente hay una proporcion 50% - 50%
summary(GlaucomaM$Class[train])
summary(GlaucomaM$Class[-train])


#Con mtry = 62 se informa que las 62 variables explicativas deben ser consideradas
set.seed (1)
gbag =randomForest(Class ~ .,data = GlaucomaM ,subset = train ,
                           mtry=62, importance =TRUE)
pred_gbag = predict (gbag , newdata = GlaucomaM[-train,])

table(GlaucomaM[-train, "Class"], pred_gbag)


#Comparación con un decision tree simple
gtree = rpart(Class ~ .,data = GlaucomaM ,subset = train)

pred_gtree = predict (gtree , newdata = GlaucomaM[-train,], type = "class")

table(GlaucomaM[-train, "Class"], pred_gtree)

rpart.plot(gtree, extra = 1)