library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(dplyr)
library(MLmetrics)


#Lectura y tratamiento general de la Informaci?n

RUTA <- "data/"
DATASET <- "train.csv"
DATASET2 <- "test.csv"

DATOS <- fread(paste0(RUTA, DATASET), stringsAsFactors = FALSE, sep = ",", na.strings = "")
#DATOS2 <- fread(paste0(RUTA, DATASET2), stringsAsFactors = FALSE, sep = ",", na.strings = "")

set.seed(155)
training_index  <- sample(1:nrow(DATOS), size = round(0.6 * nrow(DATOS)))
test_index <- -training_index

TRAINING <- DATOS  
TRAINING2 <- DATOS[training_index,]  
TEST2 <- DATOS[test_index,]  
targetIni <- TEST2$target
#TEST <- DATOS2 

set.seed(155)

xgb_normalizedgini <- function(preds, dtrain){
  actual <- getinfo(dtrain, "label")
  score <- NormalizedGini(preds,actual)
  return(list(metric = "NormalizedGini", value = score))
}

TEST2$target <- NA
data <- as.data.table(rbind(TRAINING, TEST2))
gc()


#Construcción de Nuevas Variables
data[, amount_nas := rowSums(data == -1, na.rm = T)]
data[, ps_car_13_ps_reg_03 := ps_car_13*ps_reg_03]
data[, ps_reg_mult := ps_reg_01*ps_reg_02*ps_reg_03]
data[, ps_ind_bin_sum := ps_ind_06_bin+ps_ind_07_bin+ps_ind_08_bin+ps_ind_09_bin+ps_ind_10_bin+ps_ind_11_bin+ps_ind_12_bin+ps_ind_13_bin+ps_ind_16_bin+ps_ind_17_bin+ps_ind_18_bin]
data[, ps_reg_01_cuad := ps_reg_01 * ps_reg_01]

data[, ps_car_13_sqrt := ps_car_13**0.5]
data[, ps_reg_03_sqrt := ps_reg_03**0.5]

data <- data[, -"ps_ind_11_bin"]
data <- data[, -"ps_ind_13_bin"]
data <- data[, -"ps_ind_14"]
data <- data[, -"ps_calc_16_bin"]


#Modelización: xgboost hace uso de matrices de aquí el código siguiente

cvFolds <- createFolds(data$target[!is.na(data$target)], k=5, list=TRUE, returnTrain=FALSE)
varnames <- setdiff(colnames(data), c("id", "target"))
train_sparse <- Matrix(as.matrix(data[!is.na(target), varnames, with=F]), sparse=TRUE)
test_sparse <- Matrix(as.matrix(data[is.na(target), varnames, with=F]), sparse=TRUE)
y_train <- data[!is.na(target),target]
test_ids <- data[is.na(target),id]
dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
dtest <- xgb.DMatrix(data=test_sparse)

#Parametrización del modelo 
#Sacada mediante Validacicón Cruzada
#Se hace uso de distinta prueba y error

param <- list(booster="gbtree",
              objective="binary:logistic",
              eta = 0.025,
              gamma = 1,
              max_depth = 5,
              min_child_weight = 1,
              subsample = 0.8,
              colsample_bytree = 0.8)


best_iter <- 250 
best_iter <- 30
  

#Se entrena el modelo

set.seed(155)  
xgb_model <- xgb.train(data = dtrain,
                     params = param,
                     nrounds = best_iter,
                     feval = xgb_normalizedgini,
                     maximize = TRUE,
                     watchlist = list(train = dtrain),
                     verbose = 1,
                     print_every_n = 25)


#Análisis de Importancia de las Variables

names <- dimnames(train_sparse)[[2]]
importance_matrix <- xgb.importance(names, model=xgb_model)
xgb.plot.importance(importance_matrix)



#Se optienen las predicciones

cat("Predict and output csv")
preds <- data.table(id=test_ids, target=predict(xgb_model,dtest))
#write.table(preds, paste0(RUTA, "submission.csv"), sep=",", dec=".", quote=FALSE, row.names=FALSE)


#C?lculo de AUC

category <- targetIni
prediction <- preds$target
length(targetIni)
length(prediction)

library(pROC)
roc_obj <- roc(category, prediction)
auc(roc_obj)


#KAGGLE COMPARA LAS PREDICCIONES CON LOS DATOS REALES