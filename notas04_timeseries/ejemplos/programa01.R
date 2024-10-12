#par(mfrow = c(1,1))

#Se observa el gráfico de la serie
plot.ts(AirPassengers)

#Función para descomponer una serie en sus componentes
AirPassengerscomponents <- decompose(AirPassengers)
plot(AirPassengerscomponents)

serie <- AirPassengerscomponents$random
write.csv(serie, "serie.csv")

#Ajuste estacional: Sólo tiene la componente tendencial y aleatoria
AirPassengersseasonallyadjusted <- AirPassengers - AirPassengerscomponents$seasonal
plot(AirPassengersseasonallyadjusted)

#Ajuste de la Tendencia por regresion lineal simple:
t = 1:length((AirPassengers))
t2 = t**0.5
AirPassengersModel <- lm(AirPassengersseasonallyadjusted ~ t)
summary(AirPassengersModel)

#Previsión de la Tendencia
library(forecast)

AirPassengersforecasts2 <- AirPassengersModel$coefficients[1] + 
  AirPassengersModel$coefficients[2] * c(145:(144+12))
plot(AirPassengersforecasts2)

#Previsiones total: Se suma la parte tendencial
AirPassengersforecasts3 <- AirPassengersforecasts2 + AirPassengerscomponents$seasonal[1:12]
AirPassengersforecasts3 <- ts(AirPassengersforecasts3, start=1961,frequency=12)

#Representación final
vector1 <- as.vector(AirPassengers)
vector2 <- as.vector(AirPassengersforecasts3)
vectorTotal <- c(vector1, vector2)
AirPassengersTotal <- ts(vectorTotal, start = 1949, frequency = 12)

plot.ts(AirPassengersTotal, type = "l", col = "red", xlim = c(1949, 1962))
par(new=TRUE)
plot.ts(AirPassengers, type = "l", col = "blue", xlim = c(1949, 1962))



################################################
############Modelo Multiplicativo###############
################################################

#Tendencia
  #Se toma un modelo linenal
  Modelo1 <- lm(as.vector(AirPassengers)~c(1:144))
  trend_air <- Modelo1$fitted.values

detrend_air =  AirPassengers / trend_air
plot(as.ts(detrend_air))

#Estacional
