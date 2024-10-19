#Se va a usar el ets de R, que es un m?todo de selecci?n
#de series temporales en R
#acorde a lo expuesto en http://robjhyndman.com/hyndsight/dailydata/
library(forecast)
library(tseries)
library(MASS)


sventasDiarias <- read.csv("D:/FJRA/FORMACION/MASTER_UIC_2024/04TimeSeries/programas/data/sventasDiarias.csv",
         sep = ",")

names(sventasDiarias) <- "Venta"


#Predicción con un ARIMA(0, 1, 0) X (1, 1, 2) 7

fit1 <- Arima(ts(sventasDiarias$Venta, frequency = 7), order = c(0,1,0), seasonal = c(1,1,2),
              include.mean = FALSE)
fc <- forecast(fit1)

plot(fc)

#Opción aún más sencilla
library(forecast)
fit2 <- ets(ts(sventasDiarias$Venta, frequency = 7))
fc2 <- forecast(fit2)
plot(fc2)
fit2

#Opción automática
fit3 <- auto.arima(ts(sventasDiarias$Venta, frequency = 7))
fc3 <- forecast(fit3)
plot(fc3)
fit3


#Se guardan los datos de predicciones
predicciones <- fc$mean
predicciones2 <- fc2$mean
predicciones3 <- fc3$mean

#Validación y comparación de modelos

library(RODBC)

con <- odbcConnectAccess("D:/FJRA/FORMACION/MASTER_UIC_2024/04TimeSeries/programas/data/REPOSITORIO_PRENSA.mdb")

Estadisticas <- sqlQuery(con, paste0("select * from Estadisticas 
                                     where 
                                      Codpubli >= 101001 and 
                                      Codpubli <= 101007 
                                      and Numportada >= 20061002 
                                      and Numportada <= 20061015"))

Venta <- Estadisticas$Normal + 
  Estadisticas$Reposicion + 
  Estadisticas$Cargos - 
  Estadisticas$Devolucion - Estadisticas$Abonos

ventasReales <- data.frame(Estadisticas$Numportada, Venta)
names(ventasReales) <- c("Numportada", "Venta")
ventasReales <- aggregate(ventasReales$Venta, 
                          by = list(ventasReales$Numportada), sum)
names(ventasReales) <- c("Dia", "Venta")

odbcClose(con)

#Errores teórico - real

Errores = abs((predicciones - ventasReales$Venta)/ventasReales$Venta)
Errores2 = abs((predicciones2 - ventasReales$Venta)/ventasReales$Venta)
Errores3 = abs((predicciones3 - ventasReales$Venta)/ventasReales$Venta)

mean(Errores)
mean(Errores2)
mean(Errores3)

acf(fit1$residuals)
pacf(fit1$residuals)

acf(fit2$residuals)
pacf(fit2$residuals)

acf(fit3$residuals)
pacf(fit3$residuals)


MeanPred = 0.333*(predicciones + predicciones2 + predicciones3)
ErroresMP = abs((MeanPred - ventasReales$Venta)/ventasReales$Venta)
mean(ErroresMP)



ventasDiarias <- read.csv("D:/FJRA/FORMACION/MASTER_UIC_2024/04TimeSeries/programas/data/ventasDiarias.csv",
                           sep = ",")



library(prophet)

fecha <- paste0(substr(ventasDiarias$Dia,1, 4), "-",
                substr(ventasDiarias$Dia,5, 6), "-",
                substr(ventasDiarias$Dia,7, 8))

aux <- as.Date(fecha, format ="%Y-%m-%d")

df = data.frame(ds = aux ,y = ventasDiarias$Venta)

mod <- prophet(changepoint.prior.scale= 6,
                changepoint.range=0.9)
mod <- add_seasonality(mod, name='monthly', 
                        period=30.5, 
                        fourier.order=12)
mod <- add_seasonality(mod, name='daily', 
                        period=1,
                        fourier.order=15)
mod <- add_seasonality(mod, name='weeKly', 
                        period=7,
                        fourier.order=20)
mod <- add_seasonality(mod, name='quarterly', 
                        period=365.25/4,
                        fourier.order=5,
                        prior.scale=15)

mod <- fit.prophet(mod, df)
#mod <- fit.prophet(mod, df)

future <- make_future_dataframe(mod, periods = 14)

forecast <- predict(mod, future)

pred_prophet <- forecast$yhat[ c((length(forecast$yhat) - 13):length(forecast$yhat))]

Erroresph = abs((pred_prophet - ventasReales$Venta)/ventasReales$Venta)
mean(Erroresph)

MeanPredMix = 0.5*(predicciones2 + pred_prophet)
ErroresMPMix = abs((MeanPredMix - ventasReales$Venta)/ventasReales$Venta)
mean(ErroresMPMix)
