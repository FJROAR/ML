#Se va a usar el ets de R, que es un m?todo de selecci?n
#de series temporales en R
#acorde a lo expuesto en http://robjhyndman.com/hyndsight/dailydata/
library(forecast)
library(tseries)
library(MASS)


sventasDiarias <- read.csv("D:/FJRA/FORMACION/MASTER_UIC_2024/04TimeSeries/programas/data/sventasDiarias.csv",
         sep = ",")

names(sventasDiarias) <- "Venta"

res_box <- boxcox(lm(sventasDiarias$Venta ~ 1))
lambda <- res_box$x[which.max(res_box$y)]
lambda

sventasDiarias$Venta <- (sventasDiarias$Venta^lambda - 1) / lambda

#Predicción con un ARIMA(0, 1, 0) X (1, 1, 2) 7


plot.ts(sventasDiarias$Venta)
plot.ts(diff(diff(sventasDiarias$Venta, 1, 7), 1))


help("diff")

fit <- Arima(ts(sventasDiarias$Venta, frequency = 7), order = c(1,1,1), seasonal = c(0,1,1),
              include.mean = FALSE)
fc <- forecast(fit)
plot(fc)

pred01 <- (fc$mean * lambda + 1)^(1 / lambda)



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

Erroresbox = mean(abs((pred01 - ventasReales$Venta)/ventasReales$Venta))
Erroresbox

acf(fit$residuals)
pacf(fit$residuals)