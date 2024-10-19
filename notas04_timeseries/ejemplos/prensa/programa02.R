#http://rprogramming.net/connect-to-ms-access-in-r/
library(RODBC)

con <- odbcConnectAccess("DATA/REPOSITORIO_PRENSA.mdb")

#Se lee la tabla Estad?sticas con una query adecuada, aqu? podr?a leerse entera, pero en un servidor real
#esa operaci?n "nos tostar?a" el disco duro

Estadisticas <- sqlQuery(con, paste0("select * from Estadisticas where Codpubli >= 101001 and Codpubli <= 101007 ",
                                     "and Numportada >= 20051001 and Numportada <= 20061001"))

#Se cierra la conexi?n a la base de datos y se trabaja en R
odbcClose(con)

#Se construye una tabla con la columna Numportada, para deducir un dato
#fecha con otra columna venta para agregar los datos

Venta = Estadisticas$Normal + Estadisticas$Reposicion + Estadisticas$Cargos - 
  Estadisticas$Devolucion - Estadisticas$Abonos

ventasDiarias <- data.frame(Estadisticas$Numportada, Venta)
names(ventasDiarias) <- c("Numportada", "Venta")
head(ventasDiarias)

#Se agregan datos por la variable Numportada

ventasDiarias <- aggregate(ventasDiarias$Venta, by = list(ventasDiarias$Numportada), sum)
names(ventasDiarias) <- c("Dia", "Venta")
head(ventasDiarias)

#Por si acaso, se ordenan los datos por Dia
ventasDiarias <- ventasDiarias[order(ventasDiarias$Dia),]


###########################################################
###########################################################

#Estad?sticos simples

n_Datos = nrow(ventasDiarias)
media <- mean(ventasDiarias$Venta)
desviacion <- sd(ventasDiarias$Venta)
maximo <- max(ventasDiarias$Venta)
minimmo <- min(ventasDiarias$Venta)
hist(ventasDiarias$Venta)

x = c(1:nrow(ventasDiarias))
plot(x, ventasDiarias$Venta, "line")


#¿Qué son los picos? ?Cómo evoluciona en general el nivel de venta?
#?Hay comportamientos estacionales?

#Cuestión importante, según n_Datos hay 363 días
#Desde el 1 de Octubre del 2005 hasta el 1 de Octubre del 2006
#deber?a haber 366 datos, sin embargo hay s?lo 363 ?A qu? puede ser 
#debido?

#Se guarda la serie temporal para su utilizaci?n en los posteriores
#proyectos

write.csv(ventasDiarias ,"DATA\\ventasDiarias.csv", 
          row.names = FALSE)