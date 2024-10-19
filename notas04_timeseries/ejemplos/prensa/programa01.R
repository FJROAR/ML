#http://rprogramming.net/connect-to-ms-access-in-r/
library(RODBC)

con <- odbcConnectAccess("D:/FJRA/FORMACION/MASTER_UIC_2024/04TimeSeries/programas/data/REPOSITORIO_PRENSA.mdb")
sqlTables(con)

#Se lee la tabla Estad?sticas con una query adecuada, aqu? podr?a leerse entera, pero en un servidor real
#esa operación "nos tostaría" el disco duro

Estadisticas <- sqlQuery(con, 
                         paste0("select * from Estadisticas where",
                                " Codpubli >= 101001 and",
                                " Codpubli <= 101007 and",
                                " Numportada >= 20051001 and",
                                " Numportada <= 20061001")
                         )

#Muestra de datos y nombres de campos
head(Estadisticas)

#Se cierra la conexión a la base de datos y se trabaja en R
odbcClose(con)

#Se construye una tabla con la columna Numportada, para deducir un dato
#fecha con otra columna venta para agregar los datos

Venta = Estadisticas$Normal + 
  Estadisticas$Reposicion + 
  Estadisticas$Cargos - 
  Estadisticas$Devolucion - 
  Estadisticas$Abonos

ventasDiarias <- data.frame(Estadisticas$Numportada, Venta)
names(ventasDiarias) <- c("Numportada", "Venta")
head(ventasDiarias)

#Se agregan datos por la variable Numportada

ventasDiarias <- aggregate(ventasDiarias$Venta, by = list(ventasDiarias$Numportada), sum)
names(ventasDiarias) <- c("Dia", "Venta")
head(ventasDiarias)

#Por si acaso, se ordenan los datos por Dia
ventasDiarias <- ventasDiarias[order(ventasDiarias$Dia),]
