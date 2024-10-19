ventasDiarias <- read.csv("data\\ventasDiarias.csv",
         sep = ",")

#Ajuste venta 20051225
auxiliar1 <- ventasDiarias[which(ventasDiarias$Dia == 20051218),]
auxiliar2 <- ventasDiarias[which(ventasDiarias$Dia == 20051211),]
auxiliar3 <- ventasDiarias[which(ventasDiarias$Dia == 20051204),]

Venta = mean(c(auxiliar1$Venta, auxiliar2$Venta, auxiliar3$Venta))

Dia1 <- data.frame(20051225, Venta)
names(Dia1) <- c("Dia", "Venta")

#Ajuste venta 20060101
auxiliar1 <- Dia1
auxiliar2 <- ventasDiarias[which(ventasDiarias$Dia == 20051218),]
auxiliar3 <- ventasDiarias[which(ventasDiarias$Dia == 20051211),]

Venta = mean(c(auxiliar1$Venta, auxiliar2$Venta, auxiliar3$Venta))

Dia2 <- data.frame(20060101, Venta)
names(Dia2) <- c("Dia", "Venta")

#Ajuste venta S?bado Santo 2006 que cay? el 15 de Abril
auxiliar1 <- ventasDiarias[which(ventasDiarias$Dia == 20060408),]
auxiliar2 <- ventasDiarias[which(ventasDiarias$Dia == 20060401),]
auxiliar3 <- ventasDiarias[which(ventasDiarias$Dia == 20060325),]

Venta = mean(c(auxiliar1$Venta, auxiliar2$Venta, auxiliar3$Venta))

Dia3 <- data.frame(20060415, Venta)
names(Dia3) <- c("Dia", "Venta")

#Se introducen los "d?as ficticios" en la serie temporal y se ordenan
#los datos

ventasDiarias <- rbind(ventasDiarias, Dia1, Dia2, Dia3)
ventasDiarias <- ventasDiarias[order(ventasDiarias$Dia),]

#Comprobaci?n de que todo est? bien
nrow(ventasDiarias)
ventasDiarias[which(ventasDiarias$Dia == 20051225),]
ventasDiarias[which(ventasDiarias$Dia == 20060101),]
ventasDiarias[which(ventasDiarias$Dia == 20060415),]

#Construcci?n de un objeto series temporales
sventasDiarias <- ts(ventasDiarias$Venta, frequency = 7)
plot.ts(sventasDiarias)

#Se guarda externamente el resultado para continuarlo en la siguiente sesi?n
write.csv(sventasDiarias,"DATA\\sventasDiarias.csv",
          row.names = FALSE)
