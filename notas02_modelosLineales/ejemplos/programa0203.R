#Los datos vienen de: http://www.theanalysisfactor.com/r-tutorial-4/

A <- structure(list(Time = c(0, 1, 2, 4, 6, 8, 9, 10, 11, 12, 13, 
                             14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30), 
                    Counts = c(126.6, 101.8, 71.6, 101.6, 68.1, 62.9, 45.5, 41.9, 
                               46.3, 34.1, 38.2, 41.7, 24.7, 41.5, 36.6, 19.6, 
                               22.8, 29.6, 23.5, 15.3, 13.4, 26.8, 9.8, 18.8, 25.9, 19.3)), .Names = c("Time", "Counts"),
               row.names = c(1L, 2L, 3L, 5L, 7L, 9L, 10L, 11L, 12L, 13L, 14L, 15L, 16L, 17L, 19L, 20L, 21L, 22L, 23L, 25L, 26L, 27L, 28L, 29L, 30L, 31L),
               class = "data.frame")

#Modelo simple

Modelo1 <- lm(Counts~Time, data = A)
summary(Modelo1)


#Modelo cuadrado

A$Time2 <- A$Time**2

Modelo2 <- lm(Counts~Time + Time2, data = A)
summary(Modelo2)

#Modelo orden 10

A$Time3 <- A$Time**3
A$Time4 <- A$Time**4
A$Time5 <- A$Time**5
A$Time6 <- A$Time**6
A$Time7 <- A$Time**7
A$Time8 <- A$Time**8
A$Time9 <- A$Time**9
A$Time10 <- A$Time**10


Modelo3 <- lm(Counts~Time + Time2 + Time3 + Time4 + Time5 + Time6 + 
                Time7  + Time8 + Time9 + Time10, data = A)
summary(Modelo3)


#Plotting


#Modelo1
plot(A$Time, A$Counts)
abline(lm(A$Counts ~ A$Time), col = "darkgreen", lwd = 3)

#Modelo2
plot(A$Time, A$Counts)
Model20Eq = Modelo2$coefficients[1] + Modelo2$coefficients[2] * A$Time +
  Modelo2$coefficients[3] * A$Time2
lines(A$Time, Model20Eq, col = "darkgreen", lwd = 3)

#Modelo3
plot(A$Time, A$Counts)

Model10Eq = Modelo3$coefficients[1] + Modelo3$coefficients[2] * A$Time +
  Modelo3$coefficients[3] * A$Time2 + Modelo3$coefficients[4] * A$Time3 +
  Modelo3$coefficients[5] * A$Time4 + Modelo3$coefficients[6] * A$Time5 +
  Modelo3$coefficients[7] * A$Time6 + Modelo3$coefficients[8] * A$Time7 +
  Modelo3$coefficients[9] * A$Time8 + Modelo3$coefficients[10] * A$Time9 +
  Modelo3$coefficients[11] * A$Time10
  
lines(A$Time, Model10Eq, col = "darkgreen", lwd = 3)