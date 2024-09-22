library(markovchain)
library(diagram)

set.seed(1)

States <- c("Rainy","Cloudy","Sunny")

TransMat <- matrix(c(0.30,0.50,0.20,0.25,0.4,0.35,0.1,0.2,0.70),
                   nrow = 3, 
                   byrow= TRUE,
                   dimnames = list(States,States))

TransMat

#Se crea el modelo de cadena de Markov
MarkovChainModel <- new("markovchain",
                        transitionMatrix=TransMat,
                        states=States, 
                        byrow = TRUE, 
                        name="MarkovChainModel")

MarkovChainModel

plot(MarkovChainModel,package="diagram")

#Predicción a 3 días (suponiendo hoy soleado)

StartState<-c(0,0,1)

Pred3Days <- StartState * (MarkovChainModel ^ 3)
print (round(Pred3Days, 3))


#Predicción a 1 semana

Pred1Week <- StartState * (MarkovChainModel ^ 7)
print (round(Pred1Week, 3))


#Distribución estacionaria: convergencia general del proceso de markov planteado
steadyStates(MarkovChainModel)


#Predicción a 1 año
YearWeatherState <- rmarkovchain(n = 365, 
                                 object = MarkovChainModel, 
                                 t0 = "Sunny")
