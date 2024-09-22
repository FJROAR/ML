rm(list=ls())

# Load the libraries
library(arules)
library(arulesViz)

#library(datasets) En la url se explica que al conjunto de datos se accede mediante data(Groceries)
#Es más instructivo acceder a un fichero externo que es más parecido al caso real

# Load the data set

Groceries <- read.transactions("data/groceries.csv", sep ="," )

#Análisis simple del conjunto de datos leído
size(Groceries)
dim(Groceries)
#image(Groceries)
itemFrequencyPlot(Groceries,topN=20,type="absolute")

#Se establece un support m?nimo de 0.001 o 0.1%
#Se establece un confidence m?nimo de 0.8 o del 80%

# Obtención de las reglas
rules <- apriori(Groceries, parameter = list(supp = 0.001, conf = 0.8, target = "rules"))
help("apriori")
# 5 reglas principales a 2 dígitos
options(digits=2)
inspect(rules[1:5])

#Información resumen de la forma de las reglas de asociación
summary(rules)


#Análisis de la información en un BMA

#(1) Ordenación por confidence
rules<-sort(rules,--------------)
inspect(rules[1:5])

#(2) Se acota la longitud de las reglas a 3
rules <- apriori(Groceries, parameter = list(supp = 0.001, conf = 0.8, -------))
rules<-sort(rules, by="confidence", decreasing=TRUE)
inspect(rules[1:5])

#(3) Teorema de Bayes

#Los que compraron leche que otras cosas adquirieron antes

rules<-apriori(data=Groceries, parameter=list(supp=0.001,conf = 0.8), 
               appearance = list(default="lhs",rhs="whole milk"),
               control = list(verbose=F))
rules<-sort(rules, by="confidence", decreasing=TRUE)
inspect(rules[1:5])

#Los que compraron leche que otras cosas adquieren despues

rules<-apriori(data=Groceries, parameter=list(supp=0.001,conf = 0.15,minlen=2), 
               appearance = list(default="rhs",lhs="whole milk"),
               control = list(verbose=F))
rules<-sort(rules, decreasing=TRUE,by="confidence")
inspect(rules[1:5])

#(4) Visualization

plot(--------------)