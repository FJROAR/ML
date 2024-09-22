#Paper Hopfield Networks

library(png)
library(grDevices)

source("hopfield_tools/hopfield.R")

weights = matrix(c(0, -1, 1, 1,-1, 0, -1, -1, 1, -1, 0, 1, 1, -1, 1, 0), 4, 4)
hopnet = list(weights = weights)

#init.y = c(-1, -1, -1, 1)
init.y = c(-1, -1, -1, -1)
init.y = c(-1, 1, -1, -1)
init.y = c(-1, -1, 1, -1) #Inestable
init.y = c(-1, -1, 1, 1) #Estable a 1 -1 1 1
init.y = c(1, -1, 1, 1) #Estable a 1 -1 1 1
init.y = c(1, 1, 1, 1) #Estable a 1 -1 1 1

for (i in 1:20){

  print(run.hopfield(hopnet, init.y, maxit = 50, stepbystep=T, topo=c(4,1)))
  
}


