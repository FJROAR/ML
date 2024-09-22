###########################################################################################################
#                                                                                                         #
#Fuente Original: https://www.r-bloggers.com/monty-hall-by-simulation-in-r/                               #
#                                                                                                         #
#Se modifica el programa original para hacer simulaciones de Monte Carlo sobre el problema de Monty Hall. #
#La simulacion tendra 1000 intentos                                                                       #
#El objetivo es demostrar que adoptando la estrategia 'cambiar' se obtiene                                #
#una mayor probabilidad de ganar que adoptando la estrategia 'quedarse'                                   #
#                                                                                                         #
###########################################################################################################

montyhall <- function(estrategia='quedarse', N=1000, print_games=TRUE)
{
  puertas=1:3
  ganar=0 # hacer un seguimiento del numero de exitos.
  for(i in 1:N) 
  {
    premio=floor(runif(n=1,min=1,max=4)) # aleatoriamente qu? puerta tiene el buen premio.
    adivinar=floor(runif(n=1,min=1,max=4)) # adivinar una puerta al azar.
    ### Revelar una de las puertas que no tiene por supuesto el buen premio.
    if(premio!=adivinar)
      revelar=puertas[-c(premio,adivinar)]
    else
      revelar=sample(puertas[-c(premio,adivinar)],1)
    ### 'Quedarse' con la eleccion inicial o 'cambiar'.
    if(estrategia=='cambiar')
      seleccionar=puertas[-c(revelar,adivinar)]
    if(estrategia=='quedarse')
      seleccionar=adivinar
    if(estrategia=='aleatorio')
      seleccionar=sample(puertas[-revelar],1)
    ### Cuente sus exitos
    if(seleccionar==premio)
    {
      ganar=ganar+1
      resultado='ganador'
    }else
      resultado='perdedor'
    if(print_games)
      cat(paste('adivinar: ',adivinar,
                '\nPuerta Revelada: ',revelar,
                '\nPuerta Elegida: ',seleccionar,
                '\nPuerta del premio: ',premio,
                '\n',resultado,'\n\n',sep=''))
  }
  cat(paste(' Usando la estrategia ',estrategia,' su porcentaje de ganar era de ', ganar/N*100,'%\n',sep=' '))
  
  return(ganar/N*100)
}


#Llamadas simples a la Funcion

montyhall(estrategia="quedarse")
montyhall(estrategia="cambiar")
montyhall(estrategia="aleatorio")


#Simulacion de 1000 escenarios de la funcion anterior y analisis de su variabilidad
#Se analizan los casos de No Cambio (quedarse) y de Cambio (cambiar)

distribucion_quedarse <- vector("numeric", 100)

for(i in (1:100)){
  
  distribucion_quedarse[i] <- montyhall(estrategia="quedarse")
  
}

hist(distribucion_quedarse)
mean(distribucion_quedarse)
sd (distribucion_quedarse)

distribucion_cambiar <- vector("numeric", 100)

for(i in (1:100)){
  
  distribucion_cambiar[i] <- montyhall(estrategia="cambiar")
  
}

hist(distribucion_cambiar)
mean(distribucion_cambiar)
sd (distribucion_cambiar)
