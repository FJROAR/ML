library(TSP)

matriz_dist <- structure(c(0, Inf, Inf, 1.9, 1.7, Inf, 0, 7.3, 7.4, 7.2, Inf, 
7.3, 0, 7.7, 7.8, 1.9, 7.4, 7.7, 0, 9.2, 1.7, 7.2, 7.8, 9.2, 
0), .Dim = c(5L, 5L), .Dimnames = list(c("2", "13", "14", "17", 
"20"), c("2", "13", "14", "17", "20")))

matriz_dist_tsp <- as.TSP(matriz_dist)
ruta <- solve_TSP(matriz_dist_tsp, method = "nearest_insertion", start = 5)
ruta
labels(ruta)

#Nótese la siguiente igualdad y que lo que se genera es una ruta cerrada
#de coste mínimo

7.2 + 7.3 + 7.7 + 1.9 + 1.7 = 25.8