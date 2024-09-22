w1 = seq(0,0.5,length=50)
w2 = w1

#Punto 1 (3; 2; 1)
#Punto 2 (1; 4; 1)
#Punto 3 (-1; 1; 0)
#Punto 4 (-1; -1; 0)
#Punto 5 (2; -1; 0)

#Definimos la función (en este caso cuadrática) que dibujaremos
f1 = function(w1,w2) 
  {
  
  (1 - (2*w1 + w2*3))^2+ 
    (1 -(w1 + w2*4))^2 + 
    (0 - ((-1)*w1 + w2))^2 +
    (0 - ((-1)*w1 + (-1)*w2))^2 + 
    (0 - (2*w1 + (-1)*w2))^2
  
  }   
 
z = outer(w1,w2,f1)                             

persp(w1,w2,z, theta = 30, phi = 30, expand = 0.5, col = "lightblue")  


#Optimización, buscar el peso mínimo en estos casos

fopt1 <- function(x) 
{
  
  w1 <- x[1]
  w2 <- x[2]
  
  (1 - (2*w1 + w2*3))^2+ 
    (1 -(w1 + w2*4))^2 + 
    (0 - ((-1)*w1 + w2))^2 +
    (0 - ((-1)*w1 + (-1)*w2))^2 + 
    (0 - (2*w1 + (-1)*w2))^2
  
}

optim(c(0,0), fopt1)


#Ejercicio: Introducir un t?rmino independiente


fopt1 <- function(x) 

{
  
  w1 <- x[1]
  w2 <- x[2]
  w0 <- x[3]
  
  (1 - (2*w1 + w2*3 + w0))^2+ 
    (1 -(w1 + w2*4 + w0))^2 + 
    (0 - ((-1)*w1 + w2 + w0))^2 +
    (0 - ((-1)*w1 + (-1)*w2 + w0))^2 + 
    (0 - (2*w1 + (-1)*w2 + w0))^2
  
}

optim(c(0,0,0), fopt1)


