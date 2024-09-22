w1 = seq(-10,10,length=50)
w2 = w1

#Punto 1 (3; 2; 1)
#Punto 2 (1; 4; 1)
#Punto 3 (-1; 1; 0)
#Punto 4 (-1; -1; 0)
#Punto 5 (2; -1; 0)

#Definimos la función que dibujaremos
f1 = function(w1,w2) 
  {
  (1 - 1/(1+exp(2*w1 + w2*3)))^2+ 
     (1 - 1/(1+exp(w1 + w2*4)))^2 + 
     (0 - 1/(1+exp((-1)*w1 + w2)))^2 +
     (0 - 1/(1+exp((-1)*w1 + (-1)*w2)))^2 + 
     (0 - 1/(1+exp(2*w1 + (-1)*w2)))^2
  }   
 
z = outer(w1,w2,f1)                             
persp(w1,w2,z, theta = 30, phi = 30, expand = 0.5, col = "lightblue")
