#¿Qué ocurre aquí?
df = data.frame(V = c(0,0,1,1,0), H = c(1,1,0,0,1), Y = c(1,0,0,1,0))

modelo <- glm(Y ~., data = df, family = "binomial")

modelo$coefficients


#Coherencia de R con una única variable

df = data.frame(V = c(0,0,1,1,0), Y = c(1,0,0,1,0))

modelo <- glm(Y ~., data = df, family = "binomial")

modelo$coefficients

V = c(0,0,1,1,0)

#Con los coeficientes de R

exp(modelo$coefficients[1] + modelo$coefficients[2] * V) / 
  (1 + exp(modelo$coefficients[1]+ modelo$coefficients[2] * V))

predict.glm(modelo, data.frame(V), type = "response")
