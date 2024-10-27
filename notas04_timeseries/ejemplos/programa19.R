library(bsts)

set.seed(123)  # Para reproducibilidad

# Crear una tendencia lineal
n <- 100  # Número de observaciones
t <- 1:n
trend <- 0.1 * t

# Crear un componente estacional (sinusoidal)
seasonal <- 2 * sin(2 * pi * t / 12)

# Agregar ruido aleatorio
noise <- rnorm(n, mean = 0, sd = 0.5)

# Serie temporal simulada
y <- trend + seasonal + noise
time_series <- ts(y, frequency = 12)

# Definir el componente de tendencia local y el componente estacional
ss <- AddLocalLinearTrend(list(), time_series)
ss <- AddSeasonal(ss, time_series, nseasons = 12)

# Ajustar el modelo
model <- bsts(time_series, state.specification = ss, niter = 500)

# Graficar la serie temporal original y los componentes del modelo
plot(model, main = "Modelo BSTS: Serie Temporal Simulada")

# Graficar las predicciones y su intervalo de confianza
pred <- predict(model, horizon = 12)  # Predicción para los próximos 12 periodos
plot(pred, main = "Predicción del modelo BSTS")
