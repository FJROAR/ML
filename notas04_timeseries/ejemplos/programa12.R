# Cargar librerías
library(rugarch)
library(dplyr)
library(timetk)
library(tidyr)
library(ggplot2)

# Cargar y preparar datos de EuStockMarkets
data("EuStockMarkets")  
stock_data <- as.data.frame(EuStockMarkets) %>%
  mutate(date = seq.Date(from = as.Date("1991-01-01"), by = "day", length.out = nrow(.))) %>%
  rename(daily_returns = DAX)  
stock_data <- stock_data %>%
  mutate(daily_returns = (daily_returns / lag(daily_returns) - 1)) %>%
  drop_na()

# Definir horizonte de predicción de 3 días y crear el marco de datos extendido
horizon <- 3
stock_data_extended <- stock_data %>%
  future_frame(.length_out = horizon, .bind_data = TRUE)

# Separar en conjuntos de entrenamiento y futuro
train_data <- stock_data_extended %>% drop_na()
future_data <- stock_data_extended %>% filter(is.na(daily_returns))

# Definir el modelo GARCH(1, 1)
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "norm"
)

# Entrenar el modelo con datos de entrenamiento
fit_garch <- ugarchfit(spec = spec, data = train_data$daily_returns)

# Extraer desviación estándar condicional de entrenamiento
cond_sd <- sigma(fit_garch)  
train_dates <- train_data$date
train_plot_data <- data.frame(
  date = train_dates,  
  cond_sd = cond_sd,
  type = "Observado"
)

# Predecir para el horizonte de 3 días
forecast <- ugarchforecast(fit_garch, n.ahead = horizon)
predicted_sigma <- sigma(forecast)

# Crear marco de datos con las predicciones y fechas extendidas
future_dates <- seq.Date(from = max(train_dates) + 1, by = "day", length.out = horizon)
future_plot_data <- data.frame(
  date = future_dates,
  cond_sd = predicted_sigma,
  type = "Predicción"
)

# Combinar datos observados y predicción en un solo dataframe
combined_plot_data <- bind_rows(train_plot_data, future_plot_data)

# Graficar la desviación estándar condicional con ggplot2
ggplot(combined_plot_data, aes(x = date, y = cond_sd, color = type)) +
  geom_line(size = 1) +
  labs(title = "Desviación Estándar Condicional (Modelo GARCH) con Predicción a 3 días",
       x = "Fecha",
       y = "Desviación Estándar Condicional") +
  theme_minimal() +
  scale_color_manual(values = c("Observado" = "blue", "Predicción" = "red"))