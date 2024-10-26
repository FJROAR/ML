# Instalar paquetes si no están instalados
# install.packages("tidyverse")
# install.packages("modeltime")
# install.packages("xgboost")
# install.packages("lubridate")
# install.packages("timetk")


library(tidyverse)
library(modeltime)
library(xgboost)
library(lubridate)
library(timetk)
library(dplyr)
library(parsnip) 

# Cargar los datos
data("AirPassengers")

# Convertir a dataframe
ap_df <- as.data.frame(AirPassengers)

# Crear un tibble con un índice de tiempo
ap_df <- tibble::tibble(
  date = seq(from = as.Date("1949-01-01"), 
             by = "month", 
             length.out = length(AirPassengers)),
  passengers = as.numeric(AirPassengers)
)

# Añadir características adicionales (mes, año)
ap_df <- ap_df %>%
  mutate(month = month(date, label = TRUE),  # Añadir mes como factor
         year = year(date)) %>%
  select(-date)  

# Crear una nueva columna para el retardo de 12 meses
ap_df <- ap_df %>%
  mutate(lag_12 = lag(passengers, 12))

# Eliminar filas con NA (debido a la creación de la columna de retardo)
ap_df <- na.omit(ap_df)


# Eliminamos la columna de fecha para el modelo
# Dividir los datos
train_size <- floor(0.8 * nrow(ap_df))
train_data <- ap_df[1:train_size, ]
test_data <- ap_df[(train_size + 1):nrow(ap_df), ]

set.seed(155)

# Crear el modelo XGBoost
xgb_model  <- 
  boost_tree(
    trees = 1000,
    learn_rate = 0.05,
    min_n = 2,
    tree_depth = 5,
    loss_reduction = 0.01,
    sample_size = 0.8,
    stop_iter = 100
  ) %>%
  set_engine("xgboost") %>%
  set_mode("regression") %>%
  fit(passengers ~ ., data = train_data)


# Realizar predicciones
predictions <- xgb_model %>%
  predict(new_data = test_data) %>%
  bind_cols(test_data)

# Calcular el MAPE
mape <- mean(abs((predictions$.pred - predictions$passengers) / predictions$passengers)) * 100

# Imprimir el MAPE
cat("MAPE:", mape, "%\n")

# Graficar los resultados
ggplot() +
  geom_line(data = train_data, aes(x = seq_along(passengers), y = passengers), color = "blue", size = 1) +
  geom_line(data = predictions, aes(x = seq_along(.pred) + train_size, y = .pred), color = "red", size = 1) +
  labs(title = "Predicción de Pasajeros de Aerolínea con XGB",
       x = "Observación",
       y = "Número de Pasajeros") +
  theme_minimal()


# Graficar los resultados
ggplot() +
  geom_line(data = train_data, aes(x = seq_along(passengers), y = passengers), color = "blue", size = 1, alpha = 0.7, linetype = "dashed") +  # Línea de entrenamiento
  geom_line(data = test_data, aes(x = seq_along(passengers) + nrow(train_data), y = passengers), color = "green", size = 1, alpha = 0.7) +  # Línea verde para los datos reales del test
  geom_line(data = predictions, aes(x = seq_along(.pred) + nrow(train_data), y = .pred), color = "red", size = 1, alpha = 0.7) +  # Línea de predicciones
  labs(title = "Predicción de Pasajeros de Aerolínea con XGBoost",
       x = "Observaciones",
       y = "Número de Pasajeros") +
  theme_minimal() +
  scale_x_continuous(breaks = seq(1, nrow(train_data) + nrow(test_data), by = 12), labels = function(x) as.character(as.Date(x, origin = "1970-01-01")))  # Mejora en la escala del eje x