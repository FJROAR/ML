#devtools::install_github("twitter/AnomalyDetection", force = TRUE)
library(AnomalyDetection)
library(ggplot2)
library(dplyr)

data("raw_data")

# Convertir ambos `timestamp` a `POSIXct`
raw_data$timestamp <- as.POSIXct(raw_data$timestamp)

res = AnomalyDetectionTs(raw_data, 
                         max_anoms=0.02, 
                         direction='both', 
                         plot=FALSE)


res$anoms$timestamp <- as.POSIXct(res$anoms$timestamp)

# Marcar anomalías usando `left_join`
raw_data <- raw_data %>%
  left_join(res$anoms %>% select(timestamp) %>% mutate(is_anomaly = TRUE), by = "timestamp") %>%
  mutate(anomaly = ifelse(is.na(is_anomaly), "Normal", "Anomaly")) %>%
  select(-is_anomaly)

# Crear el gráfico con ggplot2
ggplot(raw_data, aes(x = timestamp, y = count)) +
  geom_line(color = "blue") +
  geom_point(data = filter(raw_data, anomaly == "Anomaly"), 
             aes(x = timestamp, y = count), 
             color = "red", size = 4) +
  labs(title = "Anomaly Detection", x = "Timestamp", y = "Count") +
  theme_minimal()
