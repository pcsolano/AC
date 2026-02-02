if(!require("caret")) {
  install.packages("caret", dependencies = c("Depends", "Suggests"))
  require(caret)
}
if(!require("e1071")) {
  install.packages("e1071")
  require(e1071)
}
if(!require("randomForest")) {
  install.packages("randomForest")
  require(randomForest)
}
if(!require("kernlab")) {
  install.packages("kernlab")
  require(kernlab)
}
if (!require("pROC")) {
  install.packages("pROC")
  require(pROC)
}

library(lattice)
library(caret)
library(ggplot2)
library(mlbench)
library(e1071)
library(randomForest)
library(kernlab)
library(GGally)
library(nnet)
library(pROC)

#1.Preparación de los datos
#Cargar datos
credit <- read.csv("D:\\AC\\bd\\crx.data", sep=",",header=F, na.strings ="?")
credit.trainIdx<-readRDS("D:\\AC\\credit.trainIdx.rds")  

train_data <- credit[credit.trainIdx, ]
test_data <- credit[-credit.trainIdx, ]

#2. Analisis
cat("Filas en entrenamiento:", nrow(train_data), "\n")
cat("Filas en prueba:", nrow(test_data), "\n")

#Determinar el número de variables numéricas y categóricas
numeric_vars <- names(train_data)[sapply(train_data, is.numeric)]
categorical_vars <- names(train_data)[sapply(train_data, is.character)]

cat("Número de variables numéricas:", length(numeric_vars), "\n")
cat("Variables numéricas:", numeric_vars, "\n")
cat("Número de variables categóricas:", length(categorical_vars), "\n")
cat("Variables categóricas:", categorical_vars, "\n")

#Calcular la distribución de valores para cada variable categórica
for (var in categorical_vars) {
  cat("\nDistribución de la variable categórica:", var, "\n")
  print(table(train_data[[var]]))
}
#2.1 Analisis monovariable de V2, V8, V11
analisis_variable <- function(variable, nombre) {
  
  densidad <- density(variable, na.rm = T)
  
  hist(variable, 
       xlab=paste("Valores de", nombre), 
       ylab="Densidad", 
       main=paste("Histograma de", nombre), 
       ylim=c(0, max(densidad$y)*1.1), 
       probability=T, 
       col="lightblue")
  
  
  lines(densidad, col="blue", lwd=2)
  abline(v=mean(variable, na.rm=T), col="red", lty=2, lwd=2)
  abline(v=median(variable, na.rm=T), col="green", lty=2, lwd=2)
  
  legend("topright", 
         legend=c("Curva de densidad", "Media", "Mediana"),
         col=c("blue", "red", "green"),
         lwd=c(2, 2, 2), 
         lty=c(1, 2, 2),
         bg="white")
  
  rug(jitter(variable))
  
  bwplot(variable, main = paste("Diagrama de caja de los valores de ", nombre),
         ylab = paste("Valores de",nombre), 
         xlab = "",
         col = "lightblue")
  
  cat("Prueba de normalidad (Shapiro-Wilk) para", nombre, ":\n")
  shapiro.test(na.omit(variable))
  
}

summary(train_data$V2)
analisis_variable(train_data$V2, "V2")
bwplot(train_data$V2, main = "Diagrama de caja de los valores de V2", 
       ylab = "Valores de V2", 
       xlab = "",
       col = "lightblue")

summary(train_data$V8)
analisis_variable(train_data$V8, "V8")
bwplot(train_data$V8, main = "Diagrama de caja de los valores de V8", 
       ylab = "Valores de V8", 
       xlab = "",
       col = "lightblue")

summary(train_data$V11)
analisis_variable(train_data$V11, "V11")
bwplot(train_data$V11, main = "Diagrama de caja de los valores de V11", 
       ylab = "Valores de V11", 
       xlab = "",
       col = "lightblue")

###V9 - Variable categórica binaria
table(train_data$V9)

barplot(table(train_data$V9), 
        main = "Distribución de V9", 
        xlab = "Categorías de V9", 
        ylab = "Frecuencia", 
        col = c("lightblue", "pink"))


#Los 3 histogramas a la vez
data <- data.frame(
  V2= train_data$V2,
  V8 = train_data$V8,
  V11 = train_data$V11
)

data_complete <- data[complete.cases(data), ]
data_long <- stack(data_complete)

levels(data_long$ind) <- c("V2", "V8", "V11")

histogram(~ values | ind,       
          data = data_long, 
          layout = c(1, 3),     
          panel = function(x, ...) {
            panel.histogram(x, col = "lightblue", probability = TRUE,...)  
            panel.densityplot(x, col = "red", lwd = 2)        
            panel.abline(v = mean(x, na.rm = TRUE), col = "blue", lwd = 2, lty = 2) 
          },
          main = "Histogramas de V2, V8 y V11 con curva de densidad", 
          xlab = "Valores",                                          
          ylab = "Frecuencia")   


#2.2 Analisis multivariable
variables_numericas <- train_data[, sapply(train_data, is.numeric)]

cat("Matriz de correlaciones:\n")
cor(variables_numericas, use="complete.obs")


#Visualización multivariable
ggpairs(variables_numericas, title="Análisis multivariable")


#2.3 Tratamiento de valores omitidos y valores atípicos 
#Comprobar valores nulos en cada columna
total_missing <- colSums(is.na(train_data))
cat("\nValores nulos por columna:\n")
print(total_missing)

percent_rows_with_na <- ((sum(rowSums(is.na(train_data)) > 0)) / nrow(train_data)) * 100
cat("Porcentaje de filas con al menos un NA:\n", percent_rows_with_na, "%")


#Eliminar los filas con NA
train_data_clean <- na.omit(train_data)
test_data_clean <- na.omit(test_data)

#Identificación de valores atípicos basada en el rango intercuartílico (IQR) 
variables_numericas <- sapply(train_data_clean, is.numeric)

detectar_outliers <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR_valor <- Q3 - Q1
  limite_inferior <- Q1 - 1.5 * IQR_valor
  limite_superior <- Q3 + 1.5 * IQR_valor
  list(outliers = which(x < limite_inferior | x > limite_superior), limite_inferior = limite_inferior, limite_superior = limite_superior)
}

for (col in names(train_data_clean)[variables_numericas]) {
  limite_inferior <- detectar_outliers(train_data_clean[[col]])$limite_inferior
  limite_superior <- detectar_outliers(train_data_clean[[col]])$limite_superior
  
  outliers <- detectar_outliers(train_data_clean[[col]])$outliers
  
  porcentaje_outliers <- (length(outliers) / length(train_data_clean[[col]])) * 100
  
  desviaciones_absolutas <- abs(train_data_clean[[col]][outliers] - median(train_data_clean[[col]]))
  valor_mad <- median(desviaciones_absolutas)
  max_desviacion <- max(desviaciones_absolutas)
  
  cat("Variable:", col, "\n")
  cat("Porcentaje de valores atípicos:", round(porcentaje_outliers, 2), "%\n")
  cat("Mediana de las desviaciones absolutas de los valores atípicos respecto a la mediana:", round(valor_mad, 2), "\n")
  cat("Límite inferior:", limite_inferior, "Límite superior:", limite_superior, "\n")
  cat("Máxima desviación de los valores atípicos respecto a la mediana:", round(max_desviacion, 2), "\n")
  
  cat("\n")
}

#Elimanar los valores atípico de V2, V3 y V14 
for (col in c("V2", "V3","V14")) {
  outliers <- detectar_outliers(train_data_clean[[col]])$outliers
  limite_inferior <- detectar_outliers(train_data_clean[[col]])$limite_inferior
  limite_superior <- detectar_outliers(train_data_clean[[col]])$limite_superior
  
  train_data_clean <- train_data_clean[-outliers, ]
  test_data_clean <- test_data_clean[test_data_clean[[col]] >= limite_inferior & test_data_clean[[col]] <= limite_superior, ]
}


#Winsorizar valores atípicos de V8 y V11
for (col in c("V8", "V11")) {
  limite_inferior <- detectar_outliers(train_data_clean[[col]])$limite_inferior
  limite_superior <- detectar_outliers(train_data_clean[[col]])$limite_superior
  
  train_data_clean[[col]][train_data_clean[[col]] < limite_inferior] <- limite_inferior
  train_data_clean[[col]][train_data_clean[[col]] > limite_superior] <- limite_superior
  
  test_data_clean[[col]][test_data_clean[[col]] < limite_inferior] <- limite_inferior
  test_data_clean[[col]][test_data_clean[[col]] > limite_superior] <- limite_superior
}

#Transformación logarítmica de V15
train_data_clean$V15 <- log1p(train_data_clean$V15)
test_data_clean$V15 <- log1p(test_data_clean$V15)


#2.4 Escalamiento 
preProc <- preProcess(train_data_clean, method = c("center", "scale"))

# Método de preprocesamiento 1: Datos escalados y limpios
scaled_train_data <- predict(preProc, train_data_clean)
scaled_test_data <- predict(preProc, test_data_clean)

# Método de preprocesamiento 2: Datos no escalados y limpios
unscaled_train_data <- train_data_clean
unscaled_test_data <- test_data_clean

#2.5 PCA
numeric_data <- scaled_train_data[, sapply(scaled_train_data, is.numeric)]

pca_result <- prcomp(numeric_data, center = TRUE, scale. = TRUE)

#Variabilidad explicada por cada componente
explained_variance <- summary(pca_result)$importance[2, ]
cat("\nVarianza explicada por los primeros dos componentes principales:\n")
cat("PC1:", explained_variance[1], "\n")
cat("PC2:", explained_variance[2], "\n")

pca_data <- as.data.frame(pca_result$x)
pca_data$Class <- train_data_clean$V16 

ggplot(pca_data, aes(x = PC1, y = PC2, color = Class)) +
  geom_point(alpha = 0.7, size = 2) +
  labs(
    title = "Visualización PCA",
    x = "Primer Componente Principal (PC1)",
    y = "Segundo Componente Principal (PC2)",
    color = "Clase"
  ) +
  theme_minimal()


#3. Train los datos

scaled_train_data[sapply(scaled_train_data, is.character)] <- lapply(scaled_train_data[sapply(scaled_train_data, is.character)], as.factor)
scaled_test_data[sapply(scaled_test_data, is.character)] <- lapply(scaled_test_data[sapply(scaled_test_data, is.character)], as.factor)

for (col in names(scaled_train_data)) {
  if (is.factor(scaled_train_data[[col]])) {
    levels(scaled_test_data[[col]]) <- levels(scaled_train_data[[col]])
  }
}

train_control <- trainControl(method = "cv", number = 10)


plot_model_results <- function(nombre, predictions, training_data) {
  ggplot(training_data, aes_string(x = "V2", y ="V8", color = predictions)) +
    geom_point() +
    labs(
      title = paste("Resultados de ", nombre, " para los datos de entrenamiento"),
      x = "V2",
      y = "V8",
      color = "Clase"
    ) +
    theme_minimal()
}

bootstrap_accuracy <- function(modelo, testing_data) {
  n_bootstrap <- 1000
  accuracies <- numeric(n_bootstrap)
  
  for (i in 1:n_bootstrap) {
    indices <- sample(1:nrow(testing_data), replace = TRUE)
    bootstrap_sample <- testing_data[indices, ]
    
    predicciones <- predict(modelo, newdata = bootstrap_sample)
    
    accuracies[i] <- mean(predicciones== bootstrap_sample$V16)
  }
  
  interval <- quantile(accuracies, probs = c(0.025, 0.975))
  return(list(accuracy = accuracies, confidence_interval = interval))
}



#3.1 NNET

set.seed(123)  # Cambia el número si deseas obtener una división diferente

# Proporción para el conjunto de validación
validation_ratio <- 0.2  # 20% de los datos para validación

# Crear índices aleatorios para la división
validation_indices <- sample(1:nrow(scaled_train_data), size = floor(validation_ratio * nrow(scaled_train_data)))

# Dividir los datos en conjunto de entrenamiento e validación
v_data <- scaled_train_data[validation_indices, ]  # Conjunto de validación
in_data <- scaled_train_data[-validation_indices, ]   # Conjunto de entrenamiento

tune_grid_nnet <- expand.grid(
  size = c(5, 10, 15),      
  decay = c(0.1, 0.01, 0.001) 
)

modelo_nnet <- train(
  V16 ~ .,
  data = in_data,
  method = "nnet",
  tuneGrid = tune_grid_nnet,
  trControl = train_control,
  trace = FALSE
)

print(modelo_nnet)

test_predictions_nnet <- predict(modelo_nnet, newdata = v_data)
train_predictions_nnet <- predict(modelo_nnet, newdata= in_data)


conf_matrix_nnet <- confusionMatrix(test_predictions_nnet, v_data$V16)
print(conf_matrix_nnet)

plot(modelo_nnet)

plot_model_results(nombre = "NNET",predictions =  train_predictions_nnet, in_data)


result_nnet <- bootstrap_accuracy(modelo_nnet, v_data)
cat("Intervalo de confianza de NNET del 95% para el accuracy:", 
    round(result_nnet$confidence_interval[1], 4), "-", 
    round(result_nnet$confidence_interval[2], 4), "\n")




##3.2 Random Forest
modelo_rf <- train(V16 ~ ., 
                   data = in_data, 
                   method = "rf", 
                   trControl = train_control)

print(modelo_rf)

test_predictions_rf <- predict(modelo_rf, newdata = v_data)
train_predictions_rf <- predict(modelo_rf, newdata = in_data)

conf_matrix_rf <- confusionMatrix(test_predictions_rf, v_data$V16)
print(conf_matrix_rf)


plot(modelo_rf)

plot_model_results(nombre = "RF",predictions =train_predictions_rf, in_data)

result_rf<- bootstrap_accuracy(modelo_rf, scaled_test_data)
cat("Intervalo de confianza de Random Forest del 95% para el accuracy:", 
    round(result_rf$confidence_interval[1], 4), "-", 
    round(result_rf$confidence_interval[2], 4), "\n")


##3.2.1 Random Forest sin normalizar

set.seed(123)  #Cambia el número si deseas obtener una división diferente

#Proporción para el conjunto de validación
validation_ratio <- 0.2  #20% de los datos para validación

#Crear índices aleatorios para la división
validation_indices <- sample(1:nrow(unscaled_train_data), size = floor(validation_ratio * nrow(unscaled_train_data)))

#Dividir los datos en conjunto de entrenamiento e validación
v_data <- unscaled_train_data[validation_indices, ]  
in_data <- unscaled_train_data[-validation_indices, ] 

modelo_rfn <- train(V16 ~ ., 
                    data = in_data, 
                    method = "rf", 
                    trControl = train_control)

print(modelo_rfn)

test_predictions_rfn <- predict(modelo_rfn, newdata = v_data)
train_predictions_rfn <- predict(modelo_rfn, newdata = in_data)

test_predictions_rfn <- as.factor(test_predictions_rfn)
train_predictions_rfn <- as.factor(train_predictions_rfn)

v_data$V16 <- as.factor(v_data$V16)
levels(test_predictions_rfn) <- levels(v_data$V16)
conf_matrix_rfn <- confusionMatrix(test_predictions_rfn, v_data$V16)
print(conf_matrix_rfn)

#Visualización de la importancia de las variables
varImpPlot(modelo_rfn$finalModel)

plot(modelo_rfn)

plot_model_results(nombre = "RF sin normalizar",predictions =train_predictions_rfn, in_data)

result_rfn <- bootstrap_accuracy(modelo_rfn, v_data)
cat("Intervalo de confianza de RF sin normalizar del 95% para el accuracy:", 
    round(result_rfn$confidence_interval[1], 4), "-", 
    round(result_rfn$confidence_interval[2], 4), "\n")




#3.3 SVM

set.seed(123) 

validation_ratio <- 0.2

validation_indices <- sample(1:nrow(scaled_train_data), size = floor(validation_ratio * nrow(scaled_train_data)))

v_data <- scaled_train_data[validation_indices, ]
in_data <- scaled_train_data[-validation_indices, ]   

modelo_svm <- train(V16 ~ ., 
                    data = scaled_train_data, 
                    method = "svmRadial", 
                    trControl = train_control)

print(modelo_svm)

test_predictions_svm <- predict(modelo_svm, newdata = v_data)
train_predictions_svm <- predict(modelo_svm, newdata = in_data)

conf_matrix_svm <- confusionMatrix(test_predictions_svm, v_data$V16)
print(conf_matrix_svm)

plot(modelo_svm)

plot_model_results(nombre = "SVM", predictions =  train_predictions_svm, in_data)

result_svm <- bootstrap_accuracy(modelo_svm, v_data)
cat("Intervalo de confianza de SVM del 95% para el accuracy:", 
    round(result_svm$confidence_interval[1], 4), "-", 
    round(result_svm$confidence_interval[2], 4), "\n")


##3.3.1 SVM sin normalizar

set.seed(123)  

validation_ratio <- 0.2

validation_indices <- sample(1:nrow(unscaled_train_data), size = floor(validation_ratio * nrow(unscaled_train_data)))

v_data <- unscaled_train_data[validation_indices, ]  # Conjunto de validación
vi_data <- unscaled_train_data[validation_indices, ]  # Conjunto de validación
in_data <- unscaled_train_data[-validation_indices, ]   # Conjunto de entrenamiento


modelo_svmn <- train(V16 ~ ., 
                    data = in_data, 
                    method = "svmRadial", 
                    trControl = train_control)

print(modelo_svmn)

test_predictions_svmn <- predict(modelo_svmn, newdata = v_data)
train_predictions_svmn <- predict(modelo_svmn, newdata = in_data)

test_predictions_svmn <- as.factor(test_predictions_svmn)
train_predictions_svmn <- as.factor(train_predictions_svmn)


v_data$V16 <- as.factor(v_data$V16)
levels(test_predictions_svmn) <- levels(v_data$V16)
conf_matrix_svmn <- confusionMatrix(test_predictions_svmn, v_data$V16)
print(conf_matrix_svmn)

plot(modelo_svmn)

plot_model_results(nombre = "SVM sin normalizar",predictions =train_predictions_svmn, in_data)

result_svmn <- bootstrap_accuracy(modelo_svmn, v_data)
cat("Intervalo de confianza de SVM sin normalizar del 95% para el accuracy:", 
    round(result_svmn$confidence_interval[1], 4), "-", 
    round(result_svmn$confidence_interval[2], 4), "\n")

#3.4 GBM (Gradient Boosting Machine)


set.seed(123)  
validation_ratio <- 0.2 

validation_indices <- sample(1:nrow(scaled_train_data), size = floor(validation_ratio * nrow(scaled_train_data)))

v_data <- scaled_train_data[validation_indices, ]
in_data <- scaled_train_data[-validation_indices, ]   

modelo_gbm <- train(V16 ~ ., 
                    data = in_data, 
                    method = "gbm", 
                    trControl = train_control,
                    verbose = FALSE)

print(modelo_gbm)

test_predictions_gbm <- predict(modelo_gbm, newdata = v_data)
train_predictions_gbm <- predict(modelo_gbm, newdata = in_data)


conf_matrix_gbm <- confusionMatrix(test_predictions_gbm, v_data$V16)
print(conf_matrix_gbm)
plot(modelo_gbm)

gbm_results <- modelo_gbm$results


plot_model_results(nombre = "GBM",predictions = train_predictions_gbm, in_data)

result_gbm <- bootstrap_accuracy(modelo_gbm, v_data)
cat("Intervalo de confianza de GBM del 95% para el accuracy:", 
    round(result_gbm$confidence_interval[1], 4), "-", 
    round(result_gbm$confidence_interval[2], 4), "\n")



#3.5 Comparar modelos
modelos <- list(
  NNET = modelo_nnet,
  RF = modelo_rf,
  RF_sin_normalizar = modelo_rfn,
  SVM = modelo_svm,
  SVM_sin_normalizar = modelo_svmn,
  GBM = modelo_gbm
)

plotear_curvas_roc <- function(modelos) {
  lista_roc <- list()
  
  for (nombre_modelo in names(modelos)) {
    probabilidades<- predict(modelos[[nombre_modelo]], newdata = v_data, type = "prob")[, 2]  
    
    if (any(is.na(probabilidades))) {
      warning(paste("Modelo ",nombre_modelo, " no tiene valores válidos para la curva ROC."))
      next
    }
    
    curva_roc <- roc(v_data$V16, probabilidades, levels = rev(levels(v_data$V16)))
    lista_roc[[nombre_modelo]] <- data.frame(
      Sensibilidad = rev(curva_roc$sensitivities),
      Especificidad = rev(1 - curva_roc$specificities),
      Modelo =  nombre_modelo
    )  
  }
  
  datos_roc <- do.call(rbind, lista_roc)
  
  ggplot(datos_roc, aes(x = Especificidad, y = Sensibilidad, color = Modelo)) +
    geom_line(size = 1.2) +
    labs(
      title = "Comparación de curvas ROC de los modelos",
      x = "1 - Especificidad (Tasa de falsos positivos)",
      y = "Sensibilidad (Tasa de verdaderos positivos)",
      color = "Modelos"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
}

plotear_curvas_roc(modelos)


#Comparar en un gráfico de barras
ic_nnet <- bootstrap_accuracy(modelo_nnet, v_data)
ic_rf <- bootstrap_accuracy(modelo_rf, v_data)
ic_rfn <- bootstrap_accuracy(modelo_rfn, vi_data)
ic_svm <- bootstrap_accuracy(modelo_svm, v_data)
ic_svmn <- bootstrap_accuracy(modelo_svmn, vi_data)
ic_gbm <- bootstrap_accuracy(modelo_gbm, v_data)

metricas_modelos <- data.frame(
  Modelo = c("NNET", "Random Forest", "Random Forest SN", "SVM", "SVM SN", "GBM"),
  Precisión = c(
    conf_matrix_nnet$overall["Accuracy"],
    conf_matrix_rf$overall["Accuracy"],
    conf_matrix_rfn$overall["Accuracy"],
    conf_matrix_svm$overall["Accuracy"],
    conf_matrix_svmn$overall["Accuracy"],
    conf_matrix_gbm$overall["Accuracy"]
  ),
  Sensibilidad = c(
    conf_matrix_nnet$byClass["Sensitivity"],
    conf_matrix_rf$byClass["Sensitivity"],
    conf_matrix_rfn$byClass["Sensitivity"],
    conf_matrix_svm$byClass["Sensitivity"],
    conf_matrix_svmn$byClass["Sensitivity"],
    conf_matrix_gbm$byClass["Sensitivity"]
  ),
  Especificidad = c(
    conf_matrix_nnet$byClass["Specificity"],
    conf_matrix_rf$byClass["Specificity"],
    conf_matrix_rfn$byClass["Specificity"],
    conf_matrix_svm$byClass["Specificity"],
    conf_matrix_svmn$byClass["Specificity"],
    conf_matrix_gbm$byClass["Specificity"]
  ),
  Intervalo_Confianza = c(
    paste0(round(ic_nnet$confidence_interval[1], 4), " - ", round(ic_nnet$confidence_interval[2], 4)),
    paste0(round(ic_rf$confidence_interval[1], 4), " - ", round(ic_rf$confidence_interval[2], 4)),
    paste0(round(ic_rfn$confidence_interval[1], 4), " - ", round(ic_rfn$confidence_interval[2], 4)),
    paste0(round(ic_svm$confidence_interval[1], 4), " - ", round(ic_svm$confidence_interval[2], 4)),
    paste0(round(ic_svmn$confidence_interval[1], 4), " - ", round(ic_svm$confidence_interval[2], 4)),
    paste0(round(ic_gbm$confidence_interval[1], 4), " - ", round(ic_gbm$confidence_interval[2], 4))
  )
)



print("Comparación de los modelos:")
print(metricas_modelos)

ggplot(metricas_modelos, aes(x = Modelo, y = Precisión)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = round(Precisión, 4)), vjust = -0.5) +
  labs(title = "Comparación de Modelos: Precisión", x = "Modelo", y = "Precisión") +
  theme_minimal()




mejor_modelo <- metricas_modelos[which.max(metricas_modelos$Precisión), "Modelo"]

cat("El mejor modelo es:", mejor_modelo, "\n")
cat("El modelo", mejor_modelo, "fue seleccionado porque obtuvo la precisión más alta.")

#Elegimos Random Forest SN

modelo_final <- train(V16 ~ ., 
                    data = unscaled_train_data, 
                    method = "rf", 
                    trControl = train_control)

print(modelo_rfn)

test_predictions_rfn <- predict(modelo_rfn, newdata = unscaled_test_data)
train_predictions_rfn <- predict(modelo_rfn, newdata = unscaled_train_data)

test_predictions_rfn <- as.factor(test_predictions_rfn)
train_predictions_rfn <- as.factor(train_predictions_rfn)

unscaled_test_data$V16 <- as.factor(unscaled_test_data$V16)
levels(test_predictions_rfn) <- levels(unscaled_test_data$V16)
conf_matrix_rfn <- confusionMatrix(test_predictions_rfn, unscaled_test_data$V16)
print(conf_matrix_rfn)

#Visualización de la importancia de las variables
varImpPlot(modelo_rfn$finalModel)

plot(modelo_rfn)

plot_model_results(nombre = "RF sin normalizar",predictions =train_predictions_rfn, unscaled_train_data)

result_rfn <- bootstrap_accuracy(modelo_rfn, unscaled_test_data)
cat("Intervalo de confianza de RF sin normalizar del 95% para el accuracy:", 
    round(result_rfn$confidence_interval[1], 4), "-", 
    round(result_rfn$confidence_interval[2], 4), "\n")

