
# Tema 2. Sistemas de aprendizaje automático supervisado

# Regresión Logística

## Contenidos

> 1. Introducción
> 2. Diferencias con la regresión lineal
> 3. Modelo probabilístico: función sigmoide
> 4. Log-likelihood y ajuste del modelo
> 5. Evaluación del modelo: precisión, recall, F1-score, ROC, AUC - Multicolinealidad y regularización
> 6. Ejemplos prácticos con Python

---

## Introducción a la regresión logística
Explicación del propósito de la regresión logística en contraste con la regresión lineal. Descripción de cómo la regresión logística se emplea en problemas de clasificación binaria y no para predicciones continuas. Presentación de ejemplos prácticos donde se prefiera la regresión logística, como la clasificación de correos electrónicos en spam/no spam o la detección de fraude.

## Diferencias con la regresión lineal
Análisis comparativo entre la regresión lineal y la logística:
- **Naturaleza del problema**: La regresión lineal se utiliza para problemas de predicción continua, mientras que la regresión logística se emplea para clasificación binaria o multiclase.
- **Función de salida**: La regresión lineal produce valores continuos que pueden ser negativos o mayores que 1, mientras que la regresión logística produce una probabilidad entre 0 y 1.
- **Ajuste de la función**: Uso de una función sigmoide en la regresión logística para transformar la salida en una probabilidad.

## Modelo probabilístico: función sigmoide
Desglose matemático de la función sigmoide:
- Definición y ecuación: 
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
donde $z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$.
- Interpretación: cómo se utiliza para convertir un modelo de predicción lineal en un modelo de clasificación que produce una probabilidad.
- Visualización de la función sigmoide y explicación de cómo mapea la salida a un rango de 0 a 1.

## Log-likelihood y ajuste del modelo
Explicación del método de máxima verosimilitud (log-likelihood) para el ajuste de los coeficientes:
- Fórmula de la función de verosimilitud y su derivada logarítmica:
$$
\log L(\beta) = \sum_{i=1}^n \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
$$
- Introducción al uso de algoritmos como el **descenso de gradiente** para optimizar la función de log-verosimilitud.

## Evaluación del modelo: precisión, recall, F1-score, ROC, AUC
- Descripción de las métricas de evaluación:
  - **Precisión (Accuracy)**: proporción de predicciones correctas.
  - **Recall (Sensibilidad)**: capacidad del modelo para detectar los casos positivos.
  - **F1-score**: medida armónica que balancea precisión y recall.
- Introducción al gráfico **ROC (Receiver Operating Characteristic)** y cálculo del **AUC (Area Under the Curve)** para evaluar la capacidad discriminativa del modelo.
- Ejemplos gráficos de curvas ROC y su interpretación.

## Multicolinealidad y regularización
Explicación de cómo la multicolinealidad entre variables independientes puede afectar la estabilidad del modelo.
- Uso de técnicas de regularización como **Ridge (L2)** y **Lasso (L1)** para mitigar la multicolinealidad y controlar la complejidad del modelo.
- Ecuación del término de regularización y su impacto en la función de pérdida.

## Ejemplos prácticos con Python
- Código para implementar un modelo de regresión logística con **scikit-learn**.
- Uso de un conjunto de datos como el **Iris** o **Titanic** para mostrar:
  - Entrenamiento del modelo.
  - Evaluación con métricas de rendimiento y gráficos ROC.
- Ejemplo de regularización en Python y análisis de los coeficientes con **L1** y **L2**.

Este esquema proporciona una estructura sólida para desarrollar un contenido completo sobre regresión logística, abarcando desde los fundamentos teóricos hasta ejemplos prácticos con Python.
