# Actividad práctica: Fases de un proyecto de machine learning supervisado

## Objetivo de la Actividad

El objetivo de esta actividad es guiarte en el desarrollo de un proyecto de Machine Learning supervisado completo, desde la comprensión de los datos hasta la evaluación del modelo. Este ejercicio te permitirá aplicar las fases y conceptos estudiados, utilizando un dataset de tu elección, y profundizar en el flujo de trabajo de un proyecto de Machine Learning.

## Descripción de la Actividad

Selecciona un dataset relevante a partir de los ejemplos vistos en clase o cualquier otro que sea de tu interés (por ejemplo, predicción de precios, clasificación de clientes, etc.). A continuación, sigue las fases indicadas, aplicando los métodos estudiados para cada una.

En cualquier caso, al final del enunciado de la actividad te doy varias posibilidades de encontrar datasets listos para trabajar en algunas de las bibliotecas de python más comunes para trabajar analítica de datos

## Fases del Proyecto

### Recolección de datos

#### Objetivo

- Recopilar un conjunto de datos etiquetados que será utilizado para entrenar y probar el modelo.

#### Instrucciones

- Selecciona un dataset representativo del problema que se quiere resolver.
- Asegúrate de que el dataset incluya tanto la variable objetivo como las variables explicativas.
- Puedes elegir datasets disponibles en Scikit-learn, PyCaret, o cualquier otro que sea de tu interés.

---

### Análisis exploratorio de datos (EDA)

#### Objetivo

- Aplicar técnicas de exploración de datos para comprender su estructura y características, identificar patrones y posibles anomalías.

#### Instrucciones

- Revisa el tamaño, tipos de datos y primeros registros del dataset.
- Explora las estadísticas descriptivas y posibles patrones iniciales.
- Identifica valores atípicos y analiza las distribuciones de los datos mediante visualizaciones.
- Herramientas sugeridas: `df.info()`, `df.describe()`, visualización con histogramas y gráficos de dispersión.

---

### Preprocesamiento de datos

#### Objetivo

- Aplicar técnicas de transformación y preparación de los datos para su uso en modelos de machine learning.

#### Instrucciones

- Gestiona valores faltantes (imputación o eliminación).
- Normaliza o estandariza las variables numéricas si es necesario.
- Realiza codificación de variables categóricas (One-Hot, Label Encoding...).
- Genera nuevas características si observas relaciones relevantes entre variables.
- Herramientas sugeridas: técnicas de escalado, imputación, y transformación con `StandardScaler`, `LabelEncoder`, `pd.get_dummies()`.

---

### División del conjunto de datos

#### Objetivo

- Dividir el conjunto de datos en subconjuntos de entrenamiento, validación y prueba para evaluar correctamente el modelo.

#### Instrucciones

- Divide los datos en conjuntos de entrenamiento y prueba (por ejemplo, 80%-20%).
- Asegúrate de que el conjunto de prueba no sea utilizado hasta la evaluación final.
- Herramientas sugeridas: `train_test_split` de Scikit-learn.

---

### Entrenamiento del modelo

#### Objetivo

- Aprender patrones subyacentes en los datos mediante el ajuste de los parámetros internos del modelo.

#### Instrucciones

- Escoge el tipo de modelo en función del problema. Puedes hacer un problema de predicción o de clasificación, pero en cualquier caso los modelos (y por tanto los posibles problemas) han de ser de regresión. La idea es sólo empezar a familiarizarte con las fases del proyecto y no con el tipo de modelo.
- Entrena el modelo en los datos de entrenamiento y asegúrate de registrar el rendimiento en métricas.
- Herramientas sugeridas: modelos de Scikit-learn como `LinearRegression`, `LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, etc.

---

### Validación y ajuste del modelo

#### Objetivo

- Evaluar la capacidad de generalización del modelo y ajustar el rendimiento en función de las métricas obtenidas en el conjunto de validación.

#### Instrucciones

- Ejecuta el modelo en los datos de validación y mide su rendimiento utilizando métricas adecuadas:
  - Para regresión: RMSE, MAE, y R².
  - Para clasificación: exactitud (accuracy), precisión, recall, y F1-score.
- Evalúa posibles ajustes en la arquitectura del modelo o en el preprocesamiento.
- Herramientas sugeridas: funciones de métricas de Scikit-learn como `mean_squared_error`, `r2_score`, `accuracy_score`, `classification_report`.

---

### Ajuste de hiperparámetros

- No es necesario a estas alturas del curso acometer esta fase. Ya la trabajaremos en temas posteriores

---

### Predicción final

#### Objetivo

- Aplicar el modelo ajustado para realizar predicciones sobre nuevos datos no etiquetados.

#### Instrucciones

- Utiliza el modelo entrenado para hacer predicciones con nuevos datos no vistos.
- Asegúrate de que estos datos están preprocesados de la misma manera que los datos de entrenamiento.
- Interpreta las predicciones y considera las implicaciones de los resultados en el contexto del problema. 

## Evaluación y entrega

Al final del proyecto, entrega un informe que incluya:

1. Una breve descripción del dataset elegido y el objetivo del modelo.
2. Un resumen del análisis exploratorio y los gráficos de visualización.
3. Las decisiones tomadas en el preprocesamiento y justificación de las técnicas aplicadas.
4. El tipo de modelo utilizado y el proceso de ajuste de hiperparámetros.
5. Una evaluación de los resultados obtenidos, incluyendo métricas y posibles limitaciones.
6. Conclusiones finales y sugerencias para mejorar el modelo.

Todo ello debe incorporarse en formato Jupyter Notebook. Puedes utilizar las herramientas de GCP para ello.

Esta actividad te permitirá consolidar el conocimiento de cada fase en un proyecto de Machine Learning supervisado y poner en práctica los conceptos clave que hemos estudiado.

## Apéndice: Datasets de prueba para la realización de la práctica

De cara a la actividad de desarrollo consistente en la realización de un proyecto sencillo de regresión con el objeto de entender las distintas fases de un flujo de machine learning existen varias bibliotecas en Python que incorporan **datasets orientados a modelos regresión** que son útiles para practicar análisis predictivo y construir modelos de regresión de diferentes niveles de dificultad. Algunas de estas bibliotecas son **scikit-learn**, **statsmodels**, o **pycaret**. A continuación se describen algunos de sus datasets y ejemplos para regresión.

### Scikit-learn

Scikit-learn incluye algunos datasets integrados ideales para regresión, que pueden cargarse fácilmente para ejercicios y proyectos prácticos.

#### **Boston Housing** (código: `load_boston`):

- **Descripción**: Datos sobre viviendas en Boston, utilizados para predecir el valor medio de las viviendas en función de varias características.
- **Características**: Incluye variables como la proporción de zonas residenciales, la tasa de criminalidad, el número de habitaciones, el índice de ocupación, entre otros.
- **Código para iniciar con un DataFrame Pandas**

```python
from sklearn.datasets import load_boston
boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['PRICE'] = boston.target  # Agregar la variable objetivo
print("Boston Housing Dataset:")
print(boston_df.head(), "\n")
```

#### Diabetes (código: `load_diabetes`):

- **Descripción**: Datos médicos para predecir la progresión de la enfermedad de diabetes en función de indicadores de salud.
- **Características**: Incluye variables como edad, sexo, índice de masa corporal (BMI), presión arterial y valores de seis análisis de sangre.
- **Código para iniciar con un DataFrame Pandas**

```python
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
diabetes_df['TARGET'] = diabetes.target  # Agregar la variable objetivo
print("Diabetes Dataset:")
print(diabetes_df.head(), "\n")
```

#### **California Housing** (código: `fetch_california_housing`):

- **Descripción**: Datos sobre viviendas en California, utilizados para predecir el valor medio de las casas en función de características demográficas y geográficas.
- **Características**: Contiene variables como ingresos medios de los hogares, antigüedad de la vivienda, número de habitaciones y ocupación promedio.
- **Código para iniciar con un DataFrame Pandas**

```python
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
california_df = pd.DataFrame(california.data, columns=california.feature_names)
california_df['PRICE'] = california.target  # Agregar la variable objetivo
print("California Housing Dataset:")
print(california_df.head(), "\n")
```

#### **Ames Housing** (código: `fetch_openml`):

- **Descripción**: Extensión del dataset de Boston Housing, con más variables y una variedad más amplia de datos de viviendas.
- **Características**: Información detallada sobre casas en Ames, Iowa, como el área total del sótano, año de construcción, calificación general, y más de 70 otras características.
- **Código para iniciar con un DataFrame Pandas**

```python
from sklearn.datasets import fetch_openml
ames = fetch_openml(name="house_prices", as_frame=True)
ames_df = ames.frame
print("Ames Housing Dataset:")
print(ames_df.head(), "\n")
```

### Statsmodels

Statsmodels también incluye algunos datasets que son útiles para análisis de regresión, aunque es una biblioteca más orientada al análisis estadístico.

#### **Longley** (código: `longley`):

- **Descripción**: Datos económicos de EE.UU., comúnmente utilizados para predecir variables económicas como empleo y producción en función de otros factores.
- **Características**: Incluye variables como `GNP`, `Unemployment`, `Armed Forces`, y `Population`.
- **Código para iniciar con un DataFrame Pandas**

```python
import statsmodels.api as sm
longley_data = sm.datasets.longley.load_pandas()
longley_df = longley_data.data
print("Longley Dataset:")
print(longley_df.head(), "\n")
```

#### Stackloss (código: `stackloss`):

- **Descripción**: Datos sobre pérdida de apilamiento en una operación química, utilizados para predecir el porcentaje de pérdida en función de factores de control de calidad.
- **Características**: Incluye variables como `Air Flow`, `Water Temperature`, y `Acid Concentration`.
- **Código para iniciar con un DataFrame Pandas**

```python
import statsmodels.api as sm
stackloss_data = sm.datasets.stackloss.load_pandas()
stackloss_df = stackloss_data.data
print("Stackloss Dataset:")
print(stackloss_df.head(), "\n")
```

### PyCaret

**PyCaret** es una biblioteca para prototipado rápido de modelos de *machine learning*, e incluye varios datasets internos orientados a regresión.

#### Insurance

- **Descripción**: Dataset de aseguradoras, utilizado para predecir el costo de los seguros médicos en función de características del cliente.
- **Características**: Variables como `age`, `bmi`, `children`, y `smoker`.
- **Código para iniciar con un DataFrame Pandas**

```python
from pycaret.datasets import get_data
insurance_df = get_data("insurance")
print("Insurance Dataset:")
print(insurance_df.head(), "\n")
```

#### Concrete

- **Descripción**: Datos de resistencia del concreto en función de proporciones de ingredientes como cemento y agua.
- **Características**: Incluye `cement`, `water`, `age`, entre otros.
- **Código para iniciar con un DataFrame Pandas**

```python
from pycaret.datasets import get_data
concrete_df = get_data("concrete")
print("Concrete Dataset:")
print(concrete_df.head(), "\n")
```

#### Energy

- **Descripción**: Datos de edificios, utilizados para predecir el consumo de energía en función de características de diseño.
- **Características**: Variables como `relative_compactness`, `surface_area`, `wall_area`, y `roof_area`.
- **Uso**: Practicar regresión multivariable y análisis de eficiencia energética.
- **Código para iniciar con un DataFrame Pandas**

```python
from pycaret.datasets import get_data
energy_df = get_data("energy")
print("Energy Dataset:")
print(energy_df.head(), "\n")
```

