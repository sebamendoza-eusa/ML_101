# Tema 2. Sistemas de aprendizaje automático supervisado

## K-Vecinos más cercanos (KNN)

### Objetivos del módulo

> - Comprender el **funcionamiento del algoritmo KNN** y su enfoque basado en el concepto de *vecindad*, tanto en problemas de clasificación como de regresión.
> - Aplicar diferentes **métricas de distancia**, como la euclidiana o la Manhattan, evaluando su impacto en el rendimiento del modelo según el tipo de datos.
> - Analizar el **efecto del hiperparámetro $k$** en el sesgo y la varianza del modelo.
> - Implementar modelos KNN utilizando la librería **scikit-learn**.
> - Evaluar el rendimiento del modelo mediante métricas adecuadas.
> - Ajustar hiperparámetros de KNN utilizando técnicas como la **validación cruzada** o **GridSearch**
> - Aplicar KNN a problemas prácticos utilizando conjuntos de datos clásicos e interpretando los resultados obtenidos.

---

### **Introducción a KNN**

El algoritmo de **K-Nearest Neighbors (KNN)** es una de las técnicas más sencillas y efectivas dentro del aprendizaje supervisado. Su enfoque intuitivo lo convierte en una opción popular para resolver problemas de clasificación y regresión, basándose en la similitud entre ejemplos observados.

#### **Definición y concepto intuitivo**

KNN es un algoritmo basado en la **vecindad**, lo que significa que clasifica una nueva observación en función de las etiquetas de sus $k$ vecinos más cercanos en el espacio de características. En otras palabras, se asume que puntos de datos similares se encuentran próximos entre sí y que los elementos cercanos tienden a compartir la misma clase o valores similares.

El funcionamiento de KNN puede describirse de manera intuitiva en los siguientes pasos:

1. Se elige un valor para $k$, que representa el número de vecinos a considerar.
2. Se mide la distancia entre el nuevo punto y todos los puntos del conjunto de datos de entrenamiento.
3. Se seleccionan los $k$ vecinos más cercanos.
4. Se asigna la etiqueta más común (para clasificación) o el promedio de los valores (para regresión).

> **Ejemplo:** Supongamos que tenemos un conjunto de datos de frutas clasificadas por peso y color. Para clasificar una nueva fruta, buscamos las $k$ frutas más cercanas en términos de peso y color, y tomamos la decisión en función de la mayoría.

#### Un algoritmo no paramétrico

**KNN es un algoritmo no paramétrico** significa que **no hace suposiciones previas** sobre la distribución de los datos ni sobre la relación entre las variables de entrada y salida. A diferencia de los modelos paramétricos, como la regresión lineal, que definen una forma funcional específica (por ejemplo, una línea recta) y ajustan un conjunto fijo de parámetros durante el entrenamiento, KNN **no aprende una función explícita** a partir de los datos.

En lugar de ello, KNN almacena todo el conjunto de entrenamiento y realiza las predicciones **de manera local** al momento de la consulta. Esto significa que, para cada nueva instancia a clasificar o predecir, el algoritmo simplemente calcula la distancia entre la nueva observación y los datos existentes, basando su decisión en los vecinos más cercanos.

El carácter no paramétrico de KNN lo hace especialmente útil cuando la relación entre las características y la salida es compleja o desconocida, ya que no impone restricciones rígidas. Sin embargo, esta flexibilidad tiene un costo computacional elevado, ya que cada predicción requiere comparar la nueva muestra con todas las observaciones del conjunto de entrenamiento.

##### Para reflexionar...

>  **¿En qué tipo de problemas podría ser una ventaja o una desventaja utilizar un modelo no paramétrico como KNN?**
>  **Clave:** Considera si los datos presentan patrones claros o si el tamaño del conjunto de datos es grande y costoso de procesar.

#### La fase de entrenamiento en el algoritmo KNN

A diferencia de otros algoritmos de machine learning, **KNN no requiere un proceso de entrenamiento tradicional**, ya que no construye un modelo a partir de los datos. En su lugar, KNN simplemente **almacena el conjunto de datos de entrenamiento** y lo utiliza en la fase de predicción para encontrar los vecinos más cercanos a una nueva muestra.

En algoritmos como la regresión lineal o las redes neuronales, el entrenamiento implica ajustar parámetros internos mediante un proceso iterativo de optimización. En cambio, KNN sigue un **enfoque basado en memoria**, donde la única tarea previa a la predicción es almacenar los datos y, en algunos casos, aplicar técnicas de preprocesamiento como la normalización para mejorar la precisión del cálculo de distancias.

Cuando se realiza una predicción con KNN, el algoritmo calcula la **distancia entre la nueva instancia y todas las muestras de entrenamiento**, selecciona los $k$ vecinos más cercanos y toma una decisión basada en la mayoría (para clasificación) o el promedio (para regresión).

Este enfoque sin "entrenamiento" tiene ventajas y desventajas. Por un lado, permite implementar rápidamente un modelo sin necesidad de ajustes complejos, pero por otro lado, **el costo computacional se traslada a la fase de predicción**, ya que cada consulta requiere comparar la nueva observación con todos los datos almacenados.

> **Para reflexionar...**
>  **Si KNN no necesita entrenamiento, ¿cuáles son las principales limitaciones a considerar al aplicarlo en grandes volúmenes de datos?**
>  **Clave:** Piensa en el impacto del tiempo de predicción y el uso de memoria al almacenar todos los datos de entrenamiento.

#### **Aplicaciones comunes de KNN**

El algoritmo **KNN** es ampliamente utilizado en diversos ámbitos gracias a su **simplicidad** y **efectividad**, especialmente en problemas donde las relaciones entre las variables son complejas o no lineales.

En el campo del **reconocimiento de imágenes**, ha demostrado ser una herramienta útil para la **clasificación de dígitos escritos a mano** y la **identificación de objetos en imágenes digitales**, proporcionando soluciones efectivas sin necesidad de modelos complejos. Su aplicación también es frecuente en los **sistemas de recomendación**, donde permite **sugerir productos** a los usuarios basándose en la **similitud de preferencias** con otros clientes, facilitando experiencias personalizadas. En el ámbito de la **medicina**, se emplea para el **diagnóstico de enfermedades**, comparando los síntomas de un paciente con casos previamente registrados, lo que puede ayudar a identificar **patrones de riesgo**. Además, en la **detección de fraudes**, KNN se utiliza para analizar **transacciones financieras** y detectar **comportamientos sospechosos**, comparándolas con registros históricos y contribuyendo así a la **seguridad** en sistemas bancarios y comerciales.

##### **Para reflexionar...**

>  **¿Cuáles son las principales aplicaciones de KNN en tu campo de interés?**
>  **Clave:** Considera si tus datos son numéricos, categóricos o tienen una estructura espacial.



#### **Ventajas y limitaciones del algoritmo**

Como cualquier algoritmo de Machine Learning, KNN tiene sus fortalezas y debilidades. Es importante conocer estos aspectos antes de aplicarlo en un problema real.

##### **Ventajas**

 KNN es un algoritmo fácil de entender y de implementar, lo que lo convierte en una excelente opción para problemas en los que se requiere una solución rápida sin una compleja parametrización. Además, es un modelo **no paramétrico**, es decir, no hace suposiciones explícitas sobre la distribución de los datos. Otra ventaja clave es su capacidad para adaptarse bien a problemas donde las clases no están claramente separadas linealmente.

##### **Limitaciones**

Sin embargo, KNN puede ser computacionalmente costoso, especialmente con conjuntos de datos grandes, ya que requiere calcular la distancia entre cada punto nuevo y todos los puntos del conjunto de entrenamiento. También es sensible a la escala de las características, por lo que el **preprocesamiento** (como la normalización) es fundamental para un buen rendimiento. Otra desventaja es su susceptibilidad a datos ruidosos o irrelevantes, lo que puede afectar la precisión.

> **Para reflexionar...**
>  **¿Cómo se podría mejorar la eficiencia del algoritmo KNN para grandes volúmenes de datos?**
>  **Clave:** Piensa en técnicas como reducción de dimensionalidad o uso de estructuras de datos eficientes como KD-Trees.

------

#### **Cuándo usar KNN y cuándo evitarlo**

Es crucial entender en qué contextos KNN es una opción adecuada y cuándo es mejor optar por otros algoritmos más eficientes.

**KNN es una opción adecuada** cuando se dispone de un conjunto de datos de tamaño moderado, en el que el cálculo de distancias no represente un problema computacional significativo. Resulta especialmente útil en escenarios donde las relaciones entre las características son complejas y difíciles de modelar mediante funciones matemáticas tradicionales. Su aplicabilidad es óptima cuando los datos están bien estructurados, correctamente preprocesados y normalizados, lo que permite minimizar la influencia de posibles escalas desiguales entre las variables. Además, KNN es una alternativa ideal cuando se busca un modelo **interpretable y fácil de explicar**, ya que su mecanismo de predicción basado en vecinos cercanos permite comprender intuitivamente cómo se toman las decisiones.

Por otro lado, es recomendable **evitar el uso de KNN** en situaciones donde el tamaño del dataset es muy grande, ya que la búsqueda de vecinos más cercanos puede volverse computacionalmente costosa y lenta. También puede no ser la mejor opción cuando las características de los datos presentan escalas muy diferentes y no es posible normalizarlas adecuadamente, lo que podría afectar negativamente el rendimiento del algoritmo. Asimismo, en casos donde se requiere un modelo que generalice con rapidez sin depender de la necesidad de almacenar todo el conjunto de entrenamiento, KNN puede no ser la elección más eficiente debido a su enfoque basado en memoria.

##### **Para reflexionar...**

>  **¿Podría KNN ser una buena opción en un sistema de predicción en tiempo real? ¿Por qué sí o por qué no?**
>  **Clave:** Considera el costo de cómputo de buscar vecinos en grandes volúmenes de datos.

### **Funcionamiento del algoritmo**

El algoritmo KNN basa su funcionamiento en una idea sencilla pero poderosa: para hacer una predicción, compara una nueva observación con las muestras existentes en el conjunto de entrenamiento y determina su categoría o valor en función de los vecinos más cercanos. A diferencia de otros enfoques de aprendizaje supervisado, KNN no construye un modelo durante una fase de entrenamiento, sino que almacena los datos y los utiliza directamente en el momento de la predicción.

Cuando se introduce una nueva observación en el sistema, el primer paso consiste en calcular la distancia entre esta y cada una de las muestras previamente almacenadas. La métrica de distancia utilizada es crucial en este proceso, ya que determina qué tan "cerca" o "lejos" se encuentran los puntos en el espacio de características. La distancia euclidiana es una de las opciones más comunes, pero también pueden emplearse otras métricas como la Manhattan o la de Minkowski, dependiendo de la naturaleza de los datos.

Una vez calculadas todas las distancias, el algoritmo procede a identificar los vecinos más cercanos seleccionando los $k$ puntos con menor distancia a la nueva instancia. La elección del valor de $k$ influye directamente en el comportamiento del modelo: valores pequeños hacen que la predicción sea muy sensible a los puntos más próximos, mientras que valores más grandes suavizan la decisión al considerar un mayor número de vecinos.

El último paso consiste en tomar una decisión basada en los vecinos seleccionados. En un problema de **clasificación**, se asigna la categoría más frecuente entre los vecinos, lo que implica un proceso de votación donde la mayoría determina el resultado. Por ejemplo, si de los cinco vecinos más cercanos tres pertenecen a la clase A y dos a la clase B, la nueva observación será clasificada como A. En el caso de un problema de **regresión**, el valor final se obtiene calculando el promedio de los valores de los vecinos seleccionados, proporcionando así una estimación basada en la cercanía en el espacio de características.

El papel de la distancia en la toma de decisiones es fundamental en el rendimiento del algoritmo, ya que define qué puntos son considerados como más relevantes para la predicción. En datos donde las variables tienen diferentes escalas, la distancia puede verse distorsionada si no se ha aplicado una normalización adecuada, lo que puede llevar a decisiones poco precisas. Además, en problemas con muchas dimensiones, la noción de proximidad puede volverse difusa, un fenómeno conocido como la **"maldición de la dimensionalidad"**, que puede afectar negativamente la capacidad del algoritmo para encontrar vecinos representativos.

> [!note]
>
> La **maldición de la dimensionalidad** es un fenómeno que ocurre cuando el número de dimensiones (o características) de los datos aumenta significativamente, lo que genera problemas en algoritmos como KNN que dependen de medidas de distancia para realizar predicciones. A medida que la dimensionalidad crece, las observaciones en el espacio de características tienden a dispersarse, lo que hace que todas las muestras parezcan igualmente distantes entre sí. Como resultado, el concepto de "vecindad" pierde significado, afectando la capacidad del algoritmo para encontrar vecinos relevantes y, en consecuencia, reduciendo su precisión.
>
> Para comprender mejor este concepto, imaginemos un problema donde se utilizan solo dos características para clasificar observaciones. En un espacio bidimensional, los puntos se distribuyen relativamente cerca unos de otros, y la distancia entre muestras es significativa para identificar patrones. Sin embargo, si agregamos muchas más dimensiones irrelevantes o redundantes, la distancia entre los puntos tiende a volverse más uniforme, lo que dificulta la identificación de los vecinos más cercanos de manera efectiva. Esto se debe a que, en espacios de alta dimensionalidad, los datos se distribuyen en regiones muy dispersas, provocando que la proximidad entre puntos sea menos significativa.
>
> Este efecto tiene varias consecuencias en el rendimiento del algoritmo KNN. Primero, el cálculo de las distancias se vuelve menos discriminativo, lo que significa que los vecinos seleccionados pueden no ser representativos de la verdadera relación entre las muestras. Además, el tiempo de cómputo aumenta drásticamente, ya que el algoritmo necesita evaluar un número mayor de dimensiones para cada comparación, lo que se traduce en un incremento exponencial de la complejidad computacional.
>
> Para mitigar la maldición de la dimensionalidad en KNN, es fundamental realizar una selección adecuada de características, eliminando aquellas que aportan poca o ninguna información relevante al problema. Técnicas como la **reducción de dimensionalidad**, mediante algoritmos como **PCA (Análisis de Componentes Principales)** o **t-SNE**, pueden ayudar a transformar el conjunto de datos en un espacio de menor dimensión sin perder demasiada información valiosa. Además, es recomendable realizar un análisis exploratorio para identificar correlaciones entre las variables y descartar aquellas que no contribuyen significativamente a la separación de clases.



##### **Para reflexionar...**

>  **¿Cómo influye la elección de la métrica de distancia en la precisión del modelo?**
>  **Clave:** Considera la naturaleza de los datos y si las relaciones entre las variables dependen de patrones lineales o no lineales.

> [!tip]
>
> Entender cómo KNN procesa las observaciones y toma decisiones permite una mejor aplicación del algoritmo en diferentes contextos. Sin embargo, su desempeño está altamente influenciado por la correcta elección de la métrica de distancia, el valor de $k$ y la calidad del preprocesamiento de los datos.



### **Elección de la métrica de distancia**  

En el algoritmo KNN, la medida de proximidad entre las observaciones juega un papel fundamental en la calidad de las predicciones. La métrica de distancia utilizada determina cómo se evalúa la similitud entre los puntos en el espacio de características, lo que influye directamente en la selección de los vecinos más cercanos y, por ende, en la precisión del modelo. La elección de una métrica adecuada depende de la naturaleza de los datos y de la relación entre sus características, por lo que es importante comprender las distintas opciones disponibles y su impacto en el rendimiento.  

#### **Distancia euclidiana: la opción por defecto**  

La distancia euclidiana es la más comúnmente utilizada en KNN y se basa en el cálculo de la distancia "en línea recta" entre dos puntos en un espacio multidimensional. Matemáticamente, se expresa como: 

$$
d_{\text{euclidiana}}(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

Esta métrica es intuitiva y fácil de interpretar, ya que mide la diferencia absoluta entre dos observaciones en cada dimensión y luego toma la raíz cuadrada de la suma de estas diferencias al cuadrado. Es especialmente útil cuando las características son cuantitativas y están en la misma escala, ya que da una medida directa de la similitud geométrica entre los puntos. Sin embargo, su desempeño se ve afectado si las variables tienen escalas muy diferentes, lo que puede provocar que una característica domine sobre las demás, haciendo necesario el preprocesamiento mediante normalización o estandarización. 

#### **Otras métricas de distancia**  

Aunque la distancia euclidiana es la opción predeterminada en KNN, existen otras métricas que pueden ser más adecuadas dependiendo del tipo de datos y su distribución. Enumeramos a continuación las más comunes.

La **distancia Manhattan**, también conocida como distancia de "taxista" o "de ciudad", mide la suma de las diferencias absolutas entre los valores de cada característica. Su fórmula es la siguiente:  

$$
d_{\text{manhattan}}(x, y) = \sum_{i=1}^{n} |x_i - y_i|
$$

Esta métrica es útil cuando las características tienen estructuras de tipo cuadrícula o cuando los datos presentan relaciones en forma de bloques, como en imágenes digitales o análisis de redes. A diferencia de la distancia euclidiana, la Manhattan es menos sensible a valores atípicos, ya que evita la penalización cuadrática de las diferencias. 

<img src=".\assets\image-20250120221634850.png" alt="image-20250120221634850" />

La **distancia de Minkowski** es una generalización de las distancias euclidiana y Manhattan, permitiendo ajustar un parámetro \$p\$ para controlar el grado de penalización. Se define como:  

$$
d_{\text{minkowski}}(x, y) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{\frac{1}{p}}
$$

Cuando $p = 2$, se obtiene la distancia euclidiana, y cuando $p = 1$, se reduce a la distancia Manhattan. Esto la convierte en una métrica flexible que puede adaptarse a diferentes estructuras de datos, permitiendo explorar múltiples configuraciones para mejorar el rendimiento del modelo. 

<img src=".\assets\c_distance_metrics_euclidean_manhattan_minkowski_oh_12.jpg" alt="Distance Metrics: Euclidean, Manhattan, Minkowski, Oh My! - KDnuggets" />

La **distancia de Hamming** es una métrica utilizada para medir la **diferencia entre dos secuencias de valores discretos**, particularmente en datos categóricos o binarios. Su cálculo consiste en contar el número de posiciones en las que dos cadenas de igual longitud difieren. En términos simples, mide cuántos elementos deben cambiarse para transformar una secuencia en otra.  

Matemáticamente, la distancia de Hamming entre dos vectores $x$ e $y$ de longitud $n$ se define como:  

$$
d_{\text{Hamming}}(x, y) = \sum_{i=1}^{n} I(x_i \neq y_i)
$$

Donde $I(x_i \neq y_i)$ es una función indicadora que toma el valor 1 si los elementos $x_i$ y $y_i$ son diferentes, y 0 si son iguales. 

Esta métrica es ampliamente utilizada en problemas donde los datos se representan como secuencias discretas, tales como procesamiento de datos binarios, clasificación de datos categóricos o análisis de códigos de seguridad.

> **Ejemplo:** 
> Consideremos dos cadenas binarias de longitud 6: 
> $x$ = 101110 
> $y$ = 100100 
> La distancia de Hamming entre ellas es 2, ya que difieren en las posiciones 3 y 5. 

#### **Comparación de métricas según el tipo de datos**  

Seleccionar la métrica de distancia más apropiada requiere comprender las características de los datos y el problema a resolver. Para datos numéricos y continuos, la distancia euclidiana suele ser una buena opción, siempre que se haya aplicado normalización adecuada. Por otro lado, en situaciones donde las características tienen distribuciones heterogéneas o presentan valores atípicos, la distancia Manhattan puede ofrecer un mejor desempeño. En problemas con datos categóricos o binarios, métricas como la distancia de Hamming son más adecuadas para capturar la similitud entre observaciones.  

Es importante realizar pruebas empíricas utilizando diferentes métricas y evaluar su impacto mediante técnicas como la validación cruzada, ya que en muchos casos la elección óptima dependerá del contexto específico de los datos y del objetivo del modelo.  

##### Para reflexionar...

> **¿Cómo influye la elección de la métrica de distancia en la precisión de un modelo KNN?** 
> **Clave:** Considera la naturaleza de los datos y si las características son continuas o categóricas.  

#### **Impacto de la métrica en el rendimiento**  

La métrica de distancia seleccionada influye directamente en el rendimiento del modelo KNN en términos de precisión, tiempo de cómputo y capacidad de generalización. Una elección inadecuada puede provocar que el modelo falle en identificar correctamente las relaciones entre los datos, llevando a errores de clasificación o predicción.  

Además, la eficiencia computacional también es un factor clave. Mientras que la distancia euclidiana es relativamente rápida de calcular, métricas más complejas, como la de Minkowski con valores altos de $p$, pueden aumentar significativamente el tiempo de procesamiento, especialmente en datasets grandes.  

Un aspecto crítico a considerar es la sensibilidad de las métricas a la escala de los datos. Las distancias basadas en diferencias absolutas, como la Manhattan, pueden verse afectadas en menor medida por características con valores extremos, mientras que la distancia euclidiana puede generar un sesgo hacia atributos con mayor escala. Por esta razón, es recomendable aplicar técnicas de preprocesamiento, como la normalización, para garantizar que todas las características contribuyan de manera equitativa a la distancia calculada.  

##### **Para reflexionar...**

> **¿De qué manera el preprocesamiento de datos puede mejorar la elección de la métrica de distancia en KNN?** 
> **Clave:** Piensa en la importancia de normalizar o estandarizar características antes de calcular distancias.

### **Elección del valor de $k$**  

El parámetro $k$ en el algoritmo KNN representa el número de vecinos más cercanos que se consideran para tomar una decisión. Su elección es un aspecto crucial, ya que afecta directamente el equilibrio entre sesgo y varianza, influyendo en la capacidad del modelo para generalizar correctamente a nuevos datos. Un valor de $k$ adecuado permite encontrar un punto intermedio entre un modelo demasiado flexible y uno excesivamente rígido.  

#### **Impacto de $k$ en el sesgo y la varianza**  

El valor de $k$ tiene una relación directa con el compromiso entre sesgo y varianza en el modelo. Cuando se elige un valor de $k$ pequeño, el modelo tiende a ajustarse demasiado a los datos de entrenamiento, lo que resulta en un sesgo bajo pero una alta varianza. En otras palabras, el modelo es muy sensible a las pequeñas fluctuaciones en los datos, lo que puede provocar un sobreajuste.  

Por el contrario, un valor de $k$ grande suaviza la decisión al considerar un número mayor de vecinos, reduciendo la varianza pero incrementando el sesgo. En este caso, el modelo puede perder la capacidad de capturar patrones locales y proporcionar predicciones más generales, lo que podría llevar a una infra-adaptación a los datos de entrenamiento.  

> **Ejemplo:** 
> Si se elige $k = 1$, cada muestra de prueba se asignará a la clase del vecino más cercano, lo que puede hacer que el modelo sea muy sensible a valores atípicos. Por otro lado, si $k$ es demasiado grande, como 50, la predicción puede verse dominada por la clase más frecuente en el conjunto de entrenamiento, perdiendo precisión en la clasificación de puntos menos comunes.  



#### **Consecuencias de elegir un valor de $k$ inadecuado**  

Seleccionar un valor de $k$ demasiado pequeño puede llevar a un modelo altamente sensible al ruido y a la variabilidad en los datos. Esto se traduce en una clasificación errática en regiones donde los puntos de datos están densamente agrupados o cuando existen valores atípicos. Cada punto de entrenamiento ejerce una influencia significativa, lo que puede provocar predicciones inconsistentes.  

Por otro lado, elegir un valor de $k$ demasiado grande puede hacer que el modelo pierda su capacidad de distinguir entre clases cercanas, ya que puntos distantes pueden ser considerados en la decisión. Esto puede ser problemático cuando las clases no están distribuidas de manera uniforme en el espacio de características, provocando que el modelo falle en capturar la verdadera estructura subyacente de los datos.  

Un valor óptimo de $k$ debe equilibrar la capacidad del modelo para capturar patrones locales sin perder su generalización. Para encontrar ese equilibrio, es esencial realizar un análisis detallado utilizando técnicas de validación.  

##### **Para reflexionar...**

> **¿Cómo podrías determinar si un modelo está sufriendo de sobreajuste o infra-ajuste debido al valor de $k$?** 
> **Clave:** Observa el desempeño en el conjunto de entrenamiento y prueba para identificar si el modelo es demasiado flexible o demasiado rígido.  



#### **Técnicas para seleccionar un valor óptimo de $k$**  

Dado que la elección de $k$ influye en la precisión del modelo, se recomienda aplicar técnicas sistemáticas para determinar su valor óptimo. Dos de los métodos más utilizados para este propósito son la **validación cruzada** y el **método de la curva elbow**.  

##### **Búsqueda en malla**  (`GridSearchCV`)

La búsqueda en malla es una técnica efectiva para evaluar el rendimiento del modelo con diferentes valores de $k$. Consiste en dividir el conjunto de datos en múltiples subconjuntos y entrenar el modelo repetidamente con diferentes combinaciones de entrenamiento y prueba. Al calcular el rendimiento promedio en todas las iteraciones, es posible identificar el valor de $k$ que produce el mejor balance entre precisión y generalización.  

Este método ayuda a evitar que la elección de $k$ se base únicamente en una sola división de datos, reduciendo el riesgo de seleccionar un valor que funcione bien solo para un subconjunto particular. La validación cruzada de $k$-fold es particularmente útil para ajustar el modelo en conjuntos de datos pequeños, donde es importante maximizar el uso de los datos disponibles.  

**Método de la curva elbow**  

El método de la curva elbow es un enfoque gráfico que permite visualizar cómo varía la precisión del modelo en función de diferentes valores de $k$. Consiste en evaluar el error de clasificación o la precisión del modelo para una serie de valores de $k$ y representarlos en un gráfico.  

En general, la curva muestra una disminución rápida del error hasta cierto punto, después del cual la mejora se ralentiza. El valor óptimo de $k$ se elige en el punto donde la curva comienza a estabilizarse, formando un "codo" o punto de inflexión. Este valor representa un equilibrio entre simplicidad y precisión, ya que evita tanto el sobreajuste como el infra-ajuste.  

> **Ejemplo:** 
> Al graficar la tasa de error para valores de $k$ de 1 a 20, se observa que la reducción del error es significativa hasta $k = 7$, después de lo cual las mejoras son marginales. En este caso, $k = 7$ sería un buen candidato.  

#### 

> [!warning]
>
> La elección del valor de $k$ es un paso crítico en la implementación de KNN, ya que determina la capacidad del modelo para encontrar un equilibrio entre precisión y generalización. Es recomendable probar varios valores utilizando técnicas como la validación cruzada y el método de la curva elbow para seleccionar un valor óptimo que permita un buen rendimiento en datos nuevos.



### **Implementación de KNN en scikit-learn**  

Scikit-learn proporciona una implementación sencilla y eficiente del algoritmo KNN a través de las clases `KNeighborsClassifier` para problemas de clasificación y `KNeighborsRegressor` para regresión. Ambas clases permiten ajustar diversos hiperparámetros, como el número de vecinos $k$, la métrica de distancia y la ponderación de los vecinos, lo que permite adaptar el modelo a diferentes tipos de problemas.  

#### **Introducción a las clases `KNeighborsClassifier` y `KNeighborsRegressor`**  

La clase `KNeighborsClassifier` se utiliza cuando el objetivo es asignar una clase a una nueva observación en función de la categoría predominante entre sus vecinos más cercanos. Por otro lado, `KNeighborsRegressor` predice valores numéricos tomando el promedio (o la mediana) de los valores de los vecinos seleccionados.  

En el siguiente ejemplo de código queremos clasificar flores en base a sus características utilizando el dataset Iris.  

```python
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar el dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo KNN con k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Realizar predicciones
y_pred = knn.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo KNN: {accuracy:.2f}")
```

#### **Parámetros clave en KNN**  

Para ajustar el comportamiento del modelo KNN en scikit-learn, es importante conocer algunos de los parámetros más relevantes:  

- **`n_neighbors` (valor de $k$)**: Define el número de vecinos más cercanos a considerar. Un valor pequeño puede hacer que el modelo sea muy sensible a los datos, mientras que un valor grande tiende a suavizar la clasificación.  

- **`metric` (tipo de distancia)**: Controla la métrica de distancia utilizada para calcular la proximidad entre puntos. Algunas opciones comunes incluyen:  
  - `"euclidean"` (por defecto): Distancia euclidiana.  
  - `"manhattan"`: Distancia Manhattan (L1).  
  - `"minkowski"`: Generalización de Euclidiana y Manhattan mediante el parámetro $p$.  
  - `"hamming"`: Para datos categóricos codificados.  

- **`weights` (ponderación de los vecinos)**: Determina cómo se consideran los vecinos en la predicción. Puede tomar los valores:  
  - `"uniform"`: Todos los vecinos tienen el mismo peso (predeterminado).  
  - `"distance"`: Los vecinos más cercanos tienen mayor influencia en la predicción.  

Imagina que en el ejemplo anterior se quiere ajustar el modelo utilizando la distancia Manhattan y ponderación por distancia.  Entonces tendríamos que:

```python
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión con métrica Manhattan y ponderación por distancia: {accuracy:.2f}")
```



#### **Preprocesamiento de datos para KNN (escalado y normalización)**  

KNN es muy sensible a la escala de las características, ya que la métrica de distancia utilizada se ve afectada si las variables tienen diferentes rangos. Por esta razón, es fundamental aplicar técnicas de preprocesamiento, como la normalización o la estandarización. Ambas técnicas ya fueron detalladas en el tema correspondiente. Brevemente:

- **Normalización:** Convierte los valores de las características en un rango entre 0 y 1, útil cuando los datos tienen distribuciones heterogéneas.  
- **Estandarización:** Escala los datos para que tengan media 0 y desviación estándar 1, ideal cuando los datos siguen una distribución normal.  

Podemos aplicar en nuestro ejemplo un proceso de estandarización con `StandardScaler` antes de entrenar el modelo.  

```python
from sklearn.preprocessing import StandardScaler

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo con datos escalados
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Evaluar el modelo en datos escalados
y_pred_scaled = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred_scaled)
print(f"Precisión con datos escalados: {accuracy:.2f}")
```

#### **Evaluación del modelo KNN**  

La evaluación de KNN en problemas de clasificación y regresión requiere utilizar métricas adecuadas para medir el rendimiento del modelo.

En el caso de **clasificación**, evaluar el rendimiento del modelo KNN requiere el uso de varias métricas que permiten entender su capacidad para asignar correctamente las clases, así como identificar posibles errores en la predicción. La **precisión (accuracy)** es una de las métricas más utilizadas, ya que mide el porcentaje de predicciones correctas sobre el total de observaciones evaluadas. Sin embargo, en problemas con clases desbalanceadas, es recomendable analizar métricas adicionales como el **recall** y el **F1-score**, que permiten entender el balance entre la sensibilidad y la precisión del modelo. La **matriz de confusión** proporciona una visión detallada del rendimiento del modelo, mostrando el número de aciertos y errores para cada clase, lo que facilita la identificación de patrones de error.  

Para **regresión**, KNN se evalúa utilizando métricas que cuantifican la magnitud de los errores cometidos en las predicciones numéricas. El **error absoluto medio (MAE)** mide el promedio de las diferencias absolutas entre los valores predichos y reales, ofreciendo una interpretación clara de cuánto se está equivocando el modelo en términos absolutos. Por otro lado, el **error cuadrático medio (MSE)** penaliza de forma más severa los errores grandes al elevar al cuadrado las diferencias, lo que permite capturar desviaciones importantes en las predicciones. Finalmente, la métrica $R^2$ evalúa la capacidad del modelo para explicar la variabilidad de los datos, proporcionando una medida relativa de la calidad de las predicciones en comparación con una línea base simple, como la media de los valores observados.  

En el ejemplo de este apartado podemos evaluar el modelo utilizando la matriz de confusión y F1-score.  

```python
from sklearn.metrics import confusion_matrix, classification_report

# Imprimir matriz de confusión y reporte de clasificación
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_scaled))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred_scaled))
```

#### **Validación cruzada y ajuste de hiperparámetros con `GridSearchCV`**  

Para encontrar el mejor valor de $k$ y otras configuraciones óptimas del modelo, se puede aplicar la validación cruzada con búsqueda en malla (`GridSearchCV`).  

Veámoslo en funcionamiento en el siguiente ejemplo:

```python
from sklearn.model_selection import GridSearchCV

# Definir la cuadrícula de hiperparámetros
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11],
    'metric': ['euclidean', 'manhattan'],
    'weights': ['uniform', 'distance']
}

# Configurar GridSearchCV con validación cruzada
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Mostrar los mejores parámetros encontrados
print("Mejor configuración:", grid_search.best_params_)
print("Mejor precisión obtenida:", grid_search.best_score_)
```

#### **Comparación del rendimiento en diferentes escenarios**  

Para determinar la mejor configuración del modelo, es útil comparar el rendimiento utilizando diferentes combinaciones de hiperparámetros, métricas de distancia y técnicas de preprocesamiento. Algunas recomendaciones incluyen:  

- Evaluar el impacto del escalado en el rendimiento del modelo.  
- Comparar la precisión con diferentes métricas de distancia.  
- Analizar cómo cambia el desempeño al variar el valor de $k$.  

A continuación puede verse una comparación con y sin normalización.  

```python
knn_no_scaling = KNeighborsClassifier(n_neighbors=5)
knn_no_scaling.fit(X_train, y_train)
accuracy_no_scaling = accuracy_score(y_test, knn_no_scaling.predict(X_test))

knn_with_scaling = KNeighborsClassifier(n_neighbors=5)
knn_with_scaling.fit(X_train_scaled, y_train)
accuracy_with_scaling = accuracy_score(y_test, knn_with_scaling.predict(X_test_scaled))

print(f"Precisión sin escalado: {accuracy_no_scaling:.2f}")
print(f"Precisión con escalado: {accuracy_with_scaling:.2f}")
```

### **Consideraciones prácticas y ajuste de hiperparámetros**  

Para garantizar el mejor desempeño del algoritmo KNN, es crucial considerar una serie de aspectos prácticos relacionados con el preprocesamiento de los datos, la elección de parámetros clave y la eficiencia computacional. A diferencia de otros modelos de aprendizaje supervisado, KNN **no realiza un entrenamiento explícito**, por lo que su rendimiento depende en gran medida de la calidad y la preparación de los datos, así como de la elección adecuada de hiperparámetros como el valor de $k$, la métrica de distancia y la ponderación de los vecinos.  

#### **Estandarización y escalado de datos**  

Uno de los aspectos más importantes a considerar en KNN es la **escalabilidad de las características**, ya que el modelo se basa en medidas de distancia para realizar predicciones. Si las variables tienen escalas muy diferentes, aquellas con valores más grandes pueden dominar la medida de distancia y afectar negativamente el rendimiento del modelo. Por esta razón, es recomendable aplicar técnicas de **normalización** o **estandarización**, dependiendo de la naturaleza de los datos.  

La **normalización**, que convierte los valores de las características a un rango entre 0 y 1, es útil cuando los datos **no siguen una distribución normal** y las diferencias entre las escalas son significativas. Por otro lado, la **estandarización**, que transforma los datos para que tengan media cero y desviación estándar uno, es preferible cuando las características presentan una distribución aproximadamente gaussiana. 

#### **Peso de los vecinos: uniforme vs ponderado**  

En KNN, los vecinos más cercanos pueden contribuir de manera uniforme o ponderada a la predicción.  

Cuando se utiliza la opción de **peso uniforme**, cada vecino contribuye por igual a la decisión final, lo que puede ser adecuado en casos donde los datos están distribuidos de manera homogénea. Sin embargo, en situaciones donde los puntos más cercanos tienen mayor relevancia, es preferible utilizar una **ponderación por distancia**, donde los vecinos más cercanos tienen mayor influencia en la predicción que los más alejados.  

El uso de pesos ponderados es especialmente útil en problemas donde la densidad de puntos varía significativamente en diferentes regiones del espacio de características, evitando que vecinos distantes influyan indebidamente en la predicción.  

#### **Reducción de dimensionalidad para mejorar rendimiento (PCA)**  

A medida que el número de características aumenta, KNN se vuelve más propenso a la **maldición de la dimensionalidad**, donde las distancias entre los puntos pierden significado y se dificulta la identificación de vecinos relevantes. Para mitigar este problema, una estrategia común es aplicar técnicas de **reducción de dimensionalidad**, como el **Análisis de Componentes Principales (PCA)**, que permite transformar los datos en un espacio de menor dimensión mientras conserva la mayor parte de la variabilidad original.  

Al reducir el número de dimensiones, se puede mejorar el rendimiento computacional del modelo, además de reducir el ruido presente en los datos, lo que ayuda a mejorar la precisión de las predicciones.  

> **Ejemplo:** Aplicando PCA antes de entrenar el modelo KNN.  

```python
from sklearn.decomposition import PCA

# Aplicar PCA para reducir a 2 dimensiones
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Entrenar modelo en el espacio reducido
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)

print(f"Precisión después de PCA: {knn_pca.score(X_test_pca, y_test):.2f}")
```

Aplicar PCA antes de KNN puede ser especialmente útil cuando hay características altamente correlacionadas, ya que ayuda a capturar la variabilidad de los datos en un espacio más compacto y significativo. Sin embargo, se debe tener cuidado de no perder información relevante en el proceso de reducción. 

#### **Problemas de memoria y eficiencia con grandes volúmenes de datos**  

Uno de los desafíos más importantes de KNN es su **alta demanda de memoria y tiempo de cómputo**, especialmente en conjuntos de datos grandes. Dado que el algoritmo necesita almacenar todas las muestras de entrenamiento y calcular la distancia entre cada nueva instancia y todas las observaciones existentes, su complejidad computacional es de orden $O(n \cdot d)$, donde $n$ es el número de muestras y $d$ el número de dimensiones.  

Para mejorar la eficiencia de KNN en grandes volúmenes de datos, se pueden aplicar varias estrategias, entre ellas podemos enumerar las siguientes:

1. **Estructuras de datos optimizadas:** Uso de estructuras como KD-Trees o Ball-Trees, que permiten acelerar la búsqueda de vecinos cercanos mediante particionamiento del espacio de características, reduciendo el tiempo de búsqueda a $O(\log n)$ en ciertos casos. Scikit-learn implementa automáticamente estas estructuras cuando el número de dimensiones es bajo.  

2. **Aproximaciones mediante algoritmos de búsqueda rápida:** Técnicas como Approximate Nearest Neighbors (ANN) permiten encontrar vecinos cercanos de manera aproximada con una reducción significativa del costo computacional.  

3. **Reducción del conjunto de entrenamiento:** Selección de instancias representativas o eliminación de puntos redundantes mediante técnicas de clustering o prototipado. 
