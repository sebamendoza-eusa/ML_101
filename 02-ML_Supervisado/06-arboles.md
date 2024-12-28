# Tema 2. Sistemas de aprendizaje automático supervisado

## Árboles de Decisión y Bosques Aleatorios

### **Objetivos del Módulo**

1. Comprender los fundamentos de los Árboles de Decisión.
2. Explorar los problemas de overfitting y underfitting en los árboles.
3. Introducir los ensambles y cómo los Random Forests mejoran el rendimiento.
4. Aplicar Árboles de Decisión y Random Forests a problemas reales.
5. Comparar los resultados de ambos modelos en términos de precisión, robustez y eficiencia.

------

### Árboles de decisión: Una introducción práctica

Un **árbol de decisión** es una estructura de datos compleja basada en el establecimiento de un conjunto de reglas de decisión claras y organizadas. Su representación visual facilita la comprensión de cómo se toman las decisiones, ofreciendo una perspectiva jerárquica y lógica que es intuitiva incluso para quienes no tienen un trasfondo técnico profundo.

La estructura de un árbol de decisión está compuesta por **nodos**, **ramas** y **hojas**. Los **nodos** son los puntos donde se evalúan las características del dataset, y desde ellos surgen las **ramas**, que representan las posibles categorías o valores que puede tomar la característica evaluada. El proceso comienza en el **nodo raíz**, que es el punto inicial del árbol y desde donde se ramifican todas las decisiones. Por otro lado, los **nodos hoja**, ubicados al final de cada rama, representan los resultados finales de las decisiones, ya sea una clase (en un problema de clasificación) o un valor (en un problema de regresión).

Así, un árbol de decisión típico empieza con un único nodo raíz en la parte superior, que se ramifica hacia un conjunto más o menos amplio de nodos de decisión intermedios, hasta llegar a los nodos terminales, donde se llevan a cabo las decisiones finales. Cada nodo intermedio evalúa una sola variable, y las ramas que surgen de él representan los posibles valores o categorías de esa variable. Estas categorías pueden ser cualitativas, como "hombre" o "mujer", "grande" o "pequeño", o bien cuantitativas, como intervalos de valores: "mayor que 5", "entre 3 y 10", "menor que 8", etc. Finalmente, el nodo hoja al final del proceso devuelve el resultado que el algoritmo ha determinado.

Desde un punto de vista de la **ciencia de datos**, un **árbol de decisión** es un modelo de aprendizaje supervisado utilizado tanto para tareas de clasificación como de regresión. La estructura del árbol se asemeja a un diagrama jerárquico en el que las decisiones se toman a partir de divisiones sucesivas basadas en las características del dataset. Estas divisiones intentan separar los datos de manera que se minimice la incertidumbre y se mejore la pureza de los grupos resultantes.

La lógica de los árboles de decisión es sencilla: en cada nodo interno, se evalúa una característica para dividir los datos en subconjuntos más homogéneos. Estas posibles divisiones se evalúan en función de una métrica determinada que mide la calidad de la separación. De este modo, el proceso continúa de manera recursiva hasta alcanzar una condición de parada, como por ejemplo, un nivel máximo de profundidad o un número mínimo de datos en los nodos.

#### Un ejemplo práctico: Decisiones de carácter agrícola con un árbol de decisión

Imaginemos un escenario en el que queremos predecir si un lugar es **bueno**, **medio** o **malo** para la cosecha de un cereal, basándonos en características como el nivel de lluvias, el tipo de terreno, la disponibilidad de aguas subterráneas y el uso de fertilizantes. Supongamos que tenemos datos de 20 lugares diferentes en España, donde tomamos muestras en regiones cultivables.

Con estos datos iniciales, podríamos empezar a hacer conjeturas sobre los factores que influyen en una buena cosecha. Por ejemplo, podríamos observar que las **lluvias** son un factor importante: cuando son **altas**, parece haber una mejor producción. Asimismo, si las lluvias son **medias**, la disponibilidad de **agua subterránea** parece marcar una diferencia. Si también se usan fertilizantes, los resultados son aún mejores. Por otro lado, si el terreno es una **meseta**, podría ser clave la combinación de agua subterránea y fertilizantes. Sin embargo, con tan solo 20 datos ya podría ser complicado formular estas reglas de manera sistemática, y mucho más si estuviéramos analizando miles o millones de registros.

Un árbol de decisión nos ayuda a automatizar este proceso y establecer reglas claras. Por ejemplo, podríamos construir un árbol que tome las siguientes decisiones:

1. Si el **terreno es llano** y las **lluvias son altas**, entonces la cosecha es **óptima**.
2. Si el terreno es llano y la lluvias son bajas, evaluamos el nivel de agua subterránea:
   - Si hay agua subterránea, la cosecha es **moderada**.
   - Si no hay agua subterránea, la cosecha será **pobre**.
3. Si el terreno es una colina, el nivel de lluvias es determinante:
   - Lluvias altas → Cosecha **moderada**.
   - Lluvias bajas → Cosecha **pobre**.
4. Si el terreno es una meseta, evaluamos la combinación de agua subterránea y fertilizantes:
   - Agua subterránea presente y uso de fertilizantes → Cosecha **óptima**.
   - Agua subterránea presente, pero sin fertilizantes → Cosecha **moderada**.
   - Sin agua subterránea → Cosecha **pobre**.

De este modo, el árbol clasifica cada una de las 20 regiones en categorías según estas reglas de decisión. Además, estas reglas pueden representarse tanto de forma gráfica (con el árbol) como en forma de un conjunto de **reglas lógicas** (conjunciones e intersecciones), ofreciendo flexibilidad en su interpretación. El esquema gráfico sería el siguiente:

<img src=".\assets\image-20241227124647697.png" alt="image-20241227124647697" />

##### Variables categóricas y numéricas en árboles de decisión

En el ejemplo anterior, todas las variables predictoras son categóricas, como "llano", "colina", "meseta", "alto" o "bajo". Esto facilita la interpretación del árbol, ya que cada nodo y rama corresponde a una categoría específica. Sin embargo, los árboles de decisión también pueden manejar **variables numéricas**. Por ejemplo, si el dataset incluyera el nivel exacto de lluvia en milímetros, el árbol podría establecer divisiones como "más de 50 mm" o "menos de 30 mm".

Aunque los árboles pueden trabajar con variables numéricas, convertir estas variables en categorías puede ser beneficioso para mejorar la claridad y precisión del modelo. Por ejemplo, si trabajamos con ingresos, podríamos transformarlos en categorías como "bajos" (menos de 30.000 € al año), "medios" (entre 30.000 y 100.000 €) y "altos" (más de 100.000 €). Esto ayuda a generar reglas más intuitivas y reduce la complejidad de las divisiones.

##### ¿Cómo usar el árbol para predecir?

Para realizar una predicción, seguimos las reglas del árbol según las características de una nueva observación. Por ejemplo:

- Nueva observación: Tiempo = Soleado, Humedad = Alta.

  Seguimos la rama "Soleado" y luego "Humedad = Alta", lo que predice **Hacer deporte = No**.

- Nueva observación: Clima = Lluvioso, Temperatura = Frío. 

  Seguimos la rama "Lluvioso" y luego "Temperatura = Frío", lo que predice **Hacer deporte = No**.

##### Algunas reflexiones

> Este ejemplo les permite ver cómo se construye un árbol paso a paso. Pero se pueden discutir preguntas como:
>
> **¿Por qué elegimos "Tipo de suelo" como el nodo raíz?** 
>
> **¿Qué sucede si un grupo sigue siendo muy heterogéneo después de una división?**  



> [!tip]
>
> ##### Ventajas de los árboles de decisión
>
> Los árboles de decisión son herramientas extremadamente intuitivas y visuales. Ofrecen varias ventajas clave:
>
> 1. **Facilidad de interpretación**: Su representación gráfica permite entender cómo se toman las decisiones en cada paso.
> 2. **Flexibilidad ante datos incompletos o con ruido**: No requieren limpieza exhaustiva, ya que pueden clasificar incluso con valores faltantes o datos atípicos.
> 3. **Versatilidad**: Pueden manejar tanto variables categóricas como numéricas, lo que los hace útiles en una amplia variedad de problemas.

### Estructura de los árboles de decisión y tipos

En un árbol de decisión, cada uno de sus elementos tiene un papel fundamental en el proceso de tomar decisiones y generar predicciones. El primer componente esencial es el **nodo**, que representa un punto donde se evalúa una característica del dataset para dividir los datos en subconjuntos más homogéneos. Estos nodos actúan como las encrucijadas del árbol, estableciendo reglas claras basadas en los valores de las características.

A partir de los nodos surgen las **ramas**, que conectan un nodo con otros nodos o con las hojas del árbol. Cada rama refleja un posible resultado de la decisión tomada en el nodo que la origina. Por ejemplo, si un nodo evalúa si la temperatura es **"fría"**, las ramas podrían representar las opciones "Sí" o "No", guiando a las observaciones hacia diferentes rutas según sus características.

En los extremos del árbol se encuentran las **hojas**, que son los nodos terminales y representan el resultado final del proceso de decisión. En el caso de un problema de clasificación, las hojas indican una clase específica (como "Apto" o "No apto"), mientras que en un problema de regresión, contienen un valor numérico continuo que es la predicción final.

Un aspecto crítico del árbol es su **profundidad**, que mide el número máximo de niveles desde el nodo raíz hasta las hojas. La profundidad determina cuán complejas pueden ser las reglas del modelo. Un árbol más profundo puede aprender patrones muy detallados del dataset, pero también **corre el riesgo de ajustarse demasiado** a los datos de entrenamiento, capturando ruido en lugar de patrones generalizables. Por otro lado, un árbol poco profundo puede ser demasiado simple, incapaz de modelar relaciones importantes, lo que lleva al subajuste. Por ello, encontrar la profundidad adecuada es crucial para equilibrar la capacidad del árbol para aprender patrones complejos y evitar el sobreajuste.

Los árboles de decisión son modelos versátiles que pueden adaptarse a diferentes tipos de problemas y escenarios según su diseño y la naturaleza de la variable objetivo. Aunque todos los árboles comparten principios básicos de funcionamiento, existen varias tipologías que los distinguen en función de su objetivo, estructura o configuración. A continuación, exploramos las principales categorías de árboles de decisión.

#### Tipos de árboles de decisión

Según el objetivo del problema que se quiera resolver podremos distinguir entre dos tipos de árboles de decisión: Los **árboles de clasificación** y los **árboles de regresión** 

##### Árboles de clasificación

Los árboles de clasificación son aquellos diseñados para resolver problemas donde la variable objetivo es **categórica**. Su propósito es dividir los datos en grupos homogéneos que correspondan predominantemente a una sola clase. Este tipo de árbol evalúa las divisiones basándose en métricas como la **entropía**, el **índice de Gini** o la **ganancia de información**, y busca maximizar la pureza de las clases en los nodos hijos.

Las **aplicaciones comunes** de este tipo de árboles pueden ser el diagnóstico médico (¿El paciente tiene una enfermedad o no?), la detección de fraudes (¿Una transacción es legítima o fraudulenta?) o la clasificación de textos (¿Un correo es spam o no?).

> **Ejemplo**: Un árbol de clasificación podría predecir si un cliente comprará un producto basándose en características como edad, ingreso y frecuencia de visitas a la tienda.

##### Árboles de regresión

En contraste, los árboles de regresión se utilizan para problemas donde la variable objetivo es **continua**. En este caso, el objetivo del árbol es dividir los datos en grupos donde los valores de la variable objetivo sean lo más compactos posible alrededor de su promedio. Las métricas utilizadas para evaluar las divisiones incluyen la **varianza** o el **error cuadrático medio (MSE)**, buscando minimizar la dispersión dentro de los nodos hijos.

Igualmente existen **aplicaciones comunes** para este tipo de árboles como pueden ser la predicción de precios (precio de viviendas, acciones, etc.), la estimación de demanda (ventas esperadas de un producto) o el modelado de variables continuas en ciencias naturales (temperatura, precipitación, etc.).

> **Ejemplo**: Un árbol de regresión podría predecir el precio de una casa basándose en características como el número de habitaciones, la superficie y la ubicación.

Podemos también establecer una clasificación en los árboles de decisión teniendo en cuenta el número de ramas que sale de cada nodo. Así, podríamos distinguir entre **árboles binarios** y **árboles multinarios**.

##### Árboles binarios

Los árboles binarios son una tipología estructural donde cada nodo se divide en exactamente **dos ramas**. Este diseño es el más común y facilita el proceso de división recursiva al evaluar cada característica con un único umbral. Sus ventajas son principalmente la simplicidad en la implementación y cálculo y la mayor compatibilidad con algoritmos de ensamblaje, como *Random Forests*.

> **Ejemplo**:
> Un árbol binario que evalúe la característica "Edad" podría dividir los datos según la condición "¿Edad > 30?".

##### Árboles multinarios

En los árboles multinarios, los nodos pueden dividirse en más de dos ramas, lo que es especialmente útil para variables categóricas con múltiples niveles. Por ejemplo, una característica como "Región" podría dar lugar a ramas para "Norte", "Sur", "Este" y "Oeste". Estos árboles tienden a ser más compactos, pero pueden complicar el análisis y la interpretación. Es cierto que tienen ciertas ventajas, como por ejemplo, la reducción en la profundidad del árbol o que permiten resolver problemas donde existen características categóricas con muchos valores posibles.

> **Ejemplo**: Un árbol multinario podría dividir los datos según el "Departamento de trabajo" en un dataset de empleados, generando ramas para "Recursos Humanos", "Finanzas", "Ventas" y "Producción".

### Bases matemáticas de los árboles de decisión.

Los árboles de decisión son modelos de aprendizaje supervisado que dividen iterativamente el espacio muestral en subconjuntos más homogéneos, guiándose por principios matemáticos. Dichos principios explican no solo su capacidad para dividir datos, sino también medir la calidad de esas divisiones y optimizar la homogeneidad en los subconjuntos resultantes. Entender estos principios no solo ayuda a interpretar mejor los resultados del modelo, sino también a tomar decisiones fundamentadas sobre su diseño y aplicación en diferentes escenarios.

#### División recursiva

La construcción de un árbol de decisión se basa en un proceso de **división recursiva**, donde los datos se separan iterativamente **en subconjuntos más homogéneos** según las características evaluadas. En cada paso, el algoritmo determina cuál característica utilizar para dividir los datos, basándose en una métrica que mide la calidad de la separación. El objetivo de estas métricas es identificar las divisiones que reduzcan al máximo la incertidumbre o el desorden en los grupos resultantes.

Un conjunto se considera **puro** cuando todas las observaciones pertenecen a una sola clase, y la **ganancia de homogeneidad** refleja cuánto más cerca estamos de esa pureza después de realizar una división.

##### ¿Qué significa homogeneidad?

Imagina un conjunto de datos donde queremos predecir si las personas compran o no un producto. Al principio, el grupo es heterogéneo: algunas personas compran (clase "Sí") y otras no (clase "No"). Si logramos dividir el conjunto en grupos más homogéneos —por ejemplo, un grupo donde casi todos compran y otro donde casi nadie compra—, hemos ganado homogeneidad.

En términos prácticos, la **ganancia de homogeneidad** mide cuánto más simples y predecibles son los grupos que creamos después de una división en comparación con el conjunto original.

> **Ejemplo:** Piensa en una bolsa de caramelos de diferentes colores: rojo, verde y amarillo. Al principio, los caramelos están mezclados al azar en la bolsa, lo que refleja alta incertidumbre o desorden. Ahora, si divides los caramelos en tres bolsas separadas por color (una para los rojos, otra para los verdes, y una para los amarillos), cada bolsa será homogénea. Al realizar esta separación, has reducido el desorden y aumentado la pureza de los grupos.
>
> En un árbol de decisión, el algoritmo busca realizar algo similar con los datos: encontrar divisiones que agrupen las observaciones de manera que las clases sean lo más homogéneas posible dentro de cada grupo.

En conclusión: La división recursiva está impulsada por el concepto de **ganancia de homogeneidad**, que mide cuánto más ordenado es el dataset después de una división. Este concepto universal es aplicable tanto en árboles de clasificación como de regresión, aunque la forma exacta de medirlo (como entropía, índice de Gini o varianza) depende del tipo de problema. El objetivo principal de cada división es maximizar esta ganancia, asegurando que los nodos hijos representen subconjuntos más homogéneos y, por ende, más fáciles de predecir. Este principio guía todo el proceso de construcción del árbol, desde el nodo raíz hasta las hojas terminales. Aunque la manera de medir esta homogeneidad puede variar dependiendo del tipo de árbol, existen elementos estructurales del proceso que son comunes a todos ellos.

##### Selección de características y umbrales

Todo proceso de división recursiva comienza evaluando todas las características disponibles y determinando los **umbrales** o valores que mejor separen los datos en subconjuntos más homogéneos. Si la característica es numérica, el algoritmo considera divisiones del tipo "Edad > 30" o "Ingresos < 50.000". Para características categóricas, las divisiones pueden basarse en niveles, como "Género = Masculino" o "Estado = Aprobado".

El objetivo en este paso es encontrar la combinación característica-umbral que genere nodos hijos con la mayor ganancia de homogeneidad posible, es decir, donde los valores de la variable objetivo sean más uniformes o predecibles.

##### Criterio para medir la calidad de la división

Para decidir qué división aplicar, el algoritmo utiliza una **métrica** que evalúa la calidad de las particiones. Este criterio mide cuánto se redujo la heterogeneidad o incertidumbre al pasar del nodo padre a los nodos hijos. Aunque la métrica específica depende del tipo de problema (clasificación o regresión), el concepto subyacente es universal: se prefieren **divisiones que agrupan observaciones más similares entre sí**.

Por ejemplo, en problemas de clasificación, esta homogeneidad puede evaluarse en términos de pureza de las clases dentro de los nodos. Por su parte, en problemas de regresión, se mide la compactación de los valores continuos alrededor de un promedio.

La calidad de la división no solo depende de la homogeneidad dentro de los nodos hijos, sino también de su tamaño relativo: **nodos más grandes tienen un peso mayor en el cálculo**.

##### Recursividad en la construcción del árbol y condiciones de parada

Una vez que se selecciona la mejor división en un nodo, el proceso se repite de manera recursiva para cada nodo hijo. En cada paso, se aplican las mismas reglas: evaluar las posibles divisiones y seleccionar la que optimice la homogeneidad. Este enfoque recursivo continúa hasta que se cumplen ciertas condiciones de parada.

Esta naturaleza jerárquica y recursiva permite a los árboles de decisión capturar interacciones complejas entre las características y dividir el espacio de datos en regiones que reflejan patrones subyacentes en la variable objetivo.

El proceso de división no puede continuar indefinidamente, ya que esto podría llevar a un modelo que memorice el dataset de entrenamiento (sobreajuste). Para evitarlo, se imponen restricciones que definen cuándo debe detenerse la recursividad. Normalmente podemos encontrarnos con tres situaciones. La primera, un escenario de **pureza máxima**. Esto es, si todos los ejemplos en un nodo pertenecen a la misma clase (en clasificación) o tienen el mismo valor (en regresión), la división se detiene. La segunda posibilidad es limitar la **profundidad máxima** del árbol, es decir, limitar la cantidad de niveles del árbol para evitar que se vuelva demasiado complejo. Por último, se puede definir un **tamaño mínimo del nodo**, de modo que, si un nodo tiene menos ejemplos que un umbral predefinido, no se permite una nueva división.

Estas condiciones garantizan que el árbol tenga un equilibrio entre su capacidad de aprender patrones complejos y su capacidad de generalizar a nuevos datos.

> [!important]
>
> La **división recursiva** y la **ganancia de homogeneidad** son los pilares comunes que sustentan a todos los tipos de árboles de decisión. Aunque los criterios específicos varían según el tipo de problema, el enfoque general busca dividir los datos de manera progresiva y eficiente, creando una estructura jerárquica que facilita la predicción y captura patrones significativos en los datos. Este proceso metódico asegura que los árboles sean herramientas versátiles y efectivas tanto en problemas de clasificación como de regresión.

##### Ejemplo práctico: ¿Cómo se mide la ganancia de homogeneidad?

Para fijar ideas vamos a suponer un ejemplo práctico de un árbol de decisión en una tarea de clasificación. En este caso , la ganancia de homogeneidad se mediría **comparando el desorden antes y después de una división**. La diferencia entre el desorden inicial y el final nos da la **ganancia de información**, o lo que es lo mismo, la ganancia de homogeneidad.

Supongamos que tenemos el siguiente dataset de frutas que queremos clasificar como "Comestible" o "No comestible":

| Color    | Tamaño  | Comestible |
| -------- | ------- | ---------- |
| Rojo     | Grande  | Sí         |
| Rojo     | Pequeño | Sí         |
| Verde    | Grande  | No         |
| Verde    | Pequeño | No         |
| Amarillo | Grande  | Sí         |

En el dataset original, hay tres frutas comestibles y dos no comestibles. Las clases están mezcladas, lo que refleja un nivel moderado de desorden.

Si dividimos las frutas según su color, obtenemos los siguientes grupos:

1. **Grupo "Rojo"**: {Sí, Sí} → Completamente homogéneo (todas las frutas son comestibles)
2. **Grupo "Verde"**: {No, No} → Completamente homogéneo (ninguna fruta es comestible)
3. **Grupo "Amarillo"**: {Sí} → También homogéneo 

La ganancia de información en este caso es máxima porque la división ha creado grupos perfectamente homogéneos.

###### Dividimos por Tamaño

Si en lugar de dividir por **Color** usamos **Tamaño**, obtenemos:

1. **Grupo "Grande"**: {Sí, No, Sí} → Mezclado (dos comestibles, uno no comestible). 
2. **Grupo "Pequeño"**: {Sí, No} → Mezclado (uno comestible, uno no comestible). 

En este caso, los grupos resultantes son menos homogéneos, lo que implica una menor ganancia de información en comparación con la división por **Color**

###### Contraejemplo: Mala división de las clases

Imagina que intentamos dividir las frutas por una característica irrelevante, como el peso exacto (en gramos). Esto podría generar muchos grupos pequeños con una mezcla aleatoria de clases en cada grupo. En este caso los subconjuntos resultantes tienen alto desorden porque las clases siguen mezcladas. La ganancia de homogeneidad sería mínima, lo que indica que esta característica no es adecuada para dividir los datos.

> [!tip]
>
> La ganancia de homogeneidad es el objetivo clave de cada división en un árbol de decisión. Al medir la calidad de las divisiones, el algoritmo asegura que las características seleccionadas sean las que mejor separen las clases, creando grupos lo más homogéneos posible. Esta capacidad de transformar un conjunto desordenado en grupos organizados y predecibles es lo que hace a los árboles de decisión herramientas poderosas para la clasificación y la regresión.



#### Criterios de división: Árboles de regresión vs. árboles de clasificación

La división recursiva y la ganancia de homogeneidad son conceptos fundamentales en los árboles de decisión, pero su implementación y evaluación varían según el tipo de problema que se aborde. Mientras que los árboles de clasificación se enfocan en maximizar la pureza de las clases dentro de los nodos, los árboles de regresión buscan reducir la variabilidad de los valores continuos. En cualquier caso, el objetivo es crear nodos más homogéneos, lo que facilita predicciones más precisas y generalizables.

##### Árboles de regresión

En los árboles de regresión, donde **la variable objetivo es continua**, el objetivo principal de cada división es minimizar la **variabilidad** de los valores dentro de los nodos resultantes. A través de la división recursiva, el árbol busca agrupar los datos de manera que los valores de la variable objetivo sean lo más homogéneos posible dentro de cada nodo, logrando así que las predicciones sean más precisas.

Así, la calidad de una división en árboles de regresión se mide evaluando cómo afecta a la **dispersión** de los valores continuos de la variable objetivo en los nodos hijos. Las métricas más comunes incluyen: **Varianza, error cuadrático medio (MSE) y reducción de la suma de errores**

La ganancia de homogeneidad en árboles de regresión tiene  dos ventajas interesantes: Por un lado, genera  predicciones más precisas, ya que,  al reducir la variabilidad dentro de los nodos, el promedio de cada nodo se convierte en una estimación más representativa. Por otro lado, se pueden captura patrones locales. En efecto, al dividir los datos en regiones con valores similares, el árbol puede adaptarse a relaciones complejas entre las características y la variable objetivo.

###### **Varianza**  

La **varianza** mide cuánto se dispersan los valores de la variable objetivo en un nodo respecto a su promedio. La fórmula para la varianza de un nodo $S$ es:

$$
\text{Varianza}(S) = \frac{1}{|S|} \sum_{i=1}^{|S|} (y_i - \bar{y})^2
$$

Donde:

- $y_i$ son los valores individuales de la variable objetivo.
- $\bar{y}$ es el promedio de los valores en el nodo.

Al realizar una división, el objetivo es **reducir la suma ponderada de las varianzas en los nodos hijos**. Esto asegura que los valores en cada nodo sean más compactos alrededor de sus promedios.

###### **Error cuadrático medio (MSE)**  

El **MSE** es otra métrica común que evalúa la calidad de las divisiones considerando el error promedio entre los valores reales de la variable objetivo y el promedio de cada nodo hijo. Matemáticamente, el MSE se define como:

$$
	\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y})^2
$$

Donde:

- $y_i$ son los valores reales de la variable objetivo.
- $\hat{y}$ es el valor promedio en el nodo correspondiente.

Al igual que la varianza, el objetivo es minimizar el MSE ponderado de los nodos hijos.

###### **Reducción de la suma de errores (RSS)** 

La **RSS** mide cuánto se reduce la suma total de errores al realizar una división. Su cálculo implica evaluar la diferencia entre el error del nodo padre y la suma de los errores de los nodos hijos:

$$
\text{RSS} = \sum_{i \in S} (y_i - \bar{y})^2 - \sum_{j=1}^k \sum_{i \in S_j} (y_i - \bar{y_j})^2
$$

Donde $\bar{y}$ es el promedio en el nodo padre y $\bar{y_j}$ es el promedio en cada nodo hijo.

###### Ejemplo práctico: Predicción del precio de viviendas

> Supongamos que estamos construyendo un árbol de regresión para predecir el precio de una vivienda ($Precio$) basándonos en una única característica numérica: **Superficie** (en metros cuadrados). Nuestro dataset contiene las siguientes observaciones:
>
> | Superficie (m²) | Precio (en miles) |
> | --------------- | ----------------- |
> | 50              | 150               |
> | 60              | 180               |
> | 70              | 210               |
> | 80              | 240               |
> | 90              | 270               |
>
> ###### Paso 1: Varianza inicial
>
> Primero calculamos la varianza de los valores de $Precio$ en el nodo raíz (que contiene todos los datos). La fórmula para la varianza es:
>
> $$
> \text{Varianza} = \frac{1}{N} \sum_{i=1}^N (y_i - \bar{y})^2
> $$
>
> Donde:
>
> - $y_i$ son los valores de la variable objetivo ($Precio$),
> - $\bar{y}$ es el promedio de los valores en el nodo,
> - $N$ es el número de observaciones.
>
> El promedio de los precios es:
>
> $$
> \bar{y} = \frac{150 + 180 + 210 + 240 + 270}{5} = 210
> $$
>
> La varianza inicial es:
>
> $$
> \text{Varianza} = \frac{1}{5} \left[(150 - 210)^2 + (180 - 210)^2 + (210 - 210)^2 + (240 - 210)^2 + (270 - 210)^2 \right]
> $$
>
> Calculando:
>
> $$
> \text{Varianza} = \frac{1}{5} \left[3600 + 900 + 0 + 900 + 3600 \right] = \frac{9000}{5} = 1800
> $$
>
> ###### Paso 2: Evaluar divisiones posibles
>
> El algoritmo probará diferentes umbrales para dividir los datos según la **Superficie**. Vamos a calcular el efecto de dividir en **Superficie ≤ 70** y **Superficie > 70**.
>
> **División 1: Superficie ≤ 70 y Superficie > 70**
>
> <u>**Nodo 1 (Superficie ≤ 70):**</u>
>
> Datos: {50, 60, 70} 
>
> Precios: {150, 180, 210} 
>
> Promedio:
>
> $$
> \bar{y}_{\text{Nodo 1}} = \frac{150 + 180 + 210}{3} = 180
> $$
>
> Varianza:
>
> $$
> \text{Varianza}_{\text{Nodo 1}} = \frac{1}{3} \left[(150 - 180)^2 + (180 - 180)^2 + (210 - 180)^2 \right]
> $$
>
> $$
> = \frac{1}{3} \left[900 + 0 + 900 \right] = \frac{1800}{3} = 600
> $$
>
> <u>**Nodo 2 (Superficie > 70):**</u> 
>
> Datos: {80, 90} 
>
> Precios: {240, 270} 
>
> Promedio:
>
> $$
> \bar{y}_{\text{Nodo 2}} = \frac{240 + 270}{2} = 255
> $$
>
> Varianza:
>
> $$
> \text{Varianza}_{\text{Nodo 2}} = \frac{1}{2} \left[(240 - 255)^2 + (270 - 255)^2 \right]
> $$
>
> $$
> = \frac{1}{2} \left[225 + 225 \right] = \frac{450}{2} = 225
> $$
>
> **Varianza ponderada después de la división**
>
> La varianza ponderada total tras la división es:
>
> $$
> \text{Varianza ponderada} = \frac{3}{5} \cdot 600 + \frac{2}{5} \cdot 225
> $$
>
> $$
> = 360 + 90 = 450
> $$
>
> ###### Paso 3: Ganancia de homogeneidad
>
> La ganancia de homogeneidad (reducción de varianza) se calcula como:
>
> $$
> \text{Ganancia de homogeneidad} = \text{Varianza inicial} - \text{Varianza ponderada}
> $$
>
> $$
> = 1800 - 450 = 1350
> $$
>
> La división **Superficie ≤ 70 y Superficie > 70** reduce significativamente la varianza, por lo que el algoritmo seleccionará esta división.
>
> ###### Paso 4: Repetir el proceso recursivamente
>
> El árbol continuará dividiendo los datos dentro de cada nodo hijo, buscando nuevas divisiones que reduzcan aún más la variabilidad en los valores de $Precio$.
>
> ###### Representación del árbol resultante (hasta este paso)
>
> <img src=".\assets\image-20241208114655440.png" alt="image-20241208114655440" style="zoom: 50%;" />



> [!important]
>
> El enfoque de los árboles de regresión para maximizar la homogeneidad minimizando la variabilidad asegura que las divisiones sean óptimas en términos de precisión predictiva. Este proceso recursivo permite a los árboles de regresión ajustarse a patrones complejos en los datos, ofreciendo una solución robusta para problemas de predicción de variables continuas.

##### Árboles de clasificación

En los árboles de clasificación, donde la variable objetivo es categórica, el objetivo principal es maximizar la **pureza de las clases** en los nodos resultantes. A través de la división recursiva, el árbol evalúa diversas características y umbrales para dividir los datos, buscando crear grupos donde una sola clase predomine. La calidad de estas divisiones se mide utilizando métricas específicas que evalúan la **homogeneidad de las clases**. En este sentido, las métricas más comunes para evaluar la calidad de las divisiones son la **entropía**, el **índice de Gini** o el **error de clasificación.**

###### **Entropía y ganancia de información**

La **entropía** es una métrica que mide el nivel de desorden o incertidumbre en un conjunto de datos. Un nodo es completamente puro (entropía = 0) si todas las observaciones pertenecen a una sola clase. En cambio, si las clases están distribuidas uniformemente, la entropía es máxima.

La fórmula para la entropía es:

$$
H(S) = -\sum_{i=1}^C p_i \log_2(p_i)
$$

Donde:

- $C$ es el número de clases.
- $p_i$ es la proporción de observaciones pertenecientes a la clase $i$.

Después de realizar una división, la **ganancia de información** mide cuánto se redujo la entropía. La fórmula es:

$$
\text{Ganancia de información} = H(S) - \sum_{j=1}^k \frac{|S_j|}{|S|} H(S_j)
$$

Donde:

- $H(S)$ es la entropía del nodo original (padre).
- $H(S_j)$ es la entropía de cada nodo hijo.
- $\frac{|S_j|}{|S|}$ es el peso del nodo hijo basado en su tamaño relativo.

La ganancia de información selecciona divisiones que maximizan la reducción de incertidumbre, favoreciendo aquellas que producen nodos más homogéneos.

---

> [!tip]
>
> Un vídeo ilustrativo para entender un poco mejor este concepto:
>
> https://www.youtube.com/watch?v=YtebGVx-Fxw

---

###### **Índice de Gini**

El **índice de Gini** mide la probabilidad de clasificar incorrectamente una observación si se elige al azar del nodo. Al igual que la entropía, un valor de Gini cercano a 0 indica alta homogeneidad.

La fórmula del índice de Gini es:

$$
Gini(S) = 1 - \sum_{i=1}^C p_i^2
$$

Donde $p_i$ es la proporción de observaciones en la clase $i$.

A diferencia de la entropía, el índice de Gini penaliza menos las clases dominantes, por lo que a veces genera divisiones ligeramente diferentes.

###### **Error de clasificación**

El error de clasificación mide la proporción de observaciones que no pertenecen a la clase mayoritaria dentro de un nodo. Aunque menos común que la entropía o Gini, es útil como criterio simple para evaluar la pureza de un nodo:

$$
\text{Error de clasificación} = 1 - \max(p_i)
$$

Donde $\max(p_i)$ es la proporción de observaciones en la clase mayoritaria.

###### Ejemplo práctico: Clasificación de clientes

> Supongamos que queremos construir un árbol para predecir si un cliente comprará un producto basado en su **Edad**. Nuestro dataset es:
>
> | Edad | Compra |
> | ---- | ------ |
> | 25   | No     |
> | 30   | No     |
> | 35   | Sí     |
> | 40   | Sí     |
> | 50   | Sí     |
>
> ###### Paso 1: Entropía inicial
>
> Primero calculamos la entropía del nodo raíz. La proporción de clases es:
>
> - "No" = $\frac{2}{5}$
> - "Sí" = $\frac{3}{5}$
>
> La entropía inicial es:
>
> $$
> H(S) = -\left(\frac{2}{5} \log_2 \frac{2}{5} + \frac{3}{5} \log_2 \frac{3}{5}\right)
> $$
>
> $$
> H(S) \approx -\left(0.4 \cdot -1.322 + 0.6 \cdot -0.737\right) = 0.971
> $$
>
> ###### Paso 2: Evaluar divisiones
>
> Consideremos el umbral **Edad ≤ 30** para dividir los datos:
>
> **Nodo 1 (Edad ≤ 30):** {No, No}
>
> $$
> H(S_1) = -\left(\frac{2}{2} \cdot \log_2 \frac{2}{2}\right) = 0 \quad (\text{grupo homogéneo})
> $$
>
> **Nodo 2 (Edad > 30):** {Sí, Sí, Sí}
>
> $$
> H(S_2) = -\left(\frac{3}{3} \cdot \log_2 \frac{3}{3}\right) = 0 \quad (\text{grupo homogéneo})
> $$
>
> ###### Paso 3: Ganancia de información
>
> La ganancia de información se calcula como:
>
> $$
> \text{Ganancia de información} = H(S) - \left(\frac{2}{5} \cdot H(S_1) + \frac{3}{5} \cdot H(S_2)\right)
> $$
>
> $$
> \text{Ganancia de información} = 0.971 - \left(\frac{2}{5} \cdot 0 + \frac{3}{5} \cdot 0\right) = 0.971
> $$
>
> Esta división maximiza la ganancia de información porque produce nodos perfectamente homogéneos.
>
> ###### Representación del árbol resultante
>
> ```mermaid
> graph TD
>     A["Todos los datos (Entropía = 0.971)"] -->|Edad ≤ 30| B["Clase: No (Entropía = 0)"]
>     A -->|Edad > 30| C["Clase: Sí (Entropía = 0)"]
> ```
>



#### Modelos paramétricos vs. modelos no paramétricos

Los árboles de decisión, a diferencia de otros modelos como los de regresión, **no son modelos paramétricos**. Para comprender esta cuestión, es importante analizar qué significa que un modelo sea paramétrico y cómo los árboles de decisión se diferencian de esta categoría.

Los **modelos paramétricos** son aquellos que asumen una estructura funcional fija (como una ecuación matemática) para describir la relación entre las variables de entrada y la salida. Esta estructura está determinada por un número finito y fijo de **parámetros** que se ajustan durante el entrenamiento. Por ejemplo, en una regresión lineal, la relación entre las características y la variable objetivo está definida por una ecuación de la forma:

$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n
$$

Aquí, los parámetros $\beta_i$ se ajustan para minimizar el error en los datos. Una vez ajustados, el modelo utiliza estos mismos parámetros para realizar predicciones, independientemente de la complejidad o el tamaño del dataset.

En contraste, los árboles de decisión **no asumen una forma funcional fija** entre las características y la variable objetivo. En lugar de ajustar un conjunto finito de parámetros, los árboles construyen su estructura de forma adaptativa en función de los datos. Este proceso implica decidir qué características usar, cómo dividirlas y hasta qué profundidad crecer el árbol. La estructura resultante del árbol, incluyendo las reglas de decisión y las particiones del espacio, depende completamente del dataset específico con el que se entrena.

Por esta razón, los árboles de decisión tienen un número de parámetros **que no está fijado a priori**. El tamaño y la complejidad del modelo (es decir, el número de nodos, hojas y divisiones) dependen directamente del dataset y de las configuraciones de hiperparámetros como la profundidad máxima, el número mínimo de muestras por nodo, o los criterios de división.

Al ser no paramétricos, los árboles de decisión tienen varias características distintivas: Primero, su **mayor flexibilidad**, ya que pueden capturar relaciones complejas, no lineales y altamente interactivas entre las características sin necesidad de especificar explícitamente estas relaciones. Esto contrasta con los modelos paramétricos, que suelen necesitar términos adicionales (como polinomios o interacciones) para manejar este tipo de datos. En segundo lugar, la **dependencia del dataset** es muy alta, dependiendo por completo la estructura del árbol de los datos utilizados en el entrenamiento. Si el dataset cambia significativamente (por ejemplo, se añaden nuevas observaciones), es posible que el árbol necesite reconstruirse desde cero. En tercer lugar hay que tener en cuenta que existe una importante **escalabilidad con los datos**, ya que los árboles tienden a ser más efectivos en datasets grandes. Ello es debido a que cuentan con suficiente información para dividir el espacio de características de manera significativa. En datasets pequeños, su flexibilidad puede llevar al sobreajuste. Por último, a diferencia de un modelo paramétrico, el **número de parámetros es variable** y no está predeterminado. En su lugar, depende del número de divisiones realizadas y, por lo tanto, del tamaño del árbol final.

> [!NOte]
>
> **Comparación práctica con modelos paramétricos**
>
> Un modelo paramétrico, como la regresión lineal, funciona bien cuando los datos cumplen los supuestos del modelo (como la linealidad). Sin embargo, si las relaciones son no lineales o hay interacciones complejas, este tipo de modelo puede fallar sin modificaciones significativas. Por el contrario, un árbol de decisión no necesita que el analista especifique relaciones o interacciones de antemano; las descubrirá automáticamente durante el proceso de construcción.
>
> Por ejemplo, en un problema donde las ventas de un producto dependen del precio y la estacionalidad, un modelo paramétrico tendría que incluir términos específicos como polinomios o interacciones (e.g., $precio \times estacionalidad$) para capturar la dinámica del sistema. En cambio, un árbol de decisión crearía divisiones adaptativas basadas en estos factores sin necesidad de transformaciones adicionales.
>

#### Enfoque local vs. enfoque global

Hemos visto en la introducción como los árboles de decisión tienen la capacidad de **dividir el espacio muestral** en regiones más pequeñas y homogéneas. En efecto,  los modelos como la regresión lineal o logística utilizan una única ecuación matemática que abarca todo el espacio de datos. Esta ecuación genera un único conjunto de coeficientes que describe cómo cada predictor afecta la variable objetivo, sin diferenciar entre diferentes regiones del espacio muestral. Si los datos presentan patrones locales o interacciones complejas, estos modelos suelen ser menos efectivos, a menos que se introduzcan manualmente términos adicionales. Esta forma de abordar el problema se denomina en machine learning **enfoque global**.

Por contra, los árboles de decisión trabajan bajo lo que se denomina **enfoque local**. Con ello nos referimos a la capacidad de estos modelos para tomar decisiones basadas en **subconjuntos específicos** del espacio de características, en lugar de aplicar una fórmula o regla global que abarque todo el espacio muestral. Este enfoque permite que los árboles adapten su estructura a las particularidades de los datos en distintas regiones del espacio, lo que los hace flexibles y efectivos en escenarios complejos. Esta flexibilidad permite capturar patrones que serían difíciles de modelar con aproximaciones globales, como las utilizadas en modelos lineales.

El enfoque local de los árboles también elimina la necesidad de asumir **relaciones rígidas entre las variables**. A diferencia de los modelos que dependen de supuestos como la linealidad o la normalidad, los árboles de decisión trabajan directamente con los datos tal como se presentan. Esta propiedad los hace resistentes a datos atípicos o relaciones complejas que escapan a las limitaciones de una fórmula global.

Otra cuestión interesante a tener en cuenta es la **interpretación de los coeficientes**. En modelos globales como la regresión lineal, cada predictor tiene un coeficiente que indica cómo influye en la variable objetivo, pero esta influencia se asume constante en todo el espacio muestral. Si bien los coeficientes pueden ser interpretables matemáticamente, su significado práctico puede ser difícil de entender en situaciones donde los datos son heterogéneos o donde hay interacciones no explícitas entre las variables. Por el contrario, los árboles de decisión ajustan las reglas a regiones específicas del espacio de datos, lo que hace que la interpretación sea más relevante para cada subconjunto. Además, no requieren términos adicionales o transformaciones complejas para capturar interacciones o patrones no lineales.

Por último mencionar que su estructura intuitiva y la claridad de las reglas que generan, provee a los árboles de decisión de una importante **ventaja interpretativa** que los hace accesibles tanto para expertos como para usuarios sin un conocimiento técnico profundo. A diferencia de otros modelos más abstractos (como redes neuronales o incluso regresiones con muchas variables e interacciones), los árboles presentan un razonamiento explícito que puede ser seguido paso a paso. Los árboles de decisión se construyen como diagramas jerárquicos, comenzando desde un **nodo raíz** y dividiéndose en ramas que representan decisiones basadas en los valores de las características del dataset. Cada nodo interno contiene una regla simple y cada hoja proporciona un resultado final. Este diseño permite que cualquier observador pueda rastrear cómo se llega a una predicción específica. Es más, esta estructura en divisiones sucesivas resalta de forma natural las **características más importantes** para la predicción. Al observar las divisiones principales del árbol, es posible identificar qué variables tienen mayor impacto en el modelo, lo que es particularmente útil en escenarios donde el objetivo no es solo predecir, sino también entender qué factores subyacen en un fenómeno.

#### Regularización en árboles de decisión. El equilibrio entre sesgo y varianza.

Los árboles de decisión son modelos altamente flexibles que pueden ajustarse a los datos con gran precisión. Sin embargo, esta misma flexibilidad puede convertirse en un arma de doble filo: si el árbol es demasiado complejo, puede memorizar los datos del conjunto de entrenamiento, capturando patrones espurios o ruido, lo que lleva al **sobreajuste**. Por otro lado, si el árbol es demasiado simple, puede pasar por alto relaciones importantes, lo que resulta en un **subajuste**. Esta tensión entre sobreajuste y subajuste está estrechamente relacionada con el equilibrio entre **sesgo** y **varianza**, y son los procesos de **regularización** los que desempeñan un papel crucial para gestionar este equilibrio.

##### El problema del sobreajuste en árboles de decisión

Un árbol de decisión sin restricciones puede crecer de manera indefinida, dividiendo los datos hasta que cada nodo-hoja contenga una sola observación. Este crecimiento excesivo da lugar a un modelo altamente específico para el conjunto de entrenamiento, pero que **generaliza mal en datos nuevos**. Matemáticamente, el sobreajuste ocurre cuando el modelo minimiza el error de entrenamiento a expensas de un alto error en validación.

Por ejemplo, si se tiene un conjunto de datos ruidoso, un árbol sin restricciones podría capturar el ruido como si fuera un patrón significativo, dividiendo repetidamente el espacio muestral para acomodar anomalías que no representan el comportamiento subyacente real. Esto se traduce en, como se ha dicho, un **un error de entrenamiento muy bajo** (ya que el árbol se adapta perfectamente a los datos vistos) y un **error de generalización alto** (porque el árbol no puede hacer buenas predicciones en datos no vistos).

##### El equilibrio entre sesgo y varianza

El problema del sobreajuste y el subajuste puede analizarse a través del compromiso entre **sesgo** y **varianza**, dos conceptos clave en el aprendizaje automático y que aparecen de forma recurrente en el contexto de los distintos modelos. El **sesgo** mide cuán lejos están las predicciones del modelo de los valores reales. Un modelo con **alto sesgo** tiende a hacer suposiciones excesivamente simplistas sobre los datos, ignorando patrones importantes. En el caso de los árboles de decisión, un árbol demasiado poco profundo tendrá un alto sesgo porque no podrá capturar la complejidad del dataset. Por su parte, la **varianza** mide cuán sensibles son las predicciones del modelo a pequeñas variaciones en los datos de entrenamiento. Un modelo con **alta varianza** se adapta demasiado a los datos, capturando incluso el ruido. En el caso de los árboles, un árbol muy profundo tendrá alta varianza porque su estructura refleja detalles específicos del conjunto de entrenamiento que no se generalizan bien.

Para que un modelo sea eficaz, necesitamos encontrar un equilibrio entre sesgo y varianza. Los árboles de decisión presentan, en este sentido, un **sesgo bajo y varianza alta** cuando son muy profundos (sobreajuste), y un **sesgo alto y varianza baja** cuando son muy poco profundos (subajuste).

Matemáticamente, el error total del modelo se puede descomponer como:

$$
\text{Error total} = \text{Sesgo}^2 + \text{Varianza} + \text{Error irreducible}
$$

Aquí, el error irreducible es el ruido inherente al dataset que no puede ser explicado por ningún modelo.

El objetivo es minimizar tanto el sesgo como la varianza para obtener un modelo que generalice bien. Este equilibrio se logra a través de técnicas de **regularización**.

En la expresión anterior puede comprobarse que el sesgo aparece elevado al cuadrado. Elevar al cuadrado el sesgo en la expresión del error total tiene sentido por varias razones. La primera es que, si no se elevara al cuadrado, los valores positivos y negativos del sesgo se cancelarían entre sí al sumar, lo que podría dar la falsa impresión de que el error sistemático es menor de lo que realmente es. Elevar al cuadrado asegura que tanto los valores positivos como negativos del sesgo contribuyan de manera positiva al error total. La segunda razón tiene que ver con el hecho de que elevar al cuadrado penaliza más las grandes desviaciones del modelo con respecto al valor real. Esto es útil porque errores sistemáticos grandes suelen ser más problemáticos que errores pequeños, y cuadrar el sesgo amplifica este impacto. Por último, el sesgo, al igual que la varianza, contribuye al **error cuadrático medio (MSE)**. Dado que el MSE mide el promedio de los errores al cuadrado, es consistente elevar al cuadrado el sesgo en esta descomposición.

##### Regularización en árboles de decisión

La regularización tiene un impacto directo en el equilibrio entre sesgo y varianza. Incrementar la regularización reduce la varianza al eliminar complejidad innecesaria, si bien es cierto que puede aumentar el sesgo porque simplifica el modelo. Por otra parte **reducir la regularización** permite que el árbol capture patrones más complejos en los datos, reduciendo el sesgo, pero aumenta la varianza porque el modelo puede ajustarse demasiado a los datos de entrenamiento.

Por ejemplo: Un árbol con profundidad limitada puede ignorar algunos patrones importantes, llevando a un sesgo alto, pero un árbol sin restricciones capturará incluso los detalles más pequeños, lo que puede aumentar la varianza y el sobreajuste.

La **clave** está en encontrar el punto óptimo donde el sesgo y la varianza se equilibran, minimizando el error total.

Así pues, la **regularización** implica introducir restricciones al crecimiento del árbol para evitar que se ajuste demasiado a los datos. Esto se logra limitando su complejidad y asegurando que las divisiones sean significativas. Existen varias técnicas para conseguir esto entre las que se pueden destacar la **limitación de la profundidad máxima**, **controlar el número máximo de observaciones por nodo**, las **técnicas de *poda***, el uso de **criterios de parada anticipada** o la **penalización de nodos grandes**

###### **Profundidad máxima del árbol**

Limitar la profundidad del árbol (número máximo de niveles) es una de las formas más efectivas de controlar el sobreajuste. Un árbol muy profundo tendrá nodos extremadamente específicos, adaptándose incluso al ruido de los datos. Limitar la profundidad obliga al árbol a dividir solo cuando la ganancia de homogeneidad es significativa, lo que ayuda a generalizar mejor.

Matemáticamente, esto reduce la varianza al evitar particiones excesivas del espacio muestral.

###### **Número mínimo de observaciones por nodo**

Este parámetro controla cuántas observaciones deben estar presentes en un nodo antes de que se permita una división. Un valor alto impide divisiones en nodos pequeños, reduciendo la probabilidad de sobreajuste. Por ejemplo, si se fija un mínimo de 10 observaciones por nodo, el árbol no podrá crear nodos que representen casos aislados o ruido.

###### **Poda del árbol**

La **poda** implica construir inicialmente un árbol grande y luego eliminar las ramas menos significativas. Esto se hace evaluando el impacto de cada rama en el error de validación y eliminando aquellas que no mejoren el rendimiento. La poda puede basarse en métricas como la reducción de la entropía o del índice de Gini.

Matemáticamente, esto optimiza la estructura del árbol para minimizar el error total (entrenamiento + validación).

###### **Criterio de parada anticipada**

Es otra forma de regularizar un árbol. Lo que se hace en este caso establecer condiciones de parada temprana durante la construcción del árbol, como por ejemplo podrían ser la limitación de un número mínimo de observaciones en los nodos-hojas, o el establecimiento de un umbral mínimo de ganancia de homogeneidad para permitir una división.

###### **Penalización de nodos grandes**

En algunos algoritmos, se introduce una penalización que desincentiva la creación de árboles excesivamente grandes. Esto se formaliza como un término en la función de pérdida del modelo que aumenta con el número de nodos.

##### Ejemplo práctico de regularización

> Imagina que estás construyendo un árbol de decisión para predecir si un estudiante aprobará un examen basado en variables como **horas de estudio**, **participación en clase** y **calificaciones anteriores**.
>
> Si el árbol no tiene restricciones, podría crear reglas extremadamente específicas, como:
>
> - "Si estudia más de 3.5 horas, participa el 85% del tiempo y sus calificaciones previas están entre 7.8 y 8.2, aprobará".
>
> Aunque esta regla puede ajustarse perfectamente al conjunto de entrenamiento, probablemente no se generalizará bien a nuevos estudiantes.
>
> Para evitar esto:
>
> 1. Podrías limitar la profundidad máxima del árbol a 3 niveles, creando reglas más generales como:
>    - "Si estudia más de 3 horas y tiene calificaciones mayores a 7, aprobará".
> 2. Establecer un **mínimo de 10 estudiantes por nodo** impediría que el árbol cree reglas para casos únicos.
>
> Con estas restricciones, el árbol sería menos específico, pero más capaz de generalizar a nuevos datos.



> [!important]
>
> La regularización es esencial para controlar la complejidad de los árboles de decisión y evitar el sobreajuste, mientras que el equilibrio entre sesgo y varianza asegura que el modelo generalice bien a nuevos datos. Entender esta relación permite diseñar árboles que sean efectivos tanto en rendimiento como en interpretación, optimizando su capacidad para capturar patrones significativos sin perder robustez frente a nuevos datos.

### Evaluación del modelo de árbol de decisión

Evaluar un árbol de decisión, como en el caso de otros modelos que ya hemos tratado a lo largo del presente curso,  es un paso crítico para asegurarnos de que el modelo no solo funciona bien en el conjunto de datos de entrenamiento, sino que también puede generalizar correctamente a nuevos datos. Esto implica analizar su rendimiento desde diferentes perspectivas, utilizando métricas apropiadas según el tipo de problema (clasificación o regresión) y técnicas de validación para medir su capacidad de generalización.

Un árbol de decisión bien construido debe equilibrar dos objetivos fundamentales: Por un lado debe tener obtener **resultados precisos en el conjunto de datos de entrenamiento**, asegurando que capture patrones importantes. Por otro, debe **generalizar bien a nuevos datos**, evitando el sobreajuste y garantizando que el modelo sea útil en el mundo real.

El proceso de evaluación permite detectar problemas como el **sobreajuste** (el árbol se adapta demasiado a los datos de entrenamiento) o el **subajuste** (el árbol es demasiado simple y no captura patrones significativos).

#### Evaluación según el tipo de problema

Según se ha detallado a lo largo del capítulo, podemos encontrar dos tipos de problemas a resolver con modelos basados en árboles de decisión: problemas de **clasificación** y problemas de **regresión**.

##### Evaluación en problemas de clasificación

En problemas de clasificación, donde la variable objetivo es categórica, las métricas comunes evalúan la capacidad del árbol para asignar correctamente las clases a las observaciones.  Ya hemos visto que las métricas habituales en problemas de clasificación están basadas en la **matriz de confusión** o en la **curva ROC**. Recordemos brevemente:

###### **Matriz de confusión** 

La matriz de confusión proporciona un desglose detallado de las predicciones frente a los valores reales. En concreto obtenemos cuatro tipos de resultados por cada variable categórica, ya sea positivamente clasificada o negativamente clasificada:

- **Verdaderos positivos (VP)**: Clases reales positivas predichas como positivas
- **Falsos positivos (FP)**: Clases reales negativas predichas incorrectamente como positivas.
- **Verdaderos negativos (VN)**: Clases negativas correctamente predichas como negativas.
- **Falsos negativos (FN)**: Clases reales positivas predichas incorrectamente como negativas.

De esta matriz se derivan métricas adicionales:

**Exactitud (*Accuracy*)**  

La exactitud mide la proporción de observaciones correctamente clasificadas sobre el total de observaciones:

$$
\text{Precisión} = \frac{\text{N° de predicciones correctas}}{\text{Total de observaciones}}
$$

Por ejemplo, si un árbol clasifica correctamente 90 de 100 ejemplos, su precisión sería:

$$
\text{Precisión} = \frac{90}{100} = 0.9 \, (90\%)
$$

###### Precisión y Sensibilidad

Estas métricas son especialmente útiles en problemas desbalanceados (cuando una clase ocurre mucho más que otra):

La **precisión (*precision*)**: Proporción de predicciones correctas para una clase positiva:

$$
\text{Precisión} = \frac{\text{VP}}{\text{VP + FP}}
$$

Mientras la sensibilidad (***Recall***): Proporción de observaciones positivas correctamente identificadas:

$$
\text{Sensibilidad} = \frac{\text{VP}}{\text{VP + FN}}
$$

###### F1-Score  

El F1-Score es la media armónica entre la precisión y el recall, útil cuando se busca un balance entre ambos:

$$
\text{F1-Score} = 2 \cdot \frac{\text{Precisión} \cdot \text{Recall}}{\text{Precisión} + \text{Recall}}
$$

**AUC-ROC (Área bajo la curva ROC)** 
Esta métrica evalúa el rendimiento del modelo al variar el umbral de clasificación, mostrando la capacidad del árbol para distinguir entre clases. Una curva ROC que se acerque al área total de 1 indica un buen rendimiento.

##### Evaluación en problemas de regresión

En problemas de regresión, donde la variable objetivo es continua, el rendimiento se mide evaluando qué tan bien el árbol puede predecir valores numéricos. Las métricas comunes incluyen, como hemos visto en otros modelos de regresión, el **error absoluto medio**, el **error cuadrático medio  (o su raíz)** ,  o el **coeficiente de determinación ($R^2$)** 

###### **Error Absoluto Medio (MAE)**  

Mide el promedio de las diferencias absolutas entre las predicciones ($\hat{y}$) y los valores reales ($y$):

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|
$$

Este error refleja qué tan lejos, en promedio, están las predicciones del modelo de los valores reales.

**Error Cuadrático Medio (MSE)** 
Mide el promedio de los errores elevados al cuadrado:

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2
$$

Penaliza más los errores grandes debido a la elevación al cuadrado, lo que lo hace más sensible a valores atípicos.

###### **Raíz del Error Cuadrático Medio (RMSE)** 

Es la raíz cuadrada del MSE, lo que permite interpretar el error en las mismas unidades que la variable objetivo:

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

###### **Coeficiente de Determinación ($R^2$)**  

Evalúa la proporción de la variación total explicada por el modelo:

$$
R^2 = 1 - \frac{\sum_{i=1}^N (\hat{y}_i - y_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}
$$

Donde $\bar{y}$ es el promedio de los valores reales. Un $R^2$ cercano a 1 indica un buen ajuste, mientras que un $R^2$ cercano a 0 indica un modelo pobre.

#### División del dataset de entrada y validación cruzada

Junto a las métricas de evaluación, es fundamental usar técnicas de división de dataset adecuadas para asegurarse de que el modelo no está sobreajustado al conjunto de entrenamiento. Ya hemos visto algunas técnicas de división comunes entre conjuntos de **entrenamiento y prueba**, de modo que los datos se organicen en dos conjuntos separados, uno para entrenar el modelo y otro para probarlo. Además podríamos, en caso de ser necesario, usar un tercer conjunto para ajustar hiperparámetros antes de evaluar el rendimiento final.

También vimos como la **validación cruzada** es una técnica utilizada para evaluar la capacidad de generalización de un modelo de manera robusta, dividiendo los datos en múltiples subconjuntos para evitar que el rendimiento del modelo dependa de una sola partición. Al probar el modelo en diferentes subconjuntos, se obtiene una evaluación más representativa del rendimiento general en datos no vistos.

Recuerda que uno de los métodos más comunes de validación cruzada es la **validación cruzada k-fold**. En este método, el conjunto de datos se divide en $k$ subconjuntos o *folds* de tamaño similar. El proceso de validación se desarrolla en varias etapas. Primeramente, en cada etapa, uno de los $k$ subconjuntos se usa como conjunto de validación, mientras que los $k-1$ subconjuntos restantes se utilizan como conjunto de entrenamiento. De aquí se obtiene la métrica correspondiente (por ejemplo, un valor de $R^2$). Esto se repite para todos los subconjuntos obteniendo un conjunto de valores de la métrica correspondiente. Por último, al final de las $k$ iteraciones, se promedian las métricas de error obtenidas en cada partición para proporcionar una **estimación del rendimiento del modelo** más robusta y menos sensible a cómo se dividen los datos. Este valor tiene que semejarse al valor de la métrica **en el caso de aplicar el modelo a todo el conjunto de datos de entrenamiento al completo**

La validación cruzada k-fold tiene varias ventajas importantes:

- **Robustez en la evaluación**: Al calcular el rendimiento promedio del modelo en varias divisiones de los datos, se reduce la sensibilidad a cualquier partición específica.
- **Mejor estimación del rendimiento en datos no vistos**: Como el modelo se entrena y evalúa en diferentes subconjuntos de datos, la evaluación resultante es más representativa de su comportamiento general.
- **Identificación de problemas de generalización**: La validación cruzada permite detectar problemas de subajuste o sobreajuste mediante el análisis de la varianza entre las métricas obtenidas en cada iteración.

La validación cruzada k-fold es útil para detectar problemas de subajuste y sobreajuste analizando el **comportamiento de la varianza y el error** entre los diferentes k-folds.

El **sobreajuste** se presentará cuando el modelo muestre una **alta varianza** en las métricas de error entre los distintos folds, es decir, el error variará considerablemente entre las iteraciones. Esto resultado se relaciona con el hecho de que el modelo se ajusta excesivamente a los datos de entrenamiento en ciertos subconjuntos, aprendiendo detalles específicos de algunos folds que no se generalizan bien a otros datos.

Por otro lado, el **subajuste**, se identificará cuando el modelo presente **alto error** en cada uno de los folds, indicando que no es lo suficientemente complejo para capturar los patrones subyacentes en los datos.

Además de k-fold, existen otras variantes de validación cruzada que pueden ser útiles según el contexto:

Por ejemplo, en la **validación cruzada Leave-One-Out (LOOCV)**, el valor de $k$ es igual al número total de observaciones en el conjunto de datos. En cada iteración, se utiliza solo una observación como conjunto de validación, mientras que el resto de las observaciones se emplean para entrenar el modelo. Este enfoque es especialmente exhaustivo y proporciona una evaluación detallada, ya que el modelo se prueba en todas las observaciones de manera individual. Sin embargo, el LOOCV puede resultar computacionalmente costoso en grandes conjuntos de datos, dado que requiere entrenar el modelo tantas veces como observaciones existan en el conjunto, lo que implica una elevada carga de procesamiento.

La **validación cruzada estratificada** es útil en problemas de clasificación o regresión en los que existen grupos o clases en los datos, como distintas categorías o rangos específicos de valores. Este método garantiza que cada fold mantenga una representación proporcional de estos grupos, asegurando que la estructura de la distribución original esté presente en cada subconjunto. La estratificación mejora la representatividad de cada fold y reduce la variabilidad en las métricas de error entre los distintos folds, proporcionando así una evaluación más equilibrada del rendimiento del modelo en contextos con datos distribuidos de forma no uniforme.



> [!important]
>
> La evaluación de un árbol de decisión requiere una combinación de métricas adecuadas y técnicas de validación para medir su rendimiento de manera confiable. En problemas de clasificación, se destacan métricas como la precisión, la matriz de confusión y el AUC-ROC, mientras que en regresión son fundamentales el MAE, MSE y  $R^2$. Validar el modelo asegura que este generalice bien a datos nuevos, garantizando su utilidad en aplicaciones prácticas. Un enfoque riguroso para evaluar y validar un árbol de decisión no solo mejora su rendimiento, sino que también contribuye a construir modelos robustos y explicables.

### Ventajas y limitaciones en el uso de árboles de decisión

Los árboles de decisión son modelos versátiles que destacan por su simplicidad y capacidad para abordar una amplia gama de problemas. Sin embargo, como cualquier herramienta, presentan ventajas y limitaciones que deben ser consideradas al decidir su uso en un proyecto de aprendizaje supervisado.

#### Ventajas

Una de las características más destacadas de los árboles de decisión es su naturaleza **intuitiva y fácil de interpretar**. Cada predicción se basa en un conjunto claro de reglas que pueden ser seguidas desde el nodo raíz hasta una hoja terminal. Esto los convierte en herramientas transparentes, donde los usuarios pueden entender cómo y por qué se tomó una decisión. Esta cualidad es especialmente valiosa en contextos donde la explicabilidad del modelo es crucial, como en medicina, finanzas o sistemas legales.

Otra fortaleza importante es su **versatilidad**. Los árboles pueden aplicarse tanto en problemas de clasificación, donde la variable objetivo es categórica, como en regresión, donde se predicen valores continuos. Esto los hace útiles en una amplia gama de aplicaciones, desde la predicción de ventas hasta el diagnóstico médico.

Además, los árboles de decisión tienen la capacidad de manejar **datos categóricos y numéricos** de forma natural, sin necesidad de preprocesamiento complejo. A diferencia de otros modelos, como las regresiones lineales, no requieren transformar variables categóricas en valores numéricos (codificación). Esto simplifica considerablemente el flujo de trabajo en problemas con datos heterogéneos.

#### Limitaciones

A pesar de sus ventajas, los árboles de decisión también presentan algunas limitaciones importantes. Una de las principales es su **propensión al sobreajuste**. Un árbol sin restricciones puede crecer excesivamente, ajustándose a los patrones específicos del conjunto de entrenamiento, incluyendo el ruido. Este fenómeno hace que el modelo pierda su capacidad de generalización y funcione mal en datos nuevos.

Los árboles también son **sensibles a datasets pequeños**. Cuando el número de observaciones es limitado, las divisiones pueden estar influenciadas por valores atípicos o ruido, produciendo ramas inestables y poco fiables. Esto puede llevar a resultados inconsistentes que no reflejan patrones generalizables en los datos.

En problemas complejos, los árboles de decisión **tienden a ser superados por modelos avanzados**, como los ensambles de árboles (Random Forests, Gradient Boosting) o modelos de aprendizaje profundo. Estos métodos combinan múltiples árboles o aplican técnicas más sofisticadas para mejorar la precisión y reducir el sobreajuste. Aunque un árbol individual puede ser un buen punto de partida, es posible que no sea suficiente en escenarios donde se requiere un rendimiento óptimo.

> **Ejemplo:** 
>
> Consideremos un dataset que contiene información sobre clientes de una tienda y su decisión de comprar un producto. Las características incluyen **edad**, **ingreso** y **frecuencia de visitas al sitio web**. Un árbol de decisión podría dividir los datos inicialmente según la edad, creando una rama para "Menores de 30 años" y otra para "30 años o más". Posteriormente, dentro de cada grupo, el modelo podría usar el ingreso para determinar si es probable que el cliente compre el producto. Esta estructura jerárquica permite descomponer un problema complejo en reglas simples y comprensibles.

##### Para reflexionar...

> **¿Por qué los árboles de decisión son propensos al sobreajuste en datasets pequeños?**
> **Clave**: Piensa en cómo un árbol con muchas divisiones puede memorizar patrones específicos del conjunto de entrenamiento, especialmente cuando los datos son escasos o contienen ruido. Reflexiona sobre la importancia de limitar la profundidad del árbol o el tamaño mínimo de los nodos para mejorar la capacidad de generalización.

### Bosques aleatorios: Introducción

Los **Random Forests** (o bosques aleatorios) son una técnica de aprendizaje automático basada en la combinación de múltiples modelos, en el caso que nos ocupa, **árboles de decisión**, para realizar tareas de clasificación y regresión. Este enfoque pertenece a la categoría de los **modelos de ensamblaje (*ensemble*)**, en los que varios modelos trabajan juntos para mejorar el rendimiento del proyecto en general.

El concepto detrás de los Random Forests es sencillo pero poderoso: construir una colección de árboles de decisión que operen de manera independiente y luego combinar sus predicciones. En el caso de clasificación, la predicción final se determina por un sistema de **votación mayoritaria**, mientras que para regresión se utiliza el **promedio** de las predicciones individuales.

#### ¿Por qué usar Random Forests?

En un único árbol de decisión, aunque intuitivo y fácil de interpretar, pueden aparecer dos problemas importantes: El primero, la posibilidad de aparición del **sobreajuste**. Un árbol sin restricciones tiende a adaptarse demasiado a los datos de entrenamiento, lo que limita su capacidad de generalización. El segundo problema tiene que ver con la **inestabilidad**. En efecto, los árboles de decisión son sensibles a cambios pequeños en los datos; una pequeña modificación en el conjunto de entrenamiento puede generar un árbol completamente diferente.

Los Random Forests mitigan estos problemas al construir una **colección de árboles** (el "bosque"), cada uno **entrenado sobre diferentes muestras del dataset**. Este enfoque reduce la varianza del modelo al promediar múltiples predicciones, mejorando su capacidad de generalización.

### Fundamentos de los Bosques aleatorios

#### Modelos de ensamblaje (*ensamble*)

En el ámbito del aprendizaje automático, los **modelos de *ensamble*** constituyen una clase de técnicas que buscan combinar múltiples modelos individuales para crear un modelo global más robusto, preciso y confiable. Este enfoque parte de una idea fundamental: **un conjunto de modelos puede superar en rendimiento a cualquier modelo individual si se combina de manera adecuada.**

El concepto de ensamble se basa en el principio de que **la combinación de decisiones diversas puede cancelar errores individuales**, siempre que los modelos sean suficientemente independientes y variados entre sí. Esta idea no solo es intuitiva, sino que también está respaldada matemáticamente. Por ejemplo, en estadística, la **Ley de los Grandes Números** afirma que, al promediar múltiples estimaciones independientes, se reduce la varianza del resultado combinado.

> [!Tip]
>
> **La ley de los grandes números: Una explicación intuitiva**
>
> La **ley de los grandes números** es un principio fundamental de la probabilidad que explica cómo, a medida que se incrementa el tamaño de una muestra, los resultados promedio de esa muestra se acercan al valor real (esperanza matemática) de la población de la que provienen.
>
> Imagina que estás lanzando una moneda justa, con igual probabilidad de obtener cara o cruz. Si lanzas la moneda unas pocas veces, podrías obtener más caras que cruces (o viceversa), lo que daría un promedio que no refleja la verdadera probabilidad (50%). Sin embargo, si repites el experimento muchas veces, digamos mil o diez mil lanzamientos, la proporción de caras y cruces estará cada vez más cerca de 50%. Este es el efecto de la ley de los grandes números.
>
> En términos simples, esta ley establece que, a medida que el número de observaciones crece, la media de esas observaciones converge al valor esperado de la población. Esto no significa que no haya fluctuaciones en los resultados, pero asegura que estas fluctuaciones se suavizan con una mayor cantidad de repeticiones.
>
> La ley de los grandes números es clave en estadística y machine learning, ya que respalda la idea de que, con suficientes datos, se pueden obtener estimaciones confiables sobre un fenómeno.

Los modelos de ensamble son especialmente útiles para abordar dos problemas comunes en el aprendizaje automático: la **reducción del sobreajuste**, (ya que al suavizar las predicciones de modelos que tienden a ajustarse demasiado a los datos de entrenamiento) y la **mejora de la generalización** (al combinar modelos individuales, se puede crear un modelo global más robusto y capaz de predecir con mayor precisión en datos no vistos).

Así pues, un modelo de ensamble combina las predicciones de varios modelos individuales para obtener una predicción final. Cada modelo individual, también conocido como "aprendiz" o "estimador base", puede ser de un mismo tipo (e.g., múltiples árboles de decisión) o de diferentes tipos (e.g., combinar árboles de decisión con regresión logística o redes neuronales). Dicha predicción final se obtiene mediante algún mecanismo de combinación.

#### Enfoques principales en los modelos de ensamble

Existen dos enfoques principales para construir modelos de ensamble, cada uno con sus características y aplicaciones específicas: **Bagging (Bootstrap Aggregating)** y **Boosting**. Aunque ambos enfoques buscan combinar modelos individuales para mejorar el rendimiento, difieren en su filosofía y método de construcción.

##### Bagging (*Bootstrap Aggregating*)

El enfoque de ***bagging*** busca reducir la varianza de los modelos base al construir varios modelos independientes y promediar sus resultados. En esencia, el *bagging* toma múltiples muestras aleatorias (con reemplazo) del dataset original y entrena un modelo independiente en cada una de estas muestras. Las predicciones de estos modelos se combinan para formar la predicción final.  El hecho de que de hable de **"muestras con reemplazo"** significa que, al generar los subconjuntos de datos a partir del conjunto de entrenamiento original, las observaciones se seleccionan de forma aleatoria y **pueden repetirse** dentro de una misma muestra.

> **Ejemplo de muestreo con reeemplazo:**
>
> Imaginemos que tienes un conjunto de datos con 100 observaciones. Al crear una muestra de Bootstrap con reemplazo:
>
> 1. Seleccionas aleatoriamente una observación del conjunto original.
> 2. Después de seleccionarla, la "devuelves" al conjunto original, lo que significa que esa misma observación puede ser seleccionada nuevamente en el siguiente paso.
> 3. Repites este proceso hasta crear una nueva muestra que tenga el mismo tamaño que el conjunto original (o un tamaño diferente, dependiendo de la configuración).
>
> Como resultado:
>
> - Algunas observaciones pueden aparecer varias veces en una misma muestra.
> - Algunas observaciones pueden no aparecer en absoluto en esa muestra.



Así pues, los pasos clave en un enfoque de *bagging* serían los siguientes:

1. Crear múltiples muestras de datos utilizando **Bootstrap** (muestreo con reemplazo) a partir del conjunto de datos original.
2. Entrenar un modelo base en cada muestra de Bootstrap.
3. Combinar las predicciones de los modelos: En problemas de regresión, utilizando el **promedio** de las predicciones, y en problemas de clasificación, utilizando un sistema de **votación mayoritaria**.

Visto lo anterior, ¿Qué **ventajas** presenta un enfoque de *bagging*?. Estas ventajas serían de dos tipos. Por un lado reduciría la varianza del modelo combinado, lo que lo hace más robusto frente al sobreajuste. Por otro mejoraría la estabilidad del modelo, ya que suaviza las predicciones.

> Los **random forests** son un ejemplo clásico de Bagging. En este caso, se construye un bosque de árboles de decisión independientes, entrenados en diferentes muestras de Bootstrap, y se combinan sus predicciones.

##### *Boosting*

El enfoque de ***boosting*** se centra en mejorar el rendimiento del modelo base al construir modelos de manera **secuencial**, donde cada modelo corrige los errores cometidos por los modelos anteriores. A diferencia de *bagging*, que busca reducir la varianza, el *boosting* se centra en **reducir el sesgo** del modelo combinado.

¿Cuáles serían los **pasos clave en un enfoque típico de boosting?**. Se enumeran a continuación:

1. Entrenar un modelo base en el conjunto de datos original.
2. Evaluar los errores cometidos por el modelo.
3. Entrenar un segundo modelo que se enfoque en los errores del primer modelo.
4. Repetir el proceso secuencialmente, ajustando cada modelo para corregir los errores acumulados.
5. Combinar las predicciones de todos los modelos, usualmente asignando mayor peso a los modelos más recientes.

Al igual que el enfoque de *bagging*, el enfoque de ***boosting*** también tiene sus **ventajas**. Por un lado reduce el sesgo, al permitir capturar patrones más complejos en los datos. Por otro, es un enfoque particularmente útil en problemas donde los modelos individuales (llamados "aprendices débiles") tienen un rendimiento inicial bajo.

> **Gradient Boosting** es un ejemplo popular de Boosting. En este caso, los modelos base (como árboles de decisión) se ajustan iterativamente para minimizar una función de pérdida, como el error cuadrático en regresión o la entropía cruzada en clasificación.

**Comparación entre bagging y boosting**

En la siguiente tabla se enumeran de forma resumida aspectos que permiten comparar ambos enfoques a la hora de construir modelos de ensamblaje.

| Aspecto                      | Bagging                         | Boosting                                   |
| ---------------------------- | ------------------------------- | ------------------------------------------ |
| **Objetivo**                 | Reducir la varianza             | Reducir el sesgo                           |
| **Construcción**             | Modelos independientes          | Modelos secuenciales                       |
| **Muestreo**                 | Bootstrap (con reemplazo)       | Utiliza todo el dataset, pero ajusta pesos |
| **Predicciones**             | Promedio o votación mayoritaria | Combinación ponderada                      |
| **Robustez frente al ruido** | Alta                            | Menor (puede sobreajustarse si hay ruido)  |

#### Otros enfoques de ensamble

Además de Bagging y Boosting, existen otros métodos menos comunes pero también útiles en contextos específicos. Entre ellos podemos citar los dos siguientes:

1. **Stacking (apilamiento):** Combina modelos de diferentes tipos (e.g., árboles de decisión, regresiones, redes neuronales) utilizando un "meta-modelo" que aprende cómo combinar las predicciones de los modelos base. Este enfoque es más complejo, pero puede lograr resultados impresionantes en problemas difíciles.
2. **Voting (votación simple):** En problemas de clasificación, combina modelos base utilizando una votación simple para decidir la clase final. Aunque es más simple que Bagging o Boosting, puede ser efectivo si los modelos son suficientemente diversos.

#### Importancia de la diversidad en los modelos de ensamble

Un punto crítico en los modelos de ensamble es la **diversidad** entre los modelos base. Si todos los modelos son idénticos (es decir, si cometen los mismos errores), la combinación de sus predicciones no generará ninguna mejora. La diversidad puede lograrse mediante:

- **Muestreo aleatorio** (como en Bagging).
- **Aleatorización de características** (como en Random Forests).
- **Diferentes arquitecturas de modelos base** (como en Stacking).

> [!important]
>
> Los modelos de ensamble son una poderosa técnica en machine learning, ya que combinan la fortaleza de múltiples modelos individuales para obtener un rendimiento superior. Los enfoques de Bagging y Boosting abordan diferentes aspectos del aprendizaje (varianza y sesgo, respectivamente), y su elección depende de las características del problema y los datos. Entender cómo funcionan estos métodos y cómo se implementan en algoritmos como Random Forests y Gradient Boosting es esencial para construir soluciones robustas y efectivas en el aprendizaje automático.

### Construcción de un bosque aleatorio
El algoritmo Random Forest (o de bosque aleatorio) sigue los siguientes pasos básicos

#### Muestreo Bootstrap (con reemplazo)

De un conjunto de datos con $n$ observaciones, se generan múltiples subconjuntos aleatorios (bolsas de datos) seleccionando con reemplazo. Recordemos que ello significa **que algunas observaciones pueden aparecer varias veces en una muestra, mientras que otras pueden quedar fuera**.

#### Selección aleatoria de características

Para cada división dentro de un árbol, solo se considera un subconjunto aleatorio de las características disponibles. Esto introduce diversidad en el bosque, ya que cada árbol evaluará diferentes combinaciones de características.

#### Crecimiento de los árboles

Cada árbol se construye hasta su máxima profundidad. Esto permite que los árboles sean muy específicos respecto a las muestras utilizadas para entrenarlos.

#### Evaluación fuera de la bolsa (*OOB*, Out-of-Bag)

Las observaciones que no fueron seleccionadas en una muestra de Bootstrap se utilizan para evaluar el rendimiento del árbol en cuestión. Este enfoque permite estimar el error del modelo sin necesidad de una validación cruzada explícita.

#### Agregación de predicciones

La clave del éxito de los Random Forests es la **combinación de predicciones** de todos los árboles del bosque. Para los casos de regresión, las predicciones de todos los árboles se **promedian**. En el caso de problemas de clasificación la predicción final se determina por votación mayoritaria (la clase más predicha).

**¿Por qué funciona esto?** Matemáticamente, el promedio de múltiples predicciones independientes reduce la varianza del modelo combinado en comparación con un modelo individual, como ya pronostica la **Ley de los Grandes Números**.  Para ilustrarlo, si asumimos que los errores individuales de los árboles son independientes y tienen la misma varianza ($\sigma^2$), la varianza del modelo combinado (es decir, del Random Forest) será

$$
\text{Varianza combinada} = \frac{\sigma^2}{n}
$$

Donde $n$ es el número de árboles en el bosque. Esto implica que, al aumentar el número de árboles, la varianza combinada disminuye, lo que mejora la estabilidad y la capacidad de generalización del modelo.

> [!Note] 
>
> Aunque los errores individuales no son perfectamente independientes (porque los árboles comparten parte de los datos), la diversificación introducida por el muestreo *bootstrap* y la selección aleatoria de características minimiza esta correlación.

### Evaluación de los Random Forest

Evaluar un **Random Forest** implica considerar tanto las fortalezas de su naturaleza como modelo de ensamble como las peculiaridades de su construcción basada en árboles individuales. A diferencia de un árbol de decisión aislado, donde la evaluación se centra en medir cómo un único modelo generaliza a datos nuevos, en los Random Forests el objetivo es evaluar el desempeño del bosque completo, que surge de la combinación de múltiples árboles.

Pero ¿qué cambia en la evaluación de los bosques aleatorios con respecto a la de los árboles individuales? En un árbol individual, las métricas de evaluación comunes como la exactitud, el error cuadrático medio (MSE) o el $R^2$ se calculan directamente al comparar las predicciones del modelo con los valores reales, ya sea en un conjunto de prueba o utilizando validación cruzada. Sin embargo, los Random Forests tienen una ventaja distintiva: el uso de observaciones **Out-of-Bag (OOB)** para evaluar su rendimiento de manera interna, eliminando la necesidad de una validación cruzada explícita.

#### **El rol del muestreo OOB**

Al construir cada árbol en un Random Forest, el algoritmo utiliza una muestra de Bootstrap con reemplazo para entrenarlo, lo que significa que alrededor del 37% de las observaciones originales quedan fuera de esa muestra. Estas observaciones, denominadas **Out-of-Bag (OOB)**, no participan en el entrenamiento del árbol en cuestión, pero se pueden usar para evaluar su desempeño.

El error OOB se calcula al promediar las predicciones hechas para las observaciones que quedaron fuera de la bolsa. En lugar de reservar un conjunto separado para validación, el modelo utiliza el error OOB como una estimación directa de su error de generalización. Esta característica de los Random Forests hace que sean particularmente eficientes, ya que la evaluación ocurre de manera inherente durante el proceso de entrenamiento.

> [!Tip]
>
> **¿Por qué se quedan fuera el 37% de las observaciones?**
>
> El hecho de que aproximadamente el **37% de las observaciones** queden fuera de las muestras de entrenamiento en el muestreo **Out-of-Bag (OOB)** puede explicarse matemáticamente considerando cómo funciona el proceso de **muestreo con reemplazo** utilizado en los **Random Forests**. Como se ha visto, en los Random Forests, cada árbol se entrena en una muestra de Bootstrap generada a partir del conjunto de datos original. Esta muestra de Bootstrap se crea seleccionando aleatoriamente observaciones del dataset original, con reemplazo, hasta formar una nueva muestra con el mismo tamaño que el dataset original. Dado que el muestreo es **con reemplazo**, cada observación tiene la posibilidad de ser seleccionada múltiples veces o incluso no ser seleccionada en absoluto.
>
> Supongamos que el conjunto de datos tiene $n$ observaciones. Para una observación específica, la probabilidad de **ser seleccionada** en un solo muestreo es:
>
> $$
> P(\text{seleccionada}) = \frac{1}{n}
> $$
> 
> Por lo tanto, la probabilidad de **no ser seleccionada** en un único muestreo es:
>
> $$
> P(\text{no seleccionada}) = 1 - \frac{1}{n}
> $$
> 
> En un muestreo de Bootstrap, seleccionamos $n$observaciones con reemplazo. Esto significa que hacemos $n$selecciones independientes. La probabilidad de que una observación no sea seleccionada en las $n$extracciones es el producto de $P(\text{no seleccionada})$ repetido $n$ veces, lo que da:
>
> $$
> P(\text{no seleccionada en \( n \) extracciones}) = \left(1 - \frac{1}{n}\right)^n
> $$
> 
> A medida que el tamaño del conjunto de datos ($n$) crece, la expresión $\left(1 - \frac{1}{n}\right)^n$se aproxima al valor de la constante matemática $e^{-1}$ (aproximadamente 0.3679). Esto ocurre porque:
>
> $$
> \lim_{n \to \infty} \left(1 - \frac{1}{n}\right)^n = e^{-1}
> $$
> 
> Por lo tanto, la probabilidad de que una observación quede **fuera de la muestra de Bootstrap** es aproximadamente **37%** cuando $n$ es grande.
>

#### **Combinación de predicciones: Más allá de los árboles individuales**

##### Para clasificación

Cada árbol contribuye con un voto para predecir la clase de una observación. La clase final es aquella que recibe la mayoría de los votos. Esta combinación reduce la probabilidad de errores, ya que las predicciones incorrectas de algunos árboles tienden a cancelarse si la mayoría de los árboles son correctos.

Al evaluar el modelo, se utilizan métricas como:
- **Exactitud (accuracy):** La proporción de observaciones correctamente clasificadas.
- **F1-score:** Especialmente útil en problemas desbalanceados.
- **AUC-ROC:** Evalúa la capacidad del modelo para distinguir entre clases en diferentes umbrales.

##### Para regresión

Cada árbol produce un valor numérico, y la predicción final es el promedio de estos valores. Este promediado reduce la varianza del modelo y lo hace más robusto frente a datos ruidosos.

Las métricas comunes en regresión incluyen:
- **Error Absoluto Medio (MAE):** Evalúa la magnitud promedio de los errores.
- **Error Cuadrático Medio (MSE):** Penaliza más los errores grandes.
- **Coeficiente de Determinación ($R^2 $):** Indica qué tan bien se ajusta el modelo a los datos.

#### Ventajas del enfoque de evaluación de Random Forest

##### Reducción de la varianza

En los árboles individuales, la evaluación puede estar influenciada por el ruido en los datos, ya que un único modelo captura tanto patrones generales como anomalías específicas. En los Random Forests, al combinar múltiples árboles, las predicciones se suavizan, reduciendo el impacto del ruido y proporcionando métricas más estables.

##### Error OOB como estimación directa

La evaluación interna mediante el error OOB elimina la necesidad de dividir los datos en conjuntos de entrenamiento y prueba o de realizar validación cruzada. Esto no solo ahorra tiempo, sino que también permite aprovechar completamente el conjunto de datos para el entrenamiento.

##### Importancia de las características

Más allá de las métricas tradicionales, los Random Forests ofrecen información valiosa sobre qué características son más relevantes para las predicciones. Esto se logra evaluando cuánto mejora la precisión o reduce el error al utilizar cada característica en los árboles del bosque.

#### Comparación con la evaluación de árboles individuales

Aunque los fundamentos de evaluación (como exactitud, $\text{MSE}$ o $R^2$) son comunes entre árboles individuales y Random Forests, las diferencias clave radican en cómo se obtiene la predicción y cómo se manejan los datos:

- **En árboles individuales:** La evaluación depende completamente del rendimiento de un único modelo, lo que lo hace más susceptible al ruido y a la elección de las divisiones específicas en los datos.
- **En Random Forests:** La evaluación se realiza en un modelo colectivo, donde el efecto de errores individuales se diluye gracias al promedio o la votación mayoritaria, lo que lleva a un desempeño más consistente.

Por ejemplo, un árbol profundo podría sobreajustarse al conjunto de entrenamiento y mostrar un error bajo en este, pero un error alto en el conjunto de prueba. En cambio, el error promedio de los árboles en un Random Forest, combinado con el uso del error OOB, proporciona una estimación más realista de cómo el modelo generaliza a nuevos datos.

> [!important]
>
> La evaluación de un Random Forest no solo hereda las métricas utilizadas para los árboles individuales, sino que también aprovecha su naturaleza de ensamble para ofrecer resultados más robustos. Al combinar la eficiencia del error OOB con métricas tradicionales como precisión, MSE o F1-score, los Random Forests permiten evaluar su rendimiento de manera precisa y sin necesidad de técnicas adicionales como la validación cruzada. Esto los convierte en una herramienta no solo poderosa, sino también práctica para una amplia gama de problemas en machine learning.

### Ventajas e inconvenientes en el uso de los Bosques Aleatorios

Los bosques aleatorios ofrecen una robustez destacable frente al sobreajuste, un problema común en modelos más simples como los árboles de decisión individuales. Esto se debe a su capacidad para promediar las predicciones de múltiples árboles, reduciendo la varianza general del modelo. Así, evitan que un solo árbol domine el resultado final, haciendo del conjunto un modelo más equilibrado y confiable. Además, su naturaleza de *ensamble* los hace especialmente resistentes a datos ruidosos y a conjuntos desequilibrados, ya que pueden ajustar los pesos de las clases para compensar desbalances y capturar patrones importantes en escenarios complejos.

Otra ventaja importante de los bosques aleatorios es su notable flexibilidad. Pueden emplearse tanto en problemas de clasificación como en regresión, sin requerir un preprocesamiento exhaustivo de los datos. Por ejemplo, no necesitan que las variables sean normalizadas ni que las categóricas sean codificadas previamente. Esta versatilidad los convierte en una herramienta poderosa para una amplia variedad de aplicaciones. Además, los Random Forests ofrecen información valiosa sobre la importancia de las características, calculando métricas que permiten identificar qué variables son más relevantes para las predicciones. Esto los hace útiles no solo para obtener predicciones, sino también para entender qué factores están impulsando los resultados.

Un aspecto práctico es que no requieren validación cruzada para evaluar su rendimiento. Gracias al uso de observaciones fuera de la bolsa (OOB), el modelo puede estimar directamente su error de generalización durante el entrenamiento, eliminando la necesidad de técnicas adicionales de validación. Esto ahorra tiempo y simplifica el flujo de trabajo.

Sin embargo, los bosques aleatorios también presentan algunas limitaciones. Uno de los principales inconvenientes es su costo computacional. Entrenar y combinar múltiples árboles puede ser muy exigente en términos de recursos, especialmente cuando se trabaja con conjuntos de datos grandes o se decide construir un bosque con un número elevado de árboles. Este costo puede limitar su uso en escenarios donde la eficiencia es clave.

Por otro lado, aunque los árboles individuales son intuitivos y fáciles de interpretar, un bosque aleatorio, al ser una combinación de muchos modelos, pierde esta ventaja. Comprender cómo interactúan cientos o miles de árboles en un bosque puede ser complicado, lo que reduce su interpretabilidad general. Finalmente, aunque los bosques aleatorios son modelos muy sólidos, en algunos problemas específicos pueden no ser la mejor opción. En escenarios donde las relaciones entre características son particularmente complejas, otros modelos de ensamble, como el Gradient Boosting, a menudo logran un rendimiento superior.

En conclusión, los bosques aleatorios son una herramienta poderosa, flexible y robusta, ideal para una amplia gama de problemas. Sin embargo, deben ser utilizados considerando sus limitaciones, especialmente en términos de interpretabilidad y costo computacional.
