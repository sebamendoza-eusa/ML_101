
# Tema 2. Sistemas de aprendizaje automático supervisado

## Máquinas de Vectores de Soporte (SVM)

### Objetivos del módulo

> 1. Entender la teoría detrás de las SVM: márgenes, hiperplanos y separación de clases.
> 2. Comprender el kernel trick: cómo y por qué usar kernels para problemas no lineales.
> 3. Familiarizarse con la implementación básica de SVM en Python: uso de `SVC` y visualización.
> 4. Distinguir entre clasificación y regresión con SVM: diferencias y aplicaciones prácticas.
> 5. Ajustar hiperparámetros: rol de `C`, `gamma` y selección de kernels usando `GridSearchCV`.
> 6. Manejar datos desbalanceados: estrategias como `class_weight='balanced'` y métricas adecuadas.
> 7. Aplicar SVM a casos prácticos avanzados: datasets reales, validación cruzada y análisis.
> 8. Reconocer limitaciones y alternativas: escalabilidad, ventajas y desventajas frente a otros modelos.

---

#### Referencias académicas

- Cortes, Corinna, and Vladimir Vapnik. 1995. “Support-Vector Networks.” *Machine Learning* 20 (3): 273–97. https://doi.org/10.1023/A:1022627411411.

- Drucker, Harris, Chris J. C. Burges, Linda Kaufman, Alex Smola, and Vladimir Vapnik. 1997. “Support Vector Regression Machines.” In *ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS 9*, 155–61. MIT Press.

- https://statisticalsupportandresearch.wordpress.com/wp-content/uploads/2017/05/vladimir-vapnik-the-nature-of-statistical-learning-springer-2010.pdf

#### Otras referencias de ayuda

- [How SVM (Support Vector Machine) algorithm works](https://www.youtube.com/watch?v=1NxnPkZM9bc)



### Introducción teórica a las SVM

Las Máquinas de Vectores de Soporte, o SVM por sus siglas en inglés (Support Vector Machines), son una de las herramientas más robustas y ampliamente utilizadas en Machine Learning, aplicables tanto a  problemas de clasificación y regresión. Su principal objetivo es encontrar **una frontera de decisión óptima entre clases** que **maximice la distancia**, o margen, entre los datos más cercanos de cada clase. Esta distancia es crucial, ya que un margen amplio no solo ayuda a separar las clases con mayor confianza, sino que también mejora la capacidad del modelo para generalizar a nuevos datos.

El concepto detrás de las SVM puede parecer complejo al principio, pero la idea clave radica en algo muy intuitivo: trazar una línea o un plano (dependiendo de la dimensionalidad) que divida de forma clara y efectiva los datos pertenecientes a diferentes clases. Imagina que tienes un conjunto de puntos en un plano bidimensional, donde cada punto pertenece a una de dos categorías, por ejemplo, triángulos y círculos. El objetivo de las SVM es encontrar la línea que no solo separa los triángulos de los círculos, sino que lo hace con el mayor margen posible, minimizando el riesgo de errores.

<img src=".\assets\1Fjj7EblDs2J88GgJmyKL8w.png" alt="Support Vector Machine(with Numerical Example)" />

#### Márgenes y fronteras de decisión

La noción de **margen** es central para comprender cómo operan las SVM. Este margen se refiere a la distancia más pequeña entre la frontera de decisión y los puntos más cercanos de cualquier clase, conocidos como **vectores de soporte**. Un margen más amplio implica un modelo más robusto, ya que una frontera que se encuentra lejos de los puntos más cercanos es menos susceptible al ruido o a pequeños cambios en los datos. La frontera óptima, entonces, no es cualquier línea de separación, sino aquella que **maximiza este margen**.

Matemáticamente, si representamos la frontera de decisión como un hiperplano definido por la ecuación:

$$
w^T x + b = 0
$$

donde $w$ es el vector que define la orientación del hiperplano, $x$ es un punto de datos y $b$ es el sesgo que ajusta la posición de este hiperplano, puede demostrarse que el margen se calcula como:

$$
M = \frac{2}{\|w\|}
$$

Así, el problema de maximización del margen puede reconvertirse a un problema de minimización de la norma de $w$ (concretamente de la norma al cuadrado), lo que estaría directamente relacionado con la optimización de las SVM. Este problema de minimización es más manejable que el problema de maximización debido a que se convierte en un problema de **optimización convexo**. La convexidad asegura un único mínimo global, lo que permite una solución del problema eficiente y estable.

$$
\min_{w, b} \frac{1}{2} \|w\|^2
$$


##### Caso práctico: Márgenes y fronteras de decisión en dos dimensiones

Para comprender mejor cómo funcionan los márgenes y las fronteras de decisión en una SVM, imaginemos un caso sencillo de **un problema de clasificación** en el que los datos se encuentran en un espacio bidimensional. En este caso, el **hiperplano** que separa las clases no es un plano, sino una simple **recta** en el plano cartesiano.  

Supongamos que tenemos dos clases de datos representadas por puntos en el plano: círculos azules ($+1$) y triángulos rojos ($-1$). Nuestro objetivo es encontrar una línea que separe ambos tipos de puntos con el **mayor margen posible**. Esa línea no puede ser cualquiera: debe estar estratégicamente ubicada para maximizar la distancia entre los puntos más cercanos de ambas clases. Esta distancia define el **margen**.

En dos dimensiones, una recta puede representarse mediante la ecuación: 

$$
w_1 x_1 + w_2 x_2 + b = 0
$$

donde: 

- $x_1$ y $x_2$ son las coordenadas de un punto en el plano,  
- $w_1$ y $w_2$ son los pesos que definen la orientación de la recta, y  
- $b$ es el sesgo que ajusta su posición.

Esta ecuación divide el plano en dos regiones: 

- Los puntos donde $w_1 x_1 + w_2 x_2 + b > 0$ pertenecen a una clase ($+1$).  
- Los puntos donde $w_1 x_1 + w_2 x_2 + b < 0$ pertenecen a la otra clase ($-1$).  

La recta misma, definida por la igualdad $w_1 x_1 + w_2 x_2 + b = 0$, actúa como la **frontera de decisión**.

El **margen** es la distancia perpendicular entre la frontera de decisión y los puntos más cercanos de ambas clases, que llamamos **vectores de soporte**. Para las SVM, el margen no es un concepto abstracto; es el objetivo principal: maximizarlo es lo que permite que el modelo sea más robusto frente a nuevos datos.

En este caso bidimensional, la distancia de un punto $x = (x_1, x_2)$ a la recta puede calcularse como: 

$$
\text{Distancia} = \frac{|w_1 x_1 + w_2 x_2 + b|}{\sqrt{w_1^2 + w_2^2}}
$$

Para los puntos que definen los márgenes (vectores de soporte), esta distancia se fija en 1. Esto significa que las rectas paralelas al hiperplano óptimo, que se encuentran a una distancia $+1$ y $-1$, definen el margen. Matemáticamente, estas rectas se representan como:

$$
w_1 x_1 + w_2 x_2 + b = 1 \quad \text{(margen positivo)}
$$

$$
w_1 x_1 + w_2 x_2 + b = -1 \quad \text{(margen negativo)}
$$

La distancia entre estas dos rectas es lo que la SVM maximiza. Representemos el escenario anterior en un gráfico bidimensional: 

<img src=".\assets\image-20250112094838426.png" alt="image-20250112094838426" />

- Los círculos azules ($+1$) están en un lado del plano.  
- Los triángulos rojos ($-1$) están en el otro lado.  
- Entre ambas clases hay una recta, la frontera de decisión, que separa los círculos de los triángulos. 

Además, hay dos rectas adicionales paralelas a la frontera de decisión que marcan los márgenes. Los puntos de cada clase que tocan estas rectas paralelas son los vectores de soporte. Estos puntos son los más críticos para determinar la posición de la recta de separación. Los puntos más alejados no afectan en absoluto la posición del hiperplano.

> **Ejemplo**: Piensa en un caso donde estás separando dos tipos de flores basándote en su altura ($x_1$) y el ancho de sus pétalos ($x_2$). La frontera de decisión será una línea que divida las flores de ambos tipos, mientras que las flores más cercanas a esta línea (los vectores de soporte) serán las que determinen su posición exacta.



> [!Warning]
>
> Maximizar el margen tiene un propósito claro: garantizar que el modelo sea más robusto frente a nuevos datos. Si la recta de separación está demasiado cerca de un grupo de puntos, pequeños cambios en los datos de entrenamiento podrían cambiar drásticamente su posición, resultando en un modelo poco fiable. En cambio, un margen amplio asegura que incluso con ligeras variaciones, la recta siga separando las clases de manera efectiva.
>

##### Para reflexionar...

> **¿Qué sucedería si algunos puntos no pudieran ser separados por una recta en el plano bidimensional?** 
> **Clave**: Considera cómo las SVM manejan este problema al introducir el concepto de márgenes blandos y términos de relajación $\xi_i$, lo que permite ciertos errores de clasificación para encontrar un equilibrio entre margen amplio y precisión.

#### Márgenes blandos (*soft-margins*) y regularización

En cualquier caso, la construcción de una SVM implica formular un problema de optimización cuyo objetivo es encontrar el hiperplano que maximice el margen. Para que esto sea posible, los datos deben cumplir ciertas condiciones de separabilidad, formalizadas como: 

$$
y_i (w^T x_i + b) \geq 1 \quad \forall i
$$

donde $y_i$ representa la etiqueta de la clase ($+1$ o $-1$) y $x_i$ es el punto de datos correspondiente. Esta desigualdad asegura que cada punto esté correctamente clasificado y se encuentre al menos a una distancia de 1 del hiperplano. Sin embargo, en problemas del mundo real, los datos no siempre son perfectamente separables, lo que da lugar al concepto de márgenes blandos.

En la práctica, es común encontrarse con conjuntos de datos donde las clases no pueden separarse perfectamente con una línea o un plano. Para manejar este escenario, las SVM introducen el concepto de **márgenes blandos**, que permiten ciertos errores de clasificación al incorporar un término de relajación $\xi_i$. Este término ajusta la cantidad de puntos que pueden violar las restricciones de clasificación: 

$$
y_i (w^T x_i + b) \geq 1 - \xi_i \quad \text{con } \xi_i \geq 0
$$

El modelo resultante equilibra dos objetivos: maximizar el margen y minimizar los errores de clasificación. Esto se logra mediante un parámetro de regularización, $C$, que controla el equilibrio entre ambos objetivos. Un valor alto de $C$ da prioridad a minimizar los errores, lo que puede llevar al sobreajuste, mientras que un valor bajo favorece márgenes más amplios y una mejor generalización.

La función objetivo para este escenario queda formulada como: 

$$
\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i
$$

Aquí, la primera parte $\frac{1}{2} \|w\|^2$ busca maximizar el margen, mientras que **la segunda parte penaliza los errores de clasificación.**

#### Intuición geométrica de los vectores de soporte

Los vectores de soporte son aquellas observaciones de nuestro conjunto de entrenamiento que definen el margen. Estos puntos son cruciales porque determinan la posición y la orientación del hiperplano. Los puntos que no son vectores de soporte, es decir, aquellos que están más alejados de la frontera de decisión, **no tienen impacto en su cálculo**. Esto hace que las SVM sean especialmente útiles cuando se trabaja con conjuntos de datos grandes, ya que el modelo se basa únicamente en un subconjunto representativo de puntos.

> **Ejemplo**: Supongamos que estás desarrollando un modelo para clasificar correos electrónicos como spam o no spam. Si solo unos pocos correos están cerca del límite entre ambas categorías, estos serán los vectores de soporte. Los correos que están claramente etiquetados (por ejemplo, un mensaje publicitario obvio o un mensaje personal claro) no afectan directamente la frontera de decisión.

> [!warning]
>
> Las SVM ofrecen un enfoque matemáticamente sólido y robusto para problemas de clasificación y regresión. Su capacidad para encontrar un margen óptimo, combinada con su flexibilidad para manejar datos no separables mediante márgenes blandos, las convierte en una herramienta esencial en el repertorio de Machine Learning. Sin embargo, comprender los conceptos de márgenes, hiperplanos y vectores de soporte es fundamental antes de avanzar a temas más avanzados, como el kernel trick.

##### Para reflexionar...

> **¿Por qué crees que las SVM maximizan el margen en lugar de simplemente minimizar los errores de clasificación en los datos de entrenamiento?** 
> **Clave**: Piensa en cómo un margen amplio puede mejorar la capacidad del modelo para generalizar a nuevos datos, evitando el sobreajuste causado por centrarse exclusivamente en los datos de entrenamiento.

### El *truco del Kernel* (Kernel trick): SVM para problemas no lineales

Una de las limitaciones iniciales de las SVM es que, en su forma más básica, están diseñadas para trabajar con datos que son **linealmente separables**. Es decir, los puntos de datos deben poder dividirse claramente por una línea (en dos dimensiones), un plano (en tres dimensiones), o un hiperplano (en dimensiones superiores). Sin embargo, en muchos problemas del mundo real, los datos no son separables por una línea recta o un plano en el espacio original de las características. Aquí es donde entra en juego el concepto de **kernel trick**.

<img src=".\assets\1J0k7TxTLoL5ZG-Hq6v34Jg.png" alt="Non-linear Support Vector Machines Explained" />

Pero antes de comenzar a trabajar con el *kernel trick* introduzcamos el concepto de ***kernel*** en el contexto de las máquinas de soporte vectorial.

En términos simples, un **kernel** es una función matemática que calcula la **similitud** entre dos puntos en un espacio de características. En las SVM, el kernel juega un papel fundamental, ya que permite medir estas similitudes de una manera que puede ser mucho más compleja que un simple cálculo en el espacio original de los datos. 

Cuando se utiliza un kernel, el objetivo final es proyectar los datos de su **espacio original de características** a un **espacio de mayor dimensión** (también llamado **espacio transformado** o **espacio del kernel**). En este nuevo espacio, se espera que los datos sean **linealmente separables**, incluso si no lo son en el espacio original.

En las SVM, el kernel se utiliza en la formulación matemática del modelo. Recordemos que las SVM intentan encontrar un hiperplano que maximice el margen entre las clases. Esto se logra resolviendo un problema de optimización que depende de los **productos escalares** entre los vectores de características $x_i$ y $x_j$.

Para problemas no lineales, en lugar de calcular directamente los productos escalares en el espacio original, el kernel calcula estos productos escalares en el espacio transformado. Matemáticamente, esto se escribe como:

$$
K(x_i, x_j) = \phi(x_i)^T \phi(x_j)
$$

Donde:

- $K(x_i, x_j)$ es el valor del kernel (es decir, la similitud entre $x_i$ y $x_j$).
- $\phi(x_i)$ y $\phi(x_j)$ son las versiones transformadas de $x_i$ y $x_j$ en el espacio de mayor dimensión.
- $\phi(x_i)^T \phi(x_j)$ es el producto escalar entre los vectores transformados.

La clave del kernel trick es que **no se necesita calcular explícitamente $\phi(x_i)$ y $\phi(x_j) $**. En lugar de eso, se evalúa directamente el kernel $K(x_i, x_j)$, que encapsula toda la complejidad de la transformación en una fórmula compacta.

##### Propiedades matemáticas de los kernels

Para que una función sea válida como kernel en las SVM, debe cumplir ciertas propiedades matemáticas. La más importante es que para garantizar que el problema de optimización de las SVM sea convexo y tenga solución, el kernel debe ser una **función de núcleo positivo semidefinido**. Esto significa que el kernel debe calcular similitudes de una manera que sea consistente con la geometría del espacio transformado. Otras propiedades importantes de los kernels son su simetría, es decir, $K(x_i, x_j) = K(x_j, x_i)$, y por supuesto, que son capaces de  evaluar directamente la relación entre puntos sin necesidad de construir el espacio transformado explícitamente.

> **Ejemplo intuitivo: El kernel en acción**
>
> Supongamos que tenemos un conjunto de datos con las siguientes coordenadas en un espacio bidimensional:
>
> Clase $+1 $:  
> - $x_1 = (1, 1) $
> - $x_2 = (2, 2) $
>
> Clase $-1 $:  
> - $x_3 = (-1, -1) $
> - $x_4 = (-2, -2) $
>
> En este caso, no es posible trazar una línea que separe las dos clases en el plano. Sin embargo, al aplicar un kernel (como un kernel polinómico o un kernel RBF), los puntos se transforman automáticamente a un espacio donde se puede encontrar un hiperplano que divida las clases.
>
> Con un **kernel polinómico** de grado 2, por ejemplo, la transformación matemática podría mapear los puntos $(x_1, x_2)$ a una nueva dimensión basada en sus interacciones cuadráticas, como $(x_1^2, x_2^2, x_1x_2)$. En este espacio, las clases serían linealmente separables

#### Concepto de *kernel trick*

Ahora sí, el **kernel trick** sería una **técnica matemática** que permite a las SVM trabajar con problemas no lineales proyectando los datos originales en un **espacio de mayor dimensión**, donde se vuelven separables linealmente. Para transformar los datos usaríamos una función kernel, sin necesidad de calcular explícitamente las coordenadas de los datos transformados. Esto reduce el costo computacional y hace que las SVM sean prácticas incluso para transformaciones complejas.

Imaginemos un problema de clasificación donde los datos no pueden separarse en dos clases usando una línea recta en dos dimensiones. Por ejemplo, un conjunto de datos donde los puntos de una clase están en forma de un círculo rodeado por los puntos de la otra clase. En este caso, no existe una línea recta que pueda dividir las dos clases en el espacio bidimensional.

El kernel trick aplica una transformación matemática que "eleva" los datos a un espacio de mayor dimensión. En este nuevo espacio, los datos pueden separarse linealmente. Por ejemplo:
- En el espacio bidimensional original, los puntos no son separables.
- Con un kernel, los datos se proyectan a un espacio tridimensional donde un plano puede dividir las dos clases.

La magia del kernel trick está en que no necesitamos calcular las coordenadas de los puntos en el espacio de mayor dimensión. En lugar de eso, el kernel calcula directamente los **productos escalares** entre los puntos transformados, lo que es mucho más eficiente computacionalmente.

<img src=".\assets\1mCwnu5kXot6buL7jeIafqQ.png" alt="What is the kernel trick? Why is it important? | by Grace Zhang | Medium" />

#### Descripción de kernels comunes

La tipología de las funciones kernel tienen que ver por tanto con su forma funcional. Veamos algunos ejemplos:

##### Kernel lineal
Este es el kernel más sencillo y no aplica ninguna transformación. Es útil cuando los datos ya son linealmente separables en el espacio original de características. La fórmula del kernel lineal es simplemente el producto escalar entre dos puntos:

$$
K(x_i, x_j) = x_i^T x_j
$$

> **Ejemplo**: Clasificación de datos que ya tienen una separación clara, como el peso y la altura de personas donde una línea recta puede dividir las clases.

##### Kernel polinómico
Este kernel proyecta los datos a un espacio de mayor dimensión utilizando términos polinómicos. Su fórmula es:

$$
K(x_i, x_j) = (x_i^T x_j + c)^d
$$

Donde:
- $d$ es el grado del polinomio (e.g., 2, 3, etc.).
- $c$ es una constante que controla la influencia de los términos de menor grado.

Este kernel es útil cuando la relación entre las características de los datos es no lineal pero puede representarse como una combinación de términos polinómicos.

> **Ejemplo**: Clasificación de datos donde las clases están separadas por una curva parabólica en lugar de una línea recta.

##### Kernel RBF (Radial Basis Function)
El kernel RBF, también conocido como kernel gaussiano, es uno de los más utilizados. Proyecta los datos a un espacio de dimensión infinita mediante una transformación basada en la distancia entre los puntos. Su fórmula es:

$$
K(x_i, x_j) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)
$$

Donde:

- $\|x_i - x_j\|^2$ es la distancia euclidiana al cuadrado entre los puntos $x_i$ y $x_j $.
- $\sigma$ controla el alcance de la influencia de cada punto.

El kernel RBF es especialmente útil en problemas donde las clases tienen fronteras de decisión muy complejas y no lineales.

> **Ejemplo**: Clasificación de imágenes donde las clases están separadas por características complejas y altamente no lineales.



##### Kernel sigmoide
El kernel sigmoide utiliza la función sigmoide como base para la transformación de los datos. Su fórmula es:

$$
K(x_i, x_j) = \tanh(\alpha x_i^T x_j + c)
$$

Donde:
- $\alpha$ y $c$ son parámetros ajustables.

Este kernel tiene similitudes con una neurona en una red neuronal, por lo que a menudo se utiliza en problemas relacionados con redes neuronales.

> **Ejemplo**: Problemas de clasificación binaria en conjuntos de datos no lineales donde las clases están separadas por una función similar a la sigmoide.

<img src=".\assets\download-(17)-(1).webp" alt="How to Choose the Best Kernel Function for SVMs - GeeksforGeeks" />

#### Ventajas y limitaciones del uso de kernels

El uso de kernels en las SVM ofrece una enorme **flexibilidad**, ya que permite abordar problemas donde los datos no son linealmente separables en su espacio original. Gracias al **kernel trick**, las SVM son capaces de proyectar los datos a un espacio de mayor dimensión, donde una separación lineal se vuelve posible, sin necesidad de realizar cálculos explícitos en ese espacio. Esto hace que los kernels sean herramientas increíblemente poderosas en escenarios complejos, como la clasificación de imágenes, el reconocimiento de patrones o la detección de anomalías. Además, el kernel RBF, por ejemplo, es particularmente efectivo en problemas con fronteras de decisión no lineales, adaptándose a estructuras de datos altamente complejas.

Otra ventaja clave es la **eficiencia computacional** del kernel trick. Aunque los datos parecen estar en un espacio de dimensión mucho mayor, el cálculo real se reduce a evaluar productos escalares en el espacio original, lo que evita un crecimiento exponencial en los cálculos. Esto permite que las SVM trabajen incluso con transformaciones muy complejas, como las que se producen con el kernel polinómico de alto grado o el kernel RBF.

Sin embargo, el uso de kernels también tiene **limitaciones importantes**. Una de las mayores dificultades radica en el **ajuste de los hiperparámetros** asociados a ciertos kernels, como el parámetro $\sigma$ en el kernel RBF o el grado $d$ en el kernel polinómico. Estos parámetros afectan directamente la capacidad del modelo para generalizar a nuevos datos y, en muchos casos, encontrar su configuración óptima puede requerir un proceso exhaustivo de validación cruzada, que consume tiempo y recursos.

Otra desventaja significativa es la **escalabilidad**. A medida que crece el tamaño del conjunto de datos, el cálculo del kernel, que implica evaluar similitudes entre todos los pares de puntos, puede volverse prohibitivamente costoso. Por esta razón, el uso de kernels en SVM suele ser más adecuado para conjuntos de datos pequeños o medianos, mientras que para grandes volúmenes de datos se prefieren métodos más escalables como las redes neuronales o los árboles de decisión.

Finalmente, la **elección del kernel adecuado** no siempre es obvia. Si el kernel seleccionado no corresponde a la estructura intrínseca de los datos, el modelo puede tener un rendimiento inferior al esperado. Por ejemplo, el kernel lineal podría fallar en datos con relaciones complejas, mientras que el kernel RBF podría ser innecesariamente complejo en datos que son linealmente separables. Este desafío convierte la selección del kernel en un paso crítico que requiere tanto conocimiento teórico como experimentación práctica.

En resumen, los kernels son una herramienta extremadamente poderosa que amplía significativamente las capacidades de las SVM, pero requieren un enfoque cuidadoso tanto en su configuración como en su aplicación para evitar problemas de rendimiento o escalabilidad.

##### Para reflexionar...

> **¿Cómo decidirías qué kernel es el más adecuado para un conjunto de datos dado?** 
> **Clave**: Piensa en la relación entre las características y las clases. ¿Es lineal? ¿Es una relación compleja pero suave? Reflexiona también sobre la capacidad computacional disponible y el tamaño del conjunto de datos.

> **¿Qué desventajas puede tener el kernel RBF en comparación con un kernel lineal?** 
> **Clave**: Considera el costo computacional de trabajar en un espacio de dimensión infinita frente a un espacio más simple y directo como el de un kernel lineal.

### Implementación básica de SVM en Python con scikit-learn

Las Máquinas de Vectores de Soporte (SVM) son uno de los algoritmos más utilizados en Machine Learning, tanto por su capacidad para manejar problemas de clasificación como por su robustez frente a datos ruidosos. En esta sección, exploraremos cómo implementar SVM utilizando la biblioteca **scikit-learn** de Python, centrándonos en la clase `SVC` (Support Vector Classifier) para problemas de clasificación. Además, incluiremos una visualización de márgenes y fronteras de decisión para comprender mejor cómo opera este algoritmo, finalizando con un ejemplo práctico utilizando el dataset **Iris**.

#### Uso de `SVC` para clasificación

En scikit-learn, la clase `SVC` permite entrenar y evaluar modelos de SVM con gran facilidad. El modelo acepta varios hiperparámetros importantes, pero los más relevantes para una implementación básica son:

- **`C`**: Controla el equilibrio entre maximizar el margen y minimizar los errores de clasificación (regularización).  
  - Un valor grande de $C$ busca minimizar los errores en los datos de entrenamiento, pero puede sobreajustarse.
  - Un valor pequeño de $C$ prioriza un margen más amplio, mejorando la generalización. 
  - Matemáticamente, el término $C$ aparece en la función objetivo de las SVM:

$$
\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i
$$
    
Donde $\xi_i$ son las variables de relajación que permiten errores de clasificación en los márgenes blandos.

- **`kernel`**: Define la función kernel que se utilizará para transformar los datos (e.g., lineal, polinómico, RBF).  
  - El kernel más sencillo es el **lineal**, que busca una frontera en el espacio original de los datos:

$$
K(x_i, x_j) = x_i^T x_j
$$
    
  - En problemas no lineales, el **kernel RBF** es una opción popular, pues proyecta los datos a un espacio de dimensión infinita:

$$
K(x_i, x_j) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)
$$

Con estos conceptos básicos en mente, veamos cómo implementar una SVM para un problema de clasificación en Python.

Implementemos un ejemplo sencillo con datos generados artificialmente. El objetivo será clasificar puntos de dos clases ($+1$ y $-1$) en un plano bidimensional utilizando un **kernel lineal**.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# Generar un conjunto de datos sintético bidimensional con dos clases
X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.0, random_state=42)
y = 2 * y - 1  # Convertir las clases de 0, 1 a -1, +1

# Crear un modelo SVM con kernel lineal
model = SVC(kernel='linear', C=1)
model.fit(X, y)

# Obtener los coeficientes del hiperplano
w = model.coef_[0]
b = model.intercept_[0]

# Calcular la frontera de decisión y los márgenes
x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
decision_boundary = -(w[0] / w[1]) * x_vals - b / w[1]
margin_positive = decision_boundary + 1 / w[1]
margin_negative = decision_boundary - 1 / w[1]

# Identificar vectores de soporte
support_vectors = model.support_vectors_

# Gráfico
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=50, edgecolors='k', label="Datos")
plt.plot(x_vals, decision_boundary, 'k-', label="Frontera de decisión")
plt.plot(x_vals, margin_positive, 'k--', label="Margen positivo")
plt.plot(x_vals, margin_negative, 'k--', label="Margen negativo")
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', edgecolors='k', s=100,linewidth=1.5, label="Vectores de soporte")
plt.title("Márgenes, frontera de decisión y vectores de soporte")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

En este ejemplo práctico, comenzamos generando un conjunto de datos artificial compuesto por dos clases claramente separables en un espacio bidimensional. Para ello, utilizamos la función `make_blobs`, que nos permite simular puntos agrupados en dos grupos bien diferenciados. Estos datos servirán como base para entrenar nuestro modelo de SVM.

Una vez generados los datos, entrenamos un modelo de Máquina de Vectores de Soporte (SVM) utilizando un **kernel lineal** mediante la clase `SVC` de scikit-learn. Este kernel lineal es ideal para este problema, ya que los datos son separables mediante una línea recta en el espacio original de características.

Con el modelo entrenado, procedemos a calcular tanto la **frontera de decisión** como los **márgenes**. La frontera de decisión corresponde a la recta que separa las dos clases y está definida matemáticamente por la ecuación $w^T x + b = 0$. Los márgenes, por otro lado, son las rectas paralelas a la frontera de decisión, definidas por las ecuaciones $w^T x + b = \pm 1$, que delimitan la distancia máxima entre las clases antes de que un punto sea considerado ambiguo.

Además, identificamos los **vectores de soporte**, que son los puntos más cercanos a los márgenes y que determinan la posición exacta de la frontera de decisión. Estos puntos se obtienen con el atributo `model.support_vectors_` y tienen un rol crucial en la construcción del modelo.

Finalmente, visualizamos los resultados gráficamente. El gráfico resultante muestra la **frontera de decisión** como una línea sólida negra, mientras que los márgenes aparecen como líneas discontinuas que son paralelas a esta frontera. Los **vectores de soporte** se destacan con un contorno adicional, lo que permite observar claramente cuáles son los puntos que definen el hiperplano separador y el margen máximo entre las clases. Esta representación visual es clave para comprender cómo operan las SVM y cómo optimizan la separación entre clases en un problema de clasificación lineal.

##### Para reflexionar...

> **¿Cómo afecta el valor de $C$ al resultado del modelo en los ejemplos anteriores?** 
> **Clave**: Reflexiona sobre cómo $C$ controla la regularización, es decir, el equilibrio entre maximizar el margen y reducir los errores de clasificación.

> **¿Qué sucede si cambiamos el kernel lineal por un kernel RBF en el caso del dataset Iris?** 
> **Clave**: Considera cómo un kernel no lineal puede mejorar la separación en datos más complejos o con fronteras no lineales.

### SVM para clasificación y regresión en Scikit-Learn

Las Máquinas de Vectores de Soporte (SVM) son herramientas altamente versátiles que no solo permiten resolver problemas de **clasificación**, sino que también pueden aplicarse a problemas de **regresión**. La diferencia principal entre ambos enfoques radica en cómo el modelo interpreta los datos y optimiza el hiperplano. A continuación, exploramos las principales diferencias entre las variantes de SVM utilizadas en clasificación (`SVC`, `LinearSVC`) y regresión (`SVR`), así como el concepto clave de **epsilon-tube** en la regresión con SVM.

#### Problemas de clasificación con SVM: Diferencias entre `SVC`, `LinearSVC`

Hemos visto cómo en scikit-learn, existen diferentes implementaciones de SVM dependiendo del tipo de problema que se desee resolver. En el caso de los problemas de clasificación podemos encontrar dos posibilidades de abordaje de éstos desde la biblioteca.

##### `SVC` (Support Vector Classification)

Esta clase se utiliza para resolver problemas de clasificación general. El modelo busca encontrar un hiperplano que maximice el margen entre las clases, permitiendo manejar problemas lineales y no lineales dependiendo del kernel elegido. Los márgenes blandos se introducen para permitir algunos errores de clasificación, y el parámetro $C$ controla el equilibrio entre maximizar el margen y minimizar los errores. Matemáticamente, la frontera de decisión en el caso de un kernel lineal se define como:

$$
w^T x + b = 0
$$

##### `LinearSVC`

Esta es una versión optimizada de SVM que utiliza exclusivamente un kernel lineal. Está diseñada para problemas de clasificación con grandes conjuntos de datos, ya que emplea algoritmos más rápidos como *liblinear*. Aunque tiene una funcionalidad similar a `SVC(kernel='linear')`, carece de soporte para algunos hiperparámetros avanzados, como lo son los kernels no lineales.

#### Problemas de regresión: Uso de `SVR` (*Support Vector Regression*)

A diferencia de `SVC`, que trabaja con fronteras de decisión para separar clases, `SVR` está diseñado para resolver problemas de regresión. En este caso, el modelo no busca clasificar los datos, sino ajustar una función que prediga valores continuos. En lugar de maximizar márgenes, `SVR` optimiza una franja, llamada **epsilon-tube**, alrededor de la función predicha, donde los errores menores a un umbral $\epsilon$ no se penalizan. Veámoslo más detalladamente.

En la regresión tradicional, como la regresión lineal, el objetivo es minimizar directamente los errores entre los valores reales y los valores predichos. Sin embargo, en la regresión con SVM, el objetivo es encontrar una **función que minimice los errores grandes**, ignorando aquellos errores que caen dentro de una tolerancia predefinida $\epsilon$.

##### La franja epsilon-tube

El concepto de **epsilon-tube** se refiere a una franja (o tubo) alrededor de la función de predicción dentro de la cual los errores no son penalizados. Esto significa que si la predicción de un punto está a una distancia menor o igual a $\epsilon$ del valor real, el modelo lo considera como un ajuste aceptable. Los puntos fuera de esta franja son penalizados, y los vectores de soporte en SVR son los puntos que se encuentran fuera o exactamente en el límite de esta franja.

Matemáticamente, el problema de optimización en SVR se plantea como minimizar la función:

$$
\frac{1}{2} \|w\|^2 + C \sum_{i=1}^N (\xi_i + \xi_i^*)
$$

Sujeto a:

$$
y_i - (w^T x_i + b) \leq \epsilon + \xi_i
$$

$$
(w^T x_i + b) - y_i \leq \epsilon + \xi_i^*
$$

$$
\xi_i, \xi_i^* \geq 0
$$

Donde:

- $w$ es el vector de pesos que define la función predicha.
- $b$ es el sesgo o bias.
- $\xi_i$ y $\xi_i^*$ son términos de relajación que permiten manejar errores fuera de la franja epsilon-tube.
- $\epsilon$ es el tamaño de la franja.

El término $C$ nuevamente controla el trade-off entre la regularización (maximizar el margen) y los errores (penalizar puntos fuera del tubo).

> **Ejemplo:**
>
> Imaginemos que queremos predecir los precios de casas según su tamaño y distancia al centro de la ciudad. En lugar de forzar al modelo a ajustar perfectamente todos los puntos, permitimos un margen de error de, por ejemplo, \$5000. Esto significa que cualquier predicción que esté dentro de este rango de tolerancia (el epsilon-tube) no será penalizada, enfocando el modelo en evitar errores grandes y no en ajustar cada punto exactamente.



##### Para reflexionar...

> **¿Por qué es útil el concepto de epsilon-tube en problemas de regresión?** 
> **Clave**: Reflexiona sobre cómo este concepto permite enfocarse en errores significativos y evitar sobreajustar pequeñas fluctuaciones en los datos.

> **¿Cómo afecta el parámetro $\epsilon$ al ajuste del modelo en SVR?** 
> **Clave**: Considera cómo un valor grande de $\epsilon$ permite una mayor tolerancia a errores, mientras que un valor pequeño hace que el modelo sea más sensible a los datos.



### Ajuste de hiperparámetros en SVM

El ajuste de hiperparámetros es una etapa crucial para mejorar el desempeño de las Máquinas de Vectores de Soporte (SVM). Aunque las SVM son algoritmos robustos, su rendimiento depende en gran medida de la configuración adecuada de tres hiperparámetros principales: **`C`**, **`gamma`** y **`kernel`**. Cada uno de ellos influye directamente en la forma en que el modelo se adapta a los datos de entrenamiento y generaliza a nuevos datos. En esta sección, exploraremos el rol de estos parámetros y cómo ajustarlos utilizando herramientas como `GridSearchCV`. También discutiremos su impacto en la capacidad de generalización del modelo.

#### El rol de los parámetros `C`, `gamma` y `kernel`

##### El parámetro `C`

El hiperparámetro $C$ controla el **trade-off** entre el margen máximo y la penalización por errores de clasificación en los datos de entrenamiento. Matemáticamente, aparece en la función objetivo de las SVM como:

$$
\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i
$$

Un valor alto de $C$ da prioridad a minimizar los errores en el conjunto de entrenamiento, lo que obliga al modelo a clasificar correctamente tantos puntos como sea posible. Sin embargo, esto puede llevar a un **sobreajuste**, ya que el modelo podría ajustarse demasiado a las particularidades de los datos de entrenamiento. Por otro lado, un valor bajo de \$1\$ permite más errores de clasificación, lo que resulta en márgenes más amplios y un modelo con mejor **capacidad de generalización**, especialmente si los datos son ruidosos.

##### El parámetro `gamma`

El parámetro $\gamma$ es relevante para kernels como el RBF (Radial Basis Function) y controla el **alcance de la influencia de un punto de datos individual**. En el caso del kernel RBF, el valor de $\gamma$ aparece en la fórmula:

$$
K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)
$$

Cuando $\gamma$ es grande, cada punto tiene un alcance muy limitado, lo que hace que el modelo sea más complejo y propenso al sobreajuste, ya que intenta capturar los detalles de cada punto. En contraste, un valor pequeño de $\gamma$ implica que los puntos tienen una influencia más amplia, lo que da como resultado un modelo más simple y más generalizable.

##### El parámetro `kernel`

El kernel define la **transformación de los datos** al espacio donde se realiza la separación lineal. Los kernels disponibles en SVM permiten modelar diferentes tipos de relaciones entre las características. Por ejemplo:
- El **kernel lineal** es ideal cuando las clases son separables en el espacio original de características.
- El **kernel polinómico** transforma los datos a un espacio de mayor dimensión utilizando términos polinómicos.
- El **kernel RBF** mapea los datos a un espacio de dimensión infinita, lo que lo hace especialmente útil para problemas no lineales.

La elección del kernel afecta significativamente el desempeño del modelo, y es importante probar diferentes opciones dependiendo de la estructura de los datos.

#### Ejemplo práctico de ajuste con `GridSearchCV`

El ajuste de hiperparámetros puede realizarse de forma eficiente utilizando `GridSearchCV` en scikit-learn. Este método evalúa combinaciones de hiperparámetros mediante validación cruzada, seleccionando la configuración que optimiza una métrica de desempeño.

Veamos un ejemplo práctico:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# Generar un conjunto de datos artificial
X, y = make_classification(n_samples=200, n_features=2, n_classes=2, random_state=42, n_informative=2, n_redundant=0)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir el modelo SVM
svm = SVC()

# Configurar los hiperparámetros a probar
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

# Implementar GridSearchCV
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Imprimir los mejores parámetros y resultados
print("Mejores parámetros:", grid_search.best_params_)
print("Mejor puntuación de validación cruzada:", grid_search.best_score_)

# Evaluar el modelo con los mejores parámetros en el conjunto de prueba
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
```

En este ejemplo práctico, demostramos cómo ajustar los hiperparámetros de una SVM utilizando la herramienta `GridSearchCV` de scikit-learn. El proceso comienza con la generación de un conjunto de datos artificial mediante la función `make_classification`. Este dataset contiene 200 puntos con dos clases, distribuidos en un espacio bidimensional, donde las características son informativas para la clasificación. Posteriormente, dividimos los datos en conjuntos de entrenamiento y prueba utilizando `train_test_split`, reservando un 30% de los datos para evaluar el modelo final.

En el siguiente paso configuramos el modelo de SVM utilizando la clase `SVC`. Sin embargo, en lugar de establecer manualmente los valores de los hiperparámetros como `C`, `gamma` o el tipo de kernel, definimos un rango de posibles valores para cada uno en un diccionario llamado `param_grid`. Este rango incluye valores típicos como $C = [0.1, 1, 10]$, $\gamma = [0.01, 0.1, 1]$, y los kernels `'linear'` y `'rbf'`. Este conjunto de combinaciones nos permitirá explorar diferentes configuraciones del modelo y encontrar la que maximice el desempeño.

Con la configuración de los hiperparámetros lista, implementamos `GridSearchCV`, una herramienta que realiza una búsqueda exhaustiva probando todas las combinaciones de parámetros en el diccionario. Además, utilizamos validación cruzada con 5 particiones (definida por el parámetro `cv=5`) para evaluar el rendimiento del modelo en diferentes subconjuntos del conjunto de entrenamiento. Esto asegura que la configuración óptima encontrada no dependa exclusivamente de una partición específica de los datos.

Tras completar la búsqueda, `GridSearchCV` nos proporciona los **mejores valores de hiperparámetros**, que se almacenan en el atributo `best_params_`. También obtenemos el **mejor modelo entrenado** con estos hiperparámetros óptimos, accesible a través de `best_estimator_`.

Finalmente, evaluamos el mejor modelo en el conjunto de prueba para comprobar su desempeño. Para ello, realizamos predicciones sobre los datos de prueba y generamos un reporte de clasificación usando `classification_report`. Este informe nos permite analizar métricas como la precisión, la sensibilidad (recall) y el puntaje F1, brindando una visión completa del rendimiento del modelo ajustado.

Este enfoque asegura que el modelo no solo se ajuste bien a los datos de entrenamiento, sino que también generalice adecuadamente a nuevos datos, optimizando su capacidad predictiva.

#### Impacto de los hiperparámetros en la capacidad de generalización

El ajuste adecuado de los hiperparámetros es esencial para lograr un modelo equilibrado que generalice bien a nuevos datos. Por ejemplo, **un valor alto de $C$** puede llevar a un modelo que se ajuste demasiado a los datos de entrenamiento, resultando en un bajo error en el entrenamiento pero un pobre desempeño en el conjunto de prueba debido al sobreajuste. Por su parte **un valor bajo de $\gamma$** suaviza el modelo al permitir que los puntos tengan una influencia más amplia, lo que ayuda a evitar que el modelo se ajuste demasiado a las pequeñas variaciones de los datos. Por último, la **elección del kernel** puede ser crítica. Un kernel lineal puede ser insuficiente si los datos tienen una relación no lineal, mientras que un kernel RBF puede capturar relaciones complejas, aunque con un mayor riesgo de sobreajuste si no se regula correctamente.

Un modelo bien ajustado encuentra el balance ideal entre **complejidad** y **generalización**, asegurando que el modelo sea flexible pero no excesivamente específico para los datos de entrenamiento.



##### Para reflexionar...

> **¿Cómo puedes determinar si un modelo SVM está sobreajustado o subajustado según los valores de los hiperparámetros?**  
> **Clave**: Reflexiona sobre las métricas de desempeño en entrenamiento y validación cruzada, y cómo los valores de $C$ y $\gamma$ pueden influir en ambos.

> **¿Por qué es importante usar validación cruzada durante el ajuste de hiperparámetros?**  
> **Clave**: Considera cómo la validación cruzada permite evaluar el modelo en múltiples subconjuntos de datos, evitando que el modelo se ajuste solo a un conjunto específico de entrenamiento.



### Manejo de datos desbalanceados

Ya hemos visto a lo largo del curso que en muchos problemas reales de clasificación, de hecho son mayoría, los datos están desbalanceados. Es decir, una clase puede estar representada por un número significativamente mayor de ejemplos en comparación con otra. Este desbalance puede afectar el desempeño del modelo, ya que los algoritmos tienden a favorecer la clase mayoritaria, lo que resulta en un modelo con alta precisión global pero mal desempeño en la clase minoritaria. En esta sección, exploraremos cómo manejar este tipo de situaciones al entrenar SVM, con técnicas como el uso de pesos balanceados, sobremuestreo y submuestreo, además de una discusión sobre las métricas adecuadas para evaluar modelos en estos casos. Algunas técnicas ya sonarán de otros capítulos, otras son propias de este tipo de modelos. Adoptaremos un punto de cista operativo basado en el uso de la biblioteca skit-learn.

#### Uso de `class_weight='balanced'` y ajuste manual de pesos

Una de las maneras más sencillas de lidiar con datos desbalanceados en SVM es asignar **pesos mayores a la clase minoritaria** para contrarrestar el efecto de su menor representación. En scikit-learn, esto se puede lograr fácilmente utilizando el parámetro `class_weight` en la clase `SVC`. Cuando se establece como `'balanced'`, los pesos de las clases se calculan automáticamente como inversamente proporcionales a sus frecuencias en los datos de entrenamiento:

$$
w_i = \frac{\text{número total de muestras}}{\text{número de clases} \times \text{número de muestras en la clase } i}
$$

Esto significa que las clases menos representadas reciben un peso mayor en la función objetivo de las SVM, dándoles más importancia en el proceso de optimización.

Si necesitas más control, también puedes especificar manualmente los pesos de cada clase utilizando un diccionario con el formato `{clase: peso}`. Por ejemplo, si la clase 0 tiene 1000 muestras y la clase 1 solo tiene 100, podrías asignar pesos de forma proporcional:

```python
class_weight = {0: 1, 1: 10}
```

Estos pesos modifican la penalización asociada a los errores de clasificación en las diferentes clases, asegurando que el modelo preste atención tanto a la clase mayoritaria como a la minoritaria.

#### Técnicas adicionales: Sobremuestreo y submuestreo

Aunque el ajuste de pesos es una técnica muy útil, no siempre es suficiente, especialmente en casos con un alto desbalance. En estos escenarios, técnicas como el **sobremuestreo** y el **submuestreo** pueden ser muy efectivas.

##### Sobremuestreo
El sobremuestreo consiste en aumentar artificialmente el número de ejemplos de la clase minoritaria. Una técnica común es duplicar aleatoriamente las muestras de la clase minoritaria hasta igualar el tamaño de la clase mayoritaria. Sin embargo, un método más avanzado y efectivo es el **SMOTE** (Synthetic Minority Oversampling Technique), que genera nuevas muestras sintéticas interpolando puntos existentes de la clase minoritaria. Esto ayuda a reducir el problema del sobreajuste que puede surgir al simplemente duplicar ejemplos.

##### Submuestreo
El submuestreo, por otro lado, reduce el tamaño de la clase mayoritaria eliminando ejemplos hasta igualar el tamaño de la clase minoritaria. Esto asegura un equilibrio en el número de muestras, pero puede resultar en pérdida de información importante si se descartan datos relevantes de la clase mayoritaria. Es especialmente útil cuando el conjunto de datos es lo suficientemente grande como para tolerar esta reducción.

Ambas técnicas pueden combinarse en un enfoque híbrido, donde se aplica un poco de sobremuestreo a la clase minoritaria y submuestreo a la clase mayoritaria para lograr un balance adecuado.

En Python, estas técnicas pueden implementarse con la ayuda de bibliotecas como **`imbalanced-learn`**, que proporciona herramientas avanzadas para el manejo de datos desbalanceados.

#### Métricas de evaluación adecuadas para datos desbalanceados

Repasemos de nuevo las métricas que evalúan el desempeño de un modelo con datos de entrenamiento desbalanceados. Ello requiere algo más que analizar la exactitud global ($\text{accuracy}$). En este tipo de problemas, un modelo que siempre prediga la clase mayoritaria podría tener una alta precisión, pero sería inútil para detectar la clase minoritaria. Por ello, es fundamental utilizar métricas que ofrezcan una evaluación más equilibrada del desempeño del modelo. Para calcular las métricas de clasificación es requisito imprescindible construir la denominada **matriz de confusión**. La matriz de confusión permite visualizar el desempeño del modelo, mostrando las verdaderas predicciones positivas y negativas, junto con los errores en cada clase. Es una herramienta esencial para interpretar resultados en problemas desbalanceados. Las métricas asociadas a la matriz de confusión eran:

##### **Precision**

Evalúa la proporción de predicciones correctas para una clase específica. Es especialmente útil cuando los falsos positivos tienen un alto costo.

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

##### **Recall (Sensibilidad)**

Mide la capacidad del modelo para identificar correctamente los ejemplos de una clase específica. Es crucial cuando los falsos negativos son costosos.

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

##### **F1-score**

Es la media armónica entre la precisión y el recall. Proporciona un balance entre ambos, especialmente útil cuando existe un desbalance significativo.

$$
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$



##### **Curva ROC y AUC (Área Bajo la Curva)**

La curva ROC muestra la relación entre el recall (verdaderos positivos) y la tasa de falsos positivos. El AUC es un resumen de esta curva y ofrece una medida global de qué tan bien separa el modelo las dos clases.



##### Ejemplo práctico: Clasificación en un conjunto de datos desbalanceado

A continuación, presentamos un ejemplo práctico para implementar el manejo de datos desbalanceados utilizando el parámetro `class_weight='balanced'` y métricas de evaluación adecuadas:

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Generar un conjunto de datos desbalanceado
X, y = make_classification(
    n_samples=1000,          # Número total de muestras
    n_features=2,            # Total de características
    n_informative=2,         # Número de características informativas (igual a n_features en este caso)
    n_redundant=0,           # Ninguna característica es redundante
    n_classes=2,             # Número de clases (binario)
    weights=[0.9, 0.1],      # Distribución de las clases (90% clase 0, 10% clase 1)
    random_state=42          # Reproducibilidad
)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar un modelo SVM con pesos balanceados
model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)

# Métricas de evaluación
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
```

En este ejemplo práctico, abordamos el problema de clasificar un conjunto de datos desbalanceado utilizando SVM y técnicas para equilibrar la influencia de las clases en el modelo. Comenzamos generando un conjunto de datos artificial mediante la función `make_classification`. Este dataset simula una situación típica de desbalance, donde el 90% de las muestras pertenecen a la clase mayoritaria y solo el 10% a la clase minoritaria. Este tipo de distribución refleja escenarios del mundo real, como la detección de fraudes, donde los eventos de interés (fraudes) son significativamente menos frecuentes que los no fraudulentos.

A continuación, dividimos los datos en conjuntos de entrenamiento y prueba utilizando la función `train_test_split`. Reservamos el 30% de los datos para evaluar el modelo, asegurándonos de mantener el desbalance proporcional en ambas particiones. Esto permite que el modelo se entrene y se evalúe en condiciones similares a las que encontraría en un entorno real.

Para lidiar con el desbalance, utilizamos el parámetro `class_weight='balanced'` en el modelo SVM, implementado con la clase `SVC`. Este ajuste automático asigna pesos a las clases de forma inversamente proporcional a su frecuencia en el conjunto de datos, de manera que los errores en la clase minoritaria se penalizan más que los errores en la clase mayoritaria. Al elegir un kernel RBF, también aseguramos que el modelo sea capaz de capturar relaciones no lineales entre las características de los datos.

Tras entrenar el modelo en los datos de entrenamiento, evaluamos su desempeño en el conjunto de prueba. Esto se realiza prediciendo las etiquetas para los datos de prueba y analizando el resultado mediante una matriz de confusión y un reporte de clasificación. La matriz de confusión nos proporciona una visión detallada de las verdaderas predicciones positivas y negativas, así como de los errores cometidos en ambas clases. Por otro lado, el reporte de clasificación incluye métricas importantes como la precisión, el recall y el F1-score, que son especialmente útiles para medir el desempeño en problemas desbalanceados.

Al final, este enfoque nos permite evaluar cómo el modelo equilibra correctamente la importancia de las clases y evita favorecer en exceso a la clase mayoritaria, logrando un desempeño más justo y adecuado en escenarios con datos desbalanceados. Este ejemplo destaca la importancia de combinar ajustes en el modelo y métricas de evaluación específicas para abordar este tipo de problemas de manera efectiva.

> [!Tip]
>
> El manejo adecuado de datos desbalanceados es esencial para evitar modelos sesgados hacia la clase mayoritaria. Ajustar los pesos con `class_weight`, aplicar técnicas de sobremuestreo o submuestreo, y utilizar métricas de evaluación específicas son estrategias clave para construir modelos más equilibrados y efectivos en estos escenarios. 
>



##### Para reflexionar...

> **¿Por qué crees que la precisión global no es una métrica adecuada para evaluar modelos en datos desbalanceados?** 
> **Clave**: Considera cómo un modelo que siempre predice la clase mayoritaria puede tener alta precisión pero ignorar completamente la clase minoritaria.

> **¿Cómo afecta el sobremuestreo al tiempo de entrenamiento de un modelo SVM?** 
> **Clave**: Reflexiona sobre cómo el aumento en el tamaño del conjunto de datos podría influir en el costo computacional.

### Ejemplos prácticos avanzados con Python  

En esta sección, exploraremos el uso de SVM en escenarios más complejos, aplicando herramientas avanzadas de Python. Comenzaremos resolviendo un problema de clasificación no lineal en datasets clásicos, los denominados problemas de "circles" y "moons". Posteriormente, trabajaremos con **Support Vector Regression (SVR)** en un caso práctico de regresión. Finalmente, utilizaremos **validación cruzada** para evaluar y analizar el desempeño del modelo, asegurándonos de que nuestras conclusiones sean generalizables a nuevos datos.

#### Clasificación en un dataset no lineal: Circles y moons  

Una de las fortalezas de las SVM es su capacidad para manejar datos no lineales mediante el uso de kernels. Los datasets "circles" (círculos concéntricos) y "moons" (semilunas) son ejemplos clásicos de problemas no lineales en los que un kernel lineal no puede separar las clases correctamente. Aquí utilizaremos el **kernel RBF**, que proyecta los datos a un espacio de mayor dimensión para que sean linealmente separables. Veámoslo en el siguiente ejemplo

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Generar datos no lineales: círculos y semilunas
X_circles, y_circles = make_circles(n_samples=500, factor=0.3, noise=0.05, random_state=42)
X_moons, y_moons = make_moons(n_samples=500, noise=0.1, random_state=42)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_circles, y_circles, test_size=0.3, random_state=42)

# Entrenar un modelo SVM con kernel RBF
model = SVC(kernel='rbf', C=1, gamma=0.5)
model.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = model.predict(X_test)
print("Reporte de clasificación (Círculos):")
print(classification_report(y_test, y_pred))

# Visualización de los resultados
xx, yy = np.meshgrid(np.linspace(X_circles[:, 0].min() - 0.5, X_circles[:, 0].max() + 0.5, 500),
                     np.linspace(X_circles[:, 1].min() - 0.5, X_circles[:, 1].max() + 0.5, 500))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='autumn', edgecolors='k', s=50, label='Datos')
plt.contourf(xx, yy, Z > 0, alpha=0.2, levels=1, cmap='coolwarm')
plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], linestyles=['--', '-', '--'])
plt.title("SVM con kernel RBF (Círculos)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()
```

En este ejemplo, abordamos el problema de clasificar datos no lineales utilizando SVM con un kernel RBF. Para empezar, generamos dos conjuntos de datos clásicos: "círculos" y "moons". Estos datasets son ampliamente utilizados para evaluar modelos en problemas donde las clases no pueden separarse con una línea recta. En particular, los datos de "círculos" consisten en dos anillos concéntricos, mientras que los de "moons" representan dos formas semicirculares entrelazadas. Además, añadimos un pequeño nivel de aleatoriedad utilizando el parámetro `noise` para simular escenarios más realistas y desafiantes.

Con los datos generados, entrenamos un modelo SVM utilizando un kernel RBF. Este kernel es ideal para manejar datos no lineales, ya que proyecta los puntos a un espacio de mayor dimensión donde las clases pueden ser separadas de manera lineal. Configuramos el modelo con $C = 1$ y $\gamma = 0.5$, lo que nos permite balancear la penalización por errores y la complejidad del modelo. Este ajuste asegura que el modelo pueda capturar las relaciones no lineales entre las clases sin caer en el sobreajuste.

Después del entrenamiento, evaluamos el desempeño del modelo en el conjunto de prueba generando un reporte de clasificación. Este reporte incluye métricas clave como precisión, recall y F1-score, que nos ofrecen una visión detallada de la capacidad del modelo para distinguir entre las dos clases.

Finalmente, visualizamos los resultados de manera gráfica. En el gráfico, mostramos los puntos de datos coloreados según sus clases, junto con la frontera de decisión generada por la SVM. También destacamos los márgenes, que indican el área en la que el modelo define una separación entre las clases. Este tipo de visualización no solo nos permite evaluar visualmente el desempeño del modelo, sino que también proporciona una comprensión intuitiva de cómo la SVM maneja problemas no lineales mediante el kernel RBF.

#### Aplicación de SVR en un problema de regresión real  

En este ejemplo, usamos **Support Vector Regression (SVR)** para ajustar un modelo a datos continuos. Simularemos un conjunto de datos que representa una función no lineal y veremos cómo el modelo SVR puede predecir valores con precisión. Tienes el código completo en el siguiente apartado.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generar un conjunto de datos no lineal
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + 0.1 * np.random.randn(100)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar un modelo SVR con kernel RBF
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_rbf.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = svr_rbf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (SVR): {mse:.3f}")

# Visualización de los resultados
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='darkorange', label="Datos reales")
plt.plot(X_test, y_pred, color='navy', lw=2, label="Predicción SVR")
plt.title("Regresión con SVM (SVR)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

En este ejemplo, abordamos el problema de regresión utilizando **Support Vector Regression (SVR)** para ajustar un modelo a datos continuos con una relación no lineal. Para comenzar, generamos un conjunto de datos basado en la función seno, a la que añadimos un pequeño ruido aleatorio para simular un escenario más realista. Este tipo de datos refleja situaciones comunes en problemas de regresión, donde la variable de respuesta $y$ sigue un patrón no lineal con cierto nivel de variación debido a ruido o incertidumbre en las observaciones.

El siguiente paso es entrenar un modelo SVR utilizando un **kernel RBF**. Este kernel es especialmente útil para capturar relaciones no lineales entre las variables independientes $X$ y dependientes $y$. Configuramos el modelo con tres hiperparámetros clave:
- $C$, que controla la penalización por errores grandes, permitiendo un ajuste más preciso pero con el riesgo de sobreajuste si se elige un valor muy alto.  
- $\gamma$, que regula el alcance de la influencia de cada punto de datos; un valor bajo genera una curva suave, mientras que un valor alto puede hacer que el modelo sea excesivamente complejo.  
- $\epsilon$, que define el margen de tolerancia para los errores pequeños, conocido como el **epsilon-tube**. Este parámetro asegura que no todos los errores sean penalizados, centrándose únicamente en aquellos que se encuentran fuera de esta franja de tolerancia.

Una vez entrenado el modelo, evaluamos su desempeño en el conjunto de prueba calculando el **error cuadrático medio (MSE)**. Esta métrica cuantifica la diferencia promedio entre los valores reales y los valores predichos, ofreciendo una medida clara de la precisión del modelo.

Por último, visualizamos los resultados para interpretar mejor cómo se ajusta el modelo SVR a los datos. En el gráfico, mostramos los puntos de datos originales como una referencia, junto con la curva predicha por el modelo. Esta representación nos permite observar visualmente cómo el modelo SVR logra capturar la relación no lineal subyacente, ignorando el ruido presente en los datos. La curva predicha refleja un equilibrio entre ajuste a los datos de entrenamiento y capacidad de generalización, lo que es fundamental en problemas de regresión real.

#### Uso de validación cruzada y análisis de resultados  

La validación cruzada es una técnica clave para evaluar el desempeño de un modelo en diferentes subconjuntos de datos, reduciendo el riesgo de sobreajuste o subajuste. A continuación, en el siguiente ejemplo de código, aplicamos validación cruzada para analizar el desempeño de un modelo SVM.

```python
from sklearn.model_selection import cross_val_score

# Validación cruzada con SVC
scores = cross_val_score(SVC(kernel='rbf', C=1, gamma=0.1), X_train, y_train, cv=5, scoring='accuracy')
print(f"Accuracy promedio en validación cruzada: {scores.mean():.3f}")
print(f"Desviación estándar de la accuracy: {scores.std():.3f}")
```

En este ejemplo, utilizamos la técnica de **validación cruzada** para evaluar el desempeño de un modelo SVM en diferentes subconjuntos de los datos de entrenamiento, lo que nos permite obtener una evaluación más fiable y robusta del modelo.

Comenzamos definiendo un **modelo SVM con kernel RBF**, configurando sus hiperparámetros clave. Establecemos $C = 1$, lo que asegura un equilibrio entre la maximización del margen y la minimización de los errores de clasificación, y $\gamma = 0.1$, que controla el alcance de la influencia de cada punto de datos. Un valor bajo de $\gamma$ ayuda a que el modelo mantenga una curva de decisión suave y generalizable, ideal para datos que no son perfectamente separables.

A continuación, implementamos la **validación cruzada** utilizando 5 particiones (5-fold cross-validation). Este proceso divide los datos de entrenamiento en 5 subconjuntos, entrenando el modelo en 4 de ellos y evaluándolo en el restante. Este procedimiento se repite 5 veces, alternando el subconjunto usado para la evaluación en cada iteración. De esta forma, el modelo se prueba en todos los subconjuntos posibles, y se calcula un promedio de su desempeño. Esto permite reducir el riesgo de que la evaluación dependa de una división específica de los datos.

Finalmente, realizamos un **análisis de los resultados** obtenidos a partir de la validación cruzada. Calculamos la precisión promedio, que refleja el desempeño global del modelo en las 5 particiones, y la desviación estándar, que mide la variabilidad entre las particiones. Un modelo con una precisión promedio alta y una desviación estándar baja es un indicador de estabilidad y robustez, lo que significa que el modelo generaliza bien a diferentes subconjuntos de datos.

Este enfoque de validación cruzada no solo evalúa la calidad del modelo, sino que también ayuda a identificar posibles problemas de sobreajuste o subajuste, asegurando que las decisiones sobre los hiperparámetros sean informadas y respaldadas por resultados consistentes en múltiples particiones.

---

##### Para reflexionar...  

> **¿Cómo afecta el parámetro $\gamma$ al ajuste del modelo en problemas no lineales?** 
> **Clave**: Piensa en cómo un $\gamma$ alto puede capturar más detalles pero corre el riesgo de sobre ajustarse, mientras que un $\gamma$ bajo genera modelos más simples y generalizables.

> **¿Qué ventajas tiene el uso de validación cruzada frente a la simple división en conjuntos de entrenamiento y prueba?** 
> **Clave**: Reflexiona sobre cómo la validación cruzada evalúa el desempeño en múltiples particiones, reduciendo la dependencia de una sola división específica.

### Limitaciones de las SVM y alternativas  

Aunque las Máquinas de Vectores de Soporte (SVM) son un algoritmo poderoso y versátil, no están exentas de limitaciones. Es fundamental entender estos desafíos para saber cuándo usarlas y cuándo optar por alternativas más adecuadas. En esta sección, exploramos las principales limitaciones de las SVM, en especial su **escalabilidad en datasets grandes**, y comparamos su desempeño y características con otros algoritmos populares, como la **regresión logística** y los **árboles de decisión**.

#### Escalabilidad de las SVM en datasets grandes  

Una de las mayores limitaciones de las SVM es su **costo computacional**. El entrenamiento de una SVM requiere resolver un problema de optimización cuadrática, cuyo tiempo de ejecución escala de forma **no lineal** con el tamaño del dataset. Específicamente, el entrenamiento tiene una complejidad aproximada de $O(N^2)$ a $O(N^3)$, donde $N$ es el número de muestras en el conjunto de datos. Esto hace que las SVM sean muy eficientes para datasets pequeños o medianos, pero problemáticas para conjuntos de datos grandes, como aquellos que contienen cientos de miles o millones de muestras.

Además, las SVM basadas en kernels pueden requerir almacenar una matriz de $N \times N$, conocida como **matriz kernel**, lo que implica un alto consumo de memoria. Esto las vuelve inapropiadas para problemas de alta dimensionalidad combinados con un gran número de muestras, como ocurre en big data o problemas de texto y visión por computadora con características extraídas automáticamente.

Si bien existen variaciones de las SVM, como las **Linear SVM** implementadas en scikit-learn con `LinearSVC`, estas están diseñadas exclusivamente para kernels lineales y son más escalables gracias a algoritmos optimizados como *liblinear*. Sin embargo, estas versiones pierden la flexibilidad de los kernels no lineales, lo que limita su aplicabilidad en problemas más complejos.

> [!Note]
>
> **¿A qué llamamos *complejidad computacional*?**
>
> El concepto de **complejidad computacional** es una medida fundamental en ciencias de la computación que describe los **recursos necesario para ejecutar un algoritmo**. Estos recursos suelen medirse en términos de **tiempo** (es decir, cuántas operaciones requiere el algoritmo para completarse) y **espacio** (la cantidad de memoria necesaria para ejecutarlo). En pocas palabras, la complejidad computacional nos dice **qué tan rápido** y **con cuánta memoria** puede ejecutarse un algoritmo en función del tamaño de la entrada.
>
> Entender la complejidad computacional es crucial porque no todos los algoritmos son igualmente eficientes. Un algoritmo que funciona bien con pocos datos puede volverse impráctico o incluso inutilizable si el tamaño de los datos crece considerablemente. Por ejemplo, un algoritmo que toma minutos en procesar 1000 datos podría tardar días o semanas si el número de datos aumenta a millones, dependiendo de su complejidad.
>
> La complejidad computacional suele expresarse utilizando la notación **Big-O** ($O$), que describe cómo escala el tiempo o el espacio necesario a medida que crece el tamaño de los datos ($N$). Enumeramos a continuación los tipos de problemas más habituales.
>
> **Constante** $O(1)$: El tiempo o espacio requerido por el algoritmo es **independiente** del tamaño de la entrada. Es la menor complejidad posible.  Por ejemplo, acceder al valor de un elemento en un array por su índice.
>
> **Logarítmica** $O(\log N)$: El tiempo o espacio aumenta lentamente a medida que crece el tamaño de los datos. Por ejemplo, una búsqueda binaria en una lista ordenada.
>
> **Lineal** $O(N)$: El tiempo o espacio requerido crece de manera **proporcional** al tamaño del dataset. Por ejemplo, recorrer todos los elementos de una lista.
>
> **Cuadrática** $O(N^2)$: El tiempo o espacio crece de manera proporcional al **cuadrado del tamaño de los datos**. Este tipo de complejidad puede ser ineficiente para datasets grandes. Un ejemplo típico es comparación de pares de elementos en un algoritmo de fuerza bruta.
>
> **Exponencial** $O(2^N)$: El tiempo o espacio crece exponencialmente con el tamaño de los datos, lo que hace que este tipo de algoritmos no sea **práctico** incluso para datasets moderadamente grandes.  Por ejemplo, los algoritmos de fuerza bruta para resolver problemas combinatorios como el "viajante de comercio" pertenecerían a este tipo.
>
> Sin embargo, cuando hablamos de complejidad computacional, solemos referirnos principalmente a la **complejidad en tiempo**, es decir, cuánto tiempo le toma a un algoritmo ejecutarse. En este sentido, no puede obviarse que, en algunos casos, la **complejidad de espacio** (memoria necesaria para implementar el algoritmo) también es crítica, especialmente en problemas donde el algoritmo necesita manejar grandes cantidades de datos en memoria.
>
> Por ejemplo, el entrenamiento de una **SVM con kernels no lineales** tiene una complejidad de tiempo de aproximadamente $O(N^3)$, ya que implica cálculos iterativos sobre todos los pares de datos. Además, requiere almacenar una **matriz kernel** de tamaño $N \times N$, lo que genera una **complejidad de espacio cuadrática** $(O(N^2))$. Estas limitaciones hacen que el entrenamiento de SVM sea prohibitivo en datasets con millones de muestras.
>
> En conclusión: La **complejidad computacional** nos ayuda a entender las limitaciones y la eficiencia de un algoritmo, especialmente cuando los datos crecen en tamaño. Aunque ciertos algoritmos son ideales para datasets pequeños o medianos, su complejidad puede hacerlos ineficientes o incluso inutilizables en problemas a gran escala. Por eso, conocer la complejidad computacional es crucial para elegir el algoritmo adecuado para cada tarea y optimizar el uso de recursos como tiempo y memoria.

#### Comparación con regresión logística y árboles de decisión  

En ciertos problemas, especialmente cuando el volumen de datos es grande, otros algoritmos como la **regresión logística** y los **árboles de decisión** pueden ser alternativas más prácticas. Cada uno tiene ventajas y desventajas que dependen de las características del problema que se desea resolver.

##### **Regresión logística**

La regresión logística comparte similitudes con las SVM en problemas de clasificación lineal. Ambos algoritmos buscan encontrar una frontera de decisión en el espacio de características. Sin embargo, mientras que las SVM maximizan el margen entre las clases, la regresión logística optimiza la probabilidad condicional de una clase mediante la función sigmoide. Esto le otorga algunas ventajas clave. Primeramente, es más rápida de entrenar en datasets grandes, ya que tiene una complejidad computacional más baja que las SVM. En segundo lugar, ofrece interpretabilidad al proporcionar probabilidades para cada clase, lo que es útil en aplicaciones donde la **confianza en la predicción** es importante. 

Sin embargo, las SVM tienden a ser más efectivas en problemas donde las clases no son perfectamente separables, gracias a su capacidad para manejar márgenes blandos.

##### **Árboles de decisión** 

Por otro lado, los árboles de decisión son una opción flexible y fácil de interpretar, que se adapta bien a problemas con relaciones no lineales entre las características. A diferencia de las SVM, los árboles no requieren normalización de las características, lo que simplifica el preprocesamiento de los datos. También son más rápidos de entrenar, incluso en conjuntos de datos grandes, ya que los cálculos de partición son eficientes. Por último, ofrecen interpretabilidad visual, lo que permite a los usuarios entender cómo se toman las decisiones en cada nodo del árbol.

Sin embargo, los árboles de decisión presentan limitaciones, como una mayor propensión al **sobreajuste** en datos ruidosos o de alta dimensionalidad. Además, su rendimiento puede ser inferior al de las SVM en problemas bien definidos con márgenes claros entre clases. En este sentido, técnicas como los **bosques aleatorios (Random Forests)** o los **gradient boosting machines** superan muchas de estas limitaciones al combinar múltiples árboles y mejorar la capacidad de generalización.

#### Conclusiones sobre las SVM y sus alternativas  

Las SVM son herramientas extremadamente efectivas para problemas con datasets pequeños o medianos y con relaciones complejas entre las características, especialmente cuando se utilizan kernels no lineales como el RBF. Sin embargo, para problemas de gran escala, su alto costo computacional y consumo de memoria pueden hacer que otras alternativas sean más prácticas.  

La **regresión logística** se destaca en problemas lineales y en situaciones donde la interpretabilidad y la velocidad de entrenamiento son cruciales. Por otro lado, los **árboles de decisión** y sus variantes (como Random Forests o XGBoost) son más adecuados para problemas no lineales y de alta dimensionalidad, donde la interpretabilidad sigue siendo una prioridad.  

De todos modos, en última instancia, la elección del algoritmo va a depender de la naturaleza del problema, el tamaño del dataset y los recursos disponibles. Las SVM, aunque limitadas en escalabilidad, siguen siendo una herramienta poderosa cuando se aplican en el contexto adecuado. 

##### Para reflexionar...

> **¿Por qué crees que las SVM no son la mejor opción en problemas con millones de muestras?** 
> **Clave**: Reflexiona sobre la complejidad computacional del entrenamiento de SVM y el impacto de la matriz kernel en términos de memoria.

> **¿En qué escenarios elegirías un modelo como la regresión logística sobre las SVM?** 
> **Clave**: Considera la simplicidad del modelo, la rapidez del entrenamiento y la necesidad de probabilidades interpretables en problemas de clasificación lineal. 
