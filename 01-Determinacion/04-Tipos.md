# Tema 1. Determinación de sistemas de aprendizaje automático (*Machine Learning*). Modelos de machine learning

## Contenidos

> 1. Definición de aprendizaje automático
> 2. Componentes del ML
> 3. Etapas en un proyecto ML
> 4. Tipos de ML
>

## Temporalización

1. **Definición de aprendizaje automático** (2 horas): Explicación básica y concepto de ML.
2. **Componentes del ML** (4 horas): Explicación detallada de los elementos clave como datos, algoritmos y evaluaciones.
3. **Etapas en un proyecto ML** (6 horas): Exploración de las fases de un proyecto real, desde la recolección de datos hasta la validación.
4. **Tipos de ML** (4 horas): Comparación y análisis de los tipos de aprendizaje (supervisado, no supervisado, por refuerzo).

---

## 1. Definición de aprendizaje automático (2 horas)

### Definición de aprendizaje automático

El **aprendizaje automático (ML)** es una rama de la inteligencia artificial que permite a las máquinas aprender a partir de datos, mejorando su rendimiento en tareas específicas sin una programación explícita para cada caso. En lugar de seguir instrucciones predefinidas, los sistemas de ML utilizan algoritmos para identificar patrones en los datos y realizar predicciones o tomar decisiones informadas. El objetivo es desarrollar modelos que puedan generalizar a partir de datos conocidos y aplicarlos a nuevas situaciones.

#### Definiciones clave

- **Tom Mitchell (1997)** define el aprendizaje automático como: "Un programa de computadora aprende de la experiencia E respecto a una tarea T y una medida de rendimiento P, si su rendimiento en T, medido por P, mejora con la experiencia E".
  
  Aquí, **T** es la tarea que se busca optimizar (como clasificación o predicción), **E** son los datos de los que se aprende, y **P** es la métrica que mide el éxito.

- **Arthur Samuel (1959)** define el aprendizaje automático como una capacidad para que las computadoras aprendan sin ser programadas explícitamente, subrayando el poder de los algoritmos para **aprender automáticamente de los datos**.

- Según el enfoque de **Russell y Norvig** en **AIMA 4th ed.**, el ML es una herramienta clave dentro de la inteligencia artificial para sistemas que requieren **adaptabilidad** y **aprendizaje continuo** a partir de experiencias pasadas.

#### Aplicaciones y ejemplos prácticos

El aprendizaje automático es ampliamente utilizado en sectores conocidos, pero también hay ejemplos en áreas menos tradicionales:

> **Ejemplo 1: Agricultura de precisión** 
>
> En la agricultura moderna, los sistemas de ML se utilizan para **predecir las cosechas** en función de datos históricos, climáticos y de salud del suelo. Esto ayuda a los agricultores a optimizar sus decisiones sobre riego, fertilización y recolección.

> **Ejemplo 2: Prevención de la caza furtiva en reservas naturales**  
>
> En algunas reservas naturales, los algoritmos de ML se emplean para predecir **patrones de caza furtiva**, analizando datos de patrullas anteriores, movimientos de animales y actividades sospechosas. Esta información permite planificar patrullas de manera más eficiente.

> **Ejemplo 3: Conservación del patrimonio cultural**
>
> En museos y sitios arqueológicos, el ML se utiliza para analizar imágenes de piezas históricas y detectar **daños estructurales** que podrían pasar desapercibidos a simple vista. Algoritmos entrenados en modelos de deterioro detectan microfracturas y alertan a los conservadores.

> **Ejemplo 4: Gestión de residuos **
>
> **En sistemas de gestión inteligente de residuos, el ML se utiliza para **optimizar rutas** de recolección de basura. Los algoritmos aprenden de datos históricos de recogida, tráfico y comportamiento de los ciudadanos para hacer más eficientes las operaciones de recolección.

#### Características del ML

El aprendizaje automático tiene varias características clave:

- **Generalización**: La capacidad del modelo de aprender de datos específicos y aplicar este conocimiento a nuevos casos. Un buen modelo no solo memoriza los datos, sino que **generaliza** los patrones que detecta.

- **Automatización**: A diferencia de los sistemas programados manualmente, el ML adapta sus reglas y mejora sin intervención humana directa.

- **Adaptabilidad**: Los algoritmos de ML pueden ajustar su comportamiento frente a **cambios en los datos** o el entorno, mejorando continuamente con el tiempo.

##### Para reflexionar…

> **¿Qué papel juega la cantidad y calidad de los datos en el aprendizaje automático?**
>
> **Pista**: Considera cómo los datos de baja calidad o insuficientes pueden limitar la capacidad del modelo para generalizar correctamente. ¿Es siempre mejor tener más datos, o también importa la representatividad de estos?

> **¿Cómo afecta la sobrecarga de información a un modelo de aprendizaje automático?**
>
> **Pista**: Piensa en cómo un exceso de información no relevante podría confundir al modelo o hacerlo ineficiente. ¿Es posible mejorar los modelos eliminando datos irrelevantes?

##### A debate…

> **¿Puede el aprendizaje automático reemplazar la toma de decisiones humanas en todos los sectores?**  
>
> **Pista**: Reflexiona sobre sectores donde las decisiones requieren **creatividad**, **empatía** o **intuición**, y considera si los modelos de ML, basados solo en datos, son adecuados para reemplazar estas cualidades humanas.

> **¿Es ético utilizar sistemas de aprendizaje automático en la predicción de comportamiento humano, como en sistemas judiciales o de vigilancia?**
>
> **Pista**: Piensa en el uso de datos para predecir delitos o comportamientos. ¿Es justo que las máquinas tomen decisiones basadas en probabilidades? ¿Qué riesgos éticos existen al depender de algoritmos para estos fines?



## 2. Componentes del ML (4 horas)

### Componentes del ML (4 horas)

Los sistemas de aprendizaje automático (ML) están formados por varios **elementos clave** que, en conjunto, permiten el desarrollo, ajuste y validación de los modelos. Cada componente es esencial para lograr un modelo eficiente y preciso. A continuación, exploramos en detalle los elementos principales.

El éxito de un proyecto de aprendizaje automático va a depender de la integración adecuada de cada componente. Los datos deben estar bien representados, los algoritmos correctamente seleccionados y los modelos deben ser evaluados rigurosamente. La capacidad de generalización de un modelo es clave para su éxito en el mundo real, lo que exige un delicado equilibrio entre el ajuste a los datos de entrenamiento y la capacidad de responder bien a datos nuevos.

### Datos

El **conjunto de datos** es la base sobre la que se construyen los modelos de ML. Estos datos pueden ser numéricos, categóricos, imágenes, texto o señales de audio, y deben ser **representativos y de calidad** para evitar sesgos y mejorar la capacidad del modelo para generalizar a nuevos datos.

Los **datos se dividen** en:
- **Entrenamiento**: Se usa para ajustar los parámetros del modelo.
- **Validación**: Evalúa el rendimiento del modelo durante el entrenamiento para evitar sobreajuste.
- **Test**: Prueba final del modelo con datos no vistos anteriormente.

> **Ejemplo**: En la predicción de **rendimientos agrícolas**, los datos históricos del clima y el uso de fertilizantes se utilizan para entrenar un modelo capaz de predecir la cantidad de producción de cultivos en diferentes condiciones climáticas.

#### Preprocesamiento de los datos

Antes de utilizar los datos, es crucial realizar un **preprocesamiento** para asegurar su calidad. Esto incluye:
- **Limpieza de datos**: Eliminación de valores perdidos o incorrectos.
- **Normalización**: Escalado de datos para que diferentes variables estén en la misma escala.
- **Codificación de categorías**: Conversión de datos categóricos en numéricos.

> **Ejemplo**: En la detección de fallos en maquinarias industriales, los datos de sensores pueden tener valores fuera de rango o registros incompletos. El preprocesamiento garantizaría que estos valores erróneos no afecten el modelo de predicción de fallos.

### Algoritmos

Un **algoritmo** de aprendizaje automático es el conjunto de reglas e instrucciones que permiten al modelo **aprender** de los datos. La elección del algoritmo depende del tipo de problema y los datos. Los principales tipos de algoritmos incluyen:

- **Regresión**: Se utiliza para predecir valores continuos (como los precios de viviendas).
- **Clasificación**: Asigna categorías a nuevas instancias (como el filtrado de spam).
- **Clustering**: Agrupa instancias similares sin etiquetas previas (como la segmentación de clientes).

#### Tipos de algoritmos

1. **Algoritmos supervisados**: El modelo aprende a partir de datos etiquetados (donde la salida es conocida).
2. **Algoritmos no supervisados**: El modelo busca patrones en datos sin etiquetas.
3. **Algoritmos por refuerzo**: El modelo aprende mediante prueba y error, recibiendo recompensas o penalizaciones.

> **Ejemplo**: Un algoritmo **de clustering** se puede utilizar en el campo del marketing para agrupar clientes según su comportamiento de compra y definir estrategias personalizadas.

##### Para reflexionar…

> **¿Cómo afecta la elección del algoritmo al rendimiento del modelo de aprendizaje automático?**
>
> **Pistas**: Considera cómo diferentes algoritmos pueden ser más adecuados según la naturaleza de los datos (estructurados o no estructurados) y la tarea (clasificación o regresión).

### Modelo

Un **modelo** es el resultado del entrenamiento de un algoritmo con un conjunto de datos. Es la representación matemática que transforma las **entradas** en **salidas**. Durante el entrenamiento, el modelo ajusta sus **parámetros** para minimizar el error en las predicciones.

Los **modelos de ML** pueden variar en complejidad:
- **Modelos lineales**: Representan la relación entre las variables de entrada y salida como una línea recta.
- **Modelos complejos**: Como las redes neuronales, que pueden modelar relaciones no lineales y son más adecuados para problemas complejos.

> **Ejemplo**: En la predicción de **fallos en equipos eléctricos**, un modelo de regresión lineal simple puede predecir el tiempo restante hasta el fallo en función del desgaste observado.

### Evaluación del modelo

La **evaluación** es una parte fundamental del desarrollo de un modelo. Permite comprobar el rendimiento y la capacidad de **generalización** del modelo a nuevos datos. Las principales métricas de evaluación incluyen:

- **Precisión**: Proporción de predicciones correctas sobre el total de predicciones realizadas.
- **Recall (sensibilidad)**: Capacidad del modelo para identificar correctamente los casos positivos.
- **F1-Score**: Combina precisión y recall en una sola métrica para problemas desbalanceados.
- **Área bajo la curva ROC**: Mide el desempeño de un clasificador en diferentes umbrales de decisión.

> **Ejemplo**: En sistemas de **detección de fraudes bancarios**, la evaluación del modelo se centra no solo en la precisión, sino también en su capacidad para minimizar los **falsos negativos**, es decir, transacciones fraudulentas no detectadas.

##### A debate…

> **¿Debe primar la precisión de un modelo o su capacidad de generalización?**
>
> **Pistas**: Reflexiona sobre los casos donde un modelo altamente preciso en los datos de entrenamiento puede fallar con nuevos datos, indicando un problema de **sobreajuste**.
>
> ##### Para reflexionar…
>

> **¿Qué sucede cuando un modelo de ML generaliza mal en nuevos datos, aunque su precisión en los datos de entrenamiento es alta?**
>
> **Pistas**: El problema de **sobreajuste** puede surgir cuando un modelo se ajusta demasiado a los detalles y ruido de los datos de entrenamiento.



## 3. Etapas en un proyecto ML (6 horas)

El desarrollo de un proyecto de **aprendizaje automático (ML)** sigue una serie de fases cruciales, cada una contribuyendo al éxito del modelo y su implementación final.

### Recolección de datos

La primera etapa en cualquier proyecto de ML es la **recolección de datos**. Estos datos pueden provenir de bases de datos internas, sensores, dispositivos conectados o interacciones de usuarios. La calidad y cantidad de los datos son factores clave, y la diversidad de los datos es esencial para capturar la complejidad del problema.

> **Ejemplo**: En el caso de una aplicación que predice **condiciones del tráfico**, los datos pueden provenir de sensores en las carreteras, reportes meteorológicos y datos de tráfico históricos en tiempo real.

### Preprocesamiento de datos

El **preprocesamiento de datos** incluye tareas como la limpieza de los datos, la eliminación de duplicados o valores faltantes, la transformación de las variables y la codificación de variables categóricas. Estas acciones garantizan que los datos sean adecuados para el modelo y aumentan la precisión de las predicciones.

> **Ejemplo**: En un proyecto de **predicción de tiempos de entrega** para una compañía de logística, se deben limpiar los registros de envíos mal etiquetados y unificar las unidades de medida de tiempo para que todos los datos sean consistentes.

### División de los datos

Una división adecuada de los datos en **entrenamiento**, **validación** y **prueba** es crucial para asegurar que el modelo pueda generalizar correctamente. El conjunto de prueba no debe ser visto por el modelo durante el entrenamiento para evitar sobreajuste.

> **Ejemplo**: En un sistema que predice el **riesgo crediticio** de los solicitantes de préstamos, dividir los datos históricos de préstamos aprobados y rechazados asegura que el modelo pueda aprender patrones sin depender de una partición específica de los datos.

### Selección de algoritmos

La **elección del algoritmo** depende de la naturaleza del problema. Algunos algoritmos son más adecuados para clasificación, otros para regresión o clustering. Es fundamental evaluar varios algoritmos y seleccionar el que mejor se ajuste al problema.

> **Ejemplo**: En un proyecto de **reconocimiento de patrones en el comportamiento del consumidor**, se podrían utilizar algoritmos de clustering como **k-means** para agrupar a los clientes con base en características comunes.

### Entrenamiento del modelo

El **entrenamiento** es el proceso mediante el cual el algoritmo aprende de los datos de entrenamiento, ajustando sus parámetros internos para minimizar el error. El ajuste de los hiperparámetros también es importante en esta fase.

> **Ejemplo**: En un sistema de predicción de **fallos en maquinaria industrial**, el modelo entrenado utiliza datos de sensores en la maquinaria para identificar patrones que predicen fallos antes de que ocurran.

### Evaluación y validación

Una vez entrenado el modelo, es esencial **evaluar su rendimiento** en los datos de prueba. Se utilizan métricas como precisión, sensibilidad, especificidad o la curva ROC para medir su capacidad de generalización y asegurar que el modelo no haya sobreajustado los datos.

> **Ejemplo**: En un sistema de **diagnóstico médico**, se evalúan diferentes métricas como el **área bajo la curva ROC** para asegurar que el modelo pueda detectar correctamente enfermedades a partir de los datos de los pacientes.

### Puesta en producción

Después de entrenar y validar el modelo, la siguiente etapa es su **puesta en producción**. Aquí es donde el modelo se implementa en un entorno real, donde debe manejar datos nuevos y tomar decisiones automatizadas o apoyar la toma de decisiones.

Esta fase implica **integración con sistemas existentes**, **optimización de latencia** y asegurarse de que el modelo funcione en entornos de producción con grandes volúmenes de datos.

> **Ejemplo**: Un modelo para **gestionar la eficiencia energética en edificios** se pondría en producción mediante la integración con los sistemas de control de clima y energía de los edificios, ajustando el consumo energético en tiempo real.

### Mantenimiento y ciclo de vida del modelo

Una vez que el modelo está en producción, es crucial realizar un **mantenimiento continuo**. Los datos y los entornos cambian con el tiempo, lo que puede provocar la degradación del rendimiento del modelo.

Es necesario **monitorear** el comportamiento del modelo en producción, realizar actualizaciones y, en algunos casos, volver a entrenarlo con nuevos datos. Esta fase también incluye la implementación de sistemas de **alertas** en caso de fallos o disminución del rendimiento.

**Ejemplo**: En un sistema de **recomendación de productos**, si los patrones de comportamiento del consumidor cambian debido a nuevas tendencias, el modelo debe actualizarse continuamente para seguir siendo relevante y efectivo.

### Para reflexionar...

### Etapas en un proyecto ML (6 horas)

El desarrollo de un proyecto de **aprendizaje automático (ML)** sigue una serie de fases cruciales, cada una contribuyendo al éxito del modelo y su implementación final. En esta ampliación, se introduce también la **puesta en producción** del modelo y su **mantenimiento** durante su ciclo de vida, aspectos fundamentales que garantizan la utilidad y longevidad de un proyecto ML.

### Recolección de datos

La primera etapa en cualquier proyecto de ML es la **recolección de datos**. Estos datos pueden provenir de bases de datos internas, sensores, dispositivos conectados o interacciones de usuarios. La calidad y cantidad de los datos son factores clave, y la diversidad de los datos es esencial para capturar la complejidad del problema.

**Ejemplo**: En el caso de una aplicación que predice **condiciones del tráfico**, los datos pueden provenir de sensores en las carreteras, reportes meteorológicos y datos de tráfico históricos en tiempo real.

### Preprocesamiento de datos

El **preprocesamiento de datos** incluye tareas como la limpieza de los datos, la eliminación de duplicados o valores faltantes, la transformación de las variables y la codificación de variables categóricas. Estas acciones garantizan que los datos sean adecuados para el modelo y aumentan la precisión de las predicciones.

**Ejemplo**: En un proyecto de **predicción de tiempos de entrega** para una compañía de logística, se deben limpiar los registros de envíos mal etiquetados y unificar las unidades de medida de tiempo para que todos los datos sean consistentes.

### División de los datos

Una división adecuada de los datos en **entrenamiento**, **validación** y **prueba** es crucial para asegurar que el modelo pueda generalizar correctamente. El conjunto de prueba no debe ser visto por el modelo durante el entrenamiento para evitar sobreajuste.

**Ejemplo**: En un sistema que predice el **riesgo crediticio** de los solicitantes de préstamos, dividir los datos históricos de préstamos aprobados y rechazados asegura que el modelo pueda aprender patrones sin depender de una partición específica de los datos.

### Selección de algoritmos

La **elección del algoritmo** depende de la naturaleza del problema. Algunos algoritmos son más adecuados para clasificación, otros para regresión o clustering. Es fundamental evaluar varios algoritmos y seleccionar el que mejor se ajuste al problema.

**Ejemplo**: En un proyecto de **reconocimiento de patrones en el comportamiento del consumidor**, se podrían utilizar algoritmos de clustering como **k-means** para agrupar a los clientes con base en características comunes.

### Entrenamiento del modelo

El **entrenamiento** es el proceso mediante el cual el algoritmo aprende de los datos de entrenamiento, ajustando sus parámetros internos para minimizar el error. El ajuste de los hiperparámetros también es importante en esta fase.

**Ejemplo**: En un sistema de predicción de **fallos en maquinaria industrial**, el modelo entrenado utiliza datos de sensores en la maquinaria para identificar patrones que predicen fallos antes de que ocurran.

### Evaluación y validación

Una vez entrenado el modelo, es esencial **evaluar su rendimiento** en los datos de prueba. Se utilizan métricas como precisión, sensibilidad, especificidad o la curva ROC para medir su capacidad de generalización y asegurar que el modelo no haya sobreajustado los datos.

**Ejemplo**: En un sistema de **diagnóstico médico**, se evalúan diferentes métricas como el **área bajo la curva ROC** para asegurar que el modelo pueda detectar correctamente enfermedades a partir de los datos de los pacientes.

### Puesta en producción

Después de entrenar y validar el modelo, la siguiente etapa es su **puesta en producción**. Aquí es donde el modelo se implementa en un entorno real, donde debe manejar datos nuevos y tomar decisiones automatizadas o apoyar la toma de decisiones. Este paso implica **integración con sistemas existentes**, **optimización de latencia** y asegurarse de que el modelo funcione en entornos de producción con grandes volúmenes de datos.

**Ejemplo**: Un modelo para **gestionar la eficiencia energética en edificios** se pondría en producción mediante la integración con los sistemas de control de clima y energía de los edificios, ajustando el consumo energético en tiempo real.

### Mantenimiento y ciclo de vida del modelo

Una vez que el modelo está en producción, es crucial realizar un **mantenimiento continuo**. Los datos y los entornos cambian con el tiempo, lo que puede provocar la degradación del rendimiento del modelo. Es necesario monitorear el comportamiento del modelo en producción, realizar actualizaciones y, en algunos casos, volver a entrenarlo con nuevos datos. Esta fase también incluye la implementación de sistemas de **alertas** en caso de fallos o disminución del rendimiento.

**Ejemplo**: En un sistema de **recomendación de productos**, si los patrones de comportamiento del consumidor cambian debido a nuevas tendencias, el modelo debe actualizarse continuamente para seguir siendo relevante y efectivo.

##### Para reflexionar...

> ##### ¿Qué riesgos implica implementar un modelo en producción sin monitoreo adecuado?
> **Pistas**: Considera cómo los cambios en los datos pueden degradar el rendimiento del modelo con el tiempo. Piensa en problemas como la **deriva de datos** o la aparición de nuevos patrones que no se reflejan en el entrenamiento inicial.

> ##### ¿Es siempre necesario reentrenar un modelo en producción cuando su rendimiento comienza a disminuir?
> **Pistas**: Reflexiona sobre cómo pequeñas adaptaciones o ajustes en los hiperparámetros, junto con la recolección de nuevos datos, pueden mejorar el rendimiento sin necesidad de un reentrenamiento completo.

##### A debate...

> **¿Debe existir un equipo dedicado al monitoreo y mantenimiento del modelo después de la puesta en producción?**
>
> **Clave1**: Considera los riesgos y costes de mantener un modelo en producción y la necesidad de un equipo que gestione el ciclo de vida completo del modelo.
>
> **Clave2**: Considera cómo los cambios en los datos pueden degradar el rendimiento del modelo con el tiempo. Piensa en problemas como la **deriva de datos** o la aparición de nuevos patrones que no se reflejan en el entrenamiento inicial.

> **¿Es siempre necesario reentrenar un modelo en producción cuando su rendimiento comienza a disminuir?**
>
> **Clave**: Reflexiona sobre cómo pequeñas adaptaciones o ajustes en los hiperparámetros, junto con la recolección de nuevos datos, pueden mejorar el rendimiento sin necesidad de un reentrenamiento completo.



## 4. Tipos de ML (4 horas)

### Aprendizaje supervisado

El **aprendizaje supervisado** es un enfoque donde el modelo aprende de ejemplos etiquetados. Se le proporciona un conjunto de datos donde las entradas están asociadas con salidas correctas, permitiendo que el modelo aprenda a hacer predicciones para nuevas entradas.

**Ejemplo**: En la **predicción de rendimiento energético** de edificios inteligentes, un modelo de aprendizaje supervisado se puede entrenar con datos históricos sobre consumo de energía, temperaturas externas e internas, y otras variables relacionadas. El sistema puede aprender a predecir el consumo futuro y optimizar el uso de energía ajustando parámetros como la climatización o la iluminación, mejorando la eficiencia y reduciendo costos operativos.

##### Para reflexionar...

> **¿Cuáles son las principales limitaciones del aprendizaje supervisado en proyectos de Machine Learning?**
>
> **Pistas**: Reflexiona sobre la necesidad de grandes cantidades de datos etiquetados y los costos asociados a obtenerlos. Piensa en problemas donde las etiquetas correctas no están fácilmente disponibles.

### Aprendizaje no supervisado

En el **aprendizaje no supervisado**, el modelo debe encontrar patrones ocultos en datos que no tienen etiquetas. Este enfoque se usa para tareas como **clustering** o reducción de dimensionalidad.

**Ejemplo**: En el análisis de clientes de un supermercado, se puede usar el clustering para identificar grupos de compradores con patrones de comportamiento similares sin saber previamente a qué categoría pertenece cada uno. Este análisis puede ayudar a la empresa a personalizar campañas de marketing o ajustar la distribución de productos en las tiendas.

##### Para reflexionar...

> **¿Cuáles son las ventajas de aplicar aprendizaje no supervisado en entornos donde no es posible etiquetar los datos manualmente?**
>
> **Pistas**: Considera cómo este enfoque permite descubrir patrones ocultos y su utilidad en aplicaciones como análisis de comportamiento de clientes o redes sociales sin la necesidad de intervención humana para etiquetar datos.

### Aprendizaje por refuerzo

El **aprendizaje por refuerzo** involucra a un agente que aprende a tomar decisiones en un entorno, maximizando recompensas y minimizando penalizaciones a través de la interacción continua.

**Ejemplo**: En la robótica autónoma, se usa aprendizaje por refuerzo para que robots aprendan a moverse en un entorno sin instrucciones específicas. Un robot de limpieza puede aprender a evitar obstáculos, como muebles, y mejorar sus rutas de limpieza para reducir el tiempo necesario, maximizando su eficiencia a medida que gana experiencia.

##### Para reflexionar...

> **¿Qué limitaciones puede tener el aprendizaje por refuerzo en entornos complejos y dinámicos?**
>
> **Pistas**: Reflexiona sobre los retos de encontrar el equilibrio entre exploración y explotación, y cómo el entorno puede cambiar rápidamente, requiriendo que el agente se adapte constantemente.

### Redes neuronales

Las **redes neuronales** son un tipo especial de modelo dentro del aprendizaje automático, inspiradas en el funcionamiento del cerebro humano. Se componen de nodos o "neuronas" conectadas en capas (entrada, ocultas y salida). Estas redes son particularmente efectivas en tareas como reconocimiento de imágenes, procesamiento del lenguaje natural y clasificación de grandes volúmenes de datos. Mediante técnicas como **deep learning**, las redes neuronales pueden aprender representaciones complejas a partir de grandes cantidades de datos no estructurados.

**Ejemplo:** En la industria agrícola, redes neuronales pueden ser utilizadas para el análisis de imágenes satelitales, detectando signos tempranos de enfermedades en cultivos o identificando áreas que requieren riego, optimizando el uso de recursos naturales y mejorando el rendimiento de las cosechas.

##### Para reflexionar...

> **¿Qué ventajas ofrecen las redes neuronales frente a otros modelos de machine learning para el procesamiento de grandes volúmenes de datos no estructurados?**
>
> **Pistas**: Considera cómo las redes neuronales pueden aprender representaciones complejas de los datos y su eficacia en tareas como el reconocimiento de patrones en imágenes o texto.

##### A debate...

> **¿Es ético implementar sistemas de aprendizaje por refuerzo en contextos de toma de decisiones críticas como el ámbito legal o médico?**
>
> **Pistas**: Reflexiona sobre los riesgos de que un sistema aprenda a través de ensayo y error en decisiones que podrían afectar la vida de las personas, como en diagnósticos médicos o sentencias judiciales, y si sería necesario establecer límites o supervisión humana constante.
