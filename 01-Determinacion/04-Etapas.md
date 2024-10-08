# Tema 1. Determinación de sistemas de aprendizaje automático (*Machine Learning*). Modelos de machine learning

## Contenidos

> 1. Definición de aprendizaje automático
> 2. Componentes del ML
> 3. **Etapas en un proyecto ML**
> 4. Tipos de ML
>

---

## 3. Etapas en un proyecto ML

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


