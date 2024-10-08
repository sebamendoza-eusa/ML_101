# Tema 1. Determinación de sistemas de aprendizaje automático (*Machine Learning*). Modelos de machine learning

## Contenidos

> 1. Definición de aprendizaje automático
> 2. Componentes del ML
> 3. Etapas en un proyecto ML
> 4. **Tipos de ML**
>

---

## 4. Tipos de ML 

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
