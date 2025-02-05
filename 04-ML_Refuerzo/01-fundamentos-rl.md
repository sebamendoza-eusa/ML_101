# Tema 4. Sistemas de aprendizaje automático por refuerzo

## Fundamentos del Aprendizaje por Refuerzo

### Objetivos del módulo

> - Comprender qué es el aprendizaje por refuerzo y cómo se diferencia del aprendizaje supervisado y no supervisado.
> - Identificar los elementos clave de un problema de RL: agente, entorno, estados, acciones y recompensas.
> - Analizar el ciclo de interacción agente-entorno y el problema exploración-explotación.
> - Formalizar problemas de RL mediante **Procesos de Decisión de Markov (MDP)** y sus propiedades.
> - Estudiar la ecuación de Bellman y su papel en la estimación de funciones de valor.
> - Explorar métodos básicos para resolver MDPs: programación dinámica y Monte Carlo.

---

### **Introducción**

Un niño que aprende a andar en bicicleta no sigue una lista de instrucciones detalladas, sino que prueba diferentes movimientos y estrategias. Cae, se levanta y ajusta su equilibrio con cada intento. Si gira demasiado rápido, puede caer; si pedalea con estabilidad, avanza sin problemas. Con el tiempo, su cerebro asocia ciertas acciones con mejores resultados y ajusta su comportamiento en consecuencia. Este proceso intuitivo de prueba, error y ajuste puede equiparse al funcionamiento del **aprendizaje por refuerzo (Reinforcement Learning, RL)**.

Un definición muy sencilla del **aprendizaje por refuerzo** podría enunciarse así: 

> Paradigma de **aprendizaje automático** en el que un **agente** interactúa con un **entorno**, toma **acciones** y recibe **recompensas**, ajustando su estrategia para maximizar las ganancias a largo plazo.

Así pues, el aprendizaje por refuerzo es un marco matemático para la **toma de decisiones secuenciales**. Un **agente** explora su entorno, realiza acciones y recibe **recompensas** o **penalizaciones**, ajustando su comportamiento en función de la experiencia acumulada. Su objetivo es aprender una **política óptima**, es decir, una estrategia que maximice las recompensas a lo largo del tiempo.

Al final, todo este enfoque se fundamenta en la idea de **exploración y explotación**. El agente debe decidir entre explorar nuevas acciones para descubrir mejores resultados o explotar el conocimiento actual para obtener recompensas inmediatas.

> **Ejemplo 1** :
>  Un robot autónomo que navega en una fábrica puede moverse en distintas direcciones. Si choca con un obstáculo, recibe una penalización; si llega a su destino, obtiene una recompensa. Con el tiempo, aprende a evitar los obstáculos y elegir la ruta más eficiente.

> **Ejemplo 2**:
>  En un videojuego, un agente de RL aprende a jugar explorando distintas estrategias, ganando puntos por acciones exitosas y perdiéndolos por errores. No tiene instrucciones explícitas sobre cómo ganar, pero mejora su desempeño con la experiencia.

#### **Diferencias entre el aprendizaje por refuerzo y otros paradigmas de Machine Learning**

El aprendizaje por refuerzo (RL) es un paradigma distinto dentro del Machine Learning que se diferencia tanto del **aprendizaje supervisado** como del **aprendizaje no supervisado** en la forma en que el modelo adquiere conocimiento y toma decisiones. Mientras que en el aprendizaje supervisado los modelos aprenden a partir de datos etiquetados y en el no supervisado buscan estructuras ocultas en los datos, el aprendizaje por refuerzo aprende **mediante la interacción con un entorno**, ajustando su comportamiento en función de las recompensas obtenidas.

En efecto, en el **aprendizaje supervisado**, un modelo se entrena con un conjunto de datos etiquetados donde cada entrada tiene una salida esperada. El objetivo es minimizar el error de predicción comparando los resultados del modelo con las etiquetas verdaderas a través de la minimización de una *función de pérdida*. En contraste, en **RL no hay un conjunto de datos fijos ni etiquetas predefinidas**. En su lugar, un **agente** interactúa con el entorno y aprende a tomar decisiones **basándose en la recompensa obtenida a lo largo del tiempo**. En vez de ajustar pesos para minimizar una función de pérdida, el agente busca maximizar el retorno acumulado en un horizonte temporal.

> **Ejemplo**:
>  En un sistema de reconocimiento de voz basado en aprendizaje supervisado, el modelo aprende a transcribir audio a texto a partir de ejemplos etiquetados. En cambio, en un asistente virtual que usa RL, el sistema ajusta sus respuestas en función de cómo el usuario interactúa, optimizando la experiencia según el nivel de satisfacción detectado.

La diferencia clave es la **retroalimentación**. En aprendizaje supervisado, el modelo recibe correcciones inmediatas (errores entre predicciones y etiquetas), mientras que en el RL, las recompensas pueden estar **diferidas en el tiempo**, lo que implica que el agente debe aprender **estrategias a largo plazo**.

Por otra parte, ya sabemos que el **aprendizaje no supervisado** se fundamenta en la detección de patrones sin necesidad de aportar etiquetas. En este escenario los modelos buscan estructuras ocultas en los datos. Pero a diferencia del aprendizaje no supervisado, en RL **no se busca estructurar datos, sino aprender una política de acción**. El agente explora su entorno y ajusta su comportamiento en función de la recompensa acumulada, mientras que un algoritmo no supervisado solo analiza relaciones entre datos sin actuar sobre ellos.

> **Ejemplo**:
>  Un modelo no supervisado puede analizar el comportamiento de los clientes en un e-commerce y agruparlos según patrones de compra. Sin embargo, un sistema basado en RL optimiza la **estrategia de recomendación de productos**, aprendiendo a mostrar ofertas personalizadas en función de la interacción del usuario para maximizar las conversiones.

En la siguiente tabla se resumen las diferencias clave entre los tres enfoques de machine learning vistos hasta ahora en el curso

| Característica           | Aprendizaje supervisado       | Aprendizaje no supervisado       | Aprendizaje por refuerzo                    |
| ------------------------ | ----------------------------- | -------------------------------- | ------------------------------------------- |
| Datos de entrenamiento   | Etiquetados                   | Sin etiquetas                    | Generados por interacción                   |
| Objetivo del aprendizaje | Minimizar error de predicción | Encontrar estructuras ocultas    | Maximizar recompensa acumulada              |
| Retroalimentación        | Corrección inmediata          | No hay retroalimentación directa | Recompensa diferida                         |
| Ejemplo típico           | Clasificación de imágenes     | Clustering de clientes           | Un agente que aprende a jugar un videojuego |

##### Para reflexionar...

> **¿En qué situaciones el aprendizaje por refuerzo puede ser más ventajoso que los enfoques supervisados o no supervisados?**
>  **Clave**: Piensa en problemas donde la mejor estrategia se aprende a través de prueba y error, sin datos etiquetados previamente.

#### **Aplicaciones y casos de uso**

El aprendizaje por refuerzo ha demostrado su potencial en múltiples sectores, donde la toma de decisiones secuenciales es clave para mejorar el rendimiento y la eficiencia. Su capacidad para aprender estrategias óptimas a partir de la interacción con un entorno lo hace especialmente útil en escenarios donde las reglas no están completamente definidas o donde la exploración es fundamental para el éxito.

Uno de los ámbitos donde RL ha tenido un impacto significativo es la **robótica**. Los robots autónomos aprenden a manipular objetos, navegar en entornos desconocidos y adaptarse a situaciones cambiantes. En lugar de programar manualmente cada movimiento, un agente de RL puede explorar distintas estrategias hasta encontrar la mejor manera de alcanzar su objetivo, como un brazo robótico que aprende a ensamblar piezas sin necesidad de instrucciones precisas.

En el mundo de los **juegos y el entretenimiento**, RL ha logrado hitos sorprendentes. Algoritmos como AlphaGo han superado a los mejores jugadores humanos en juegos complejos como *Go* y *ajedrez*, descubriendo estrategias innovadoras que antes no se habían considerado. Además, en los videojuegos, agentes de RL han aprendido a jugar títulos de estrategia en tiempo real sin instrucciones explícitas, simplemente experimentando y ajustando su comportamiento en función de las recompensas obtenidas.

En el ámbito de la **optimización de procesos**, muchas empresas utilizan RL para mejorar la eficiencia en logística y cadena de suministro. Por ejemplo, en la planificación de rutas de entrega, un agente puede aprender a minimizar los tiempos de transporte y reducir costos operativos explorando diferentes combinaciones de rutas y evaluando su impacto en el rendimiento general del sistema.

En el sector **financiero**, RL se ha aplicado con éxito en la creación de estrategias de inversión automatizadas. Dado que los mercados financieros son dinámicos e impredecibles, los agentes de RL pueden aprender a ajustar sus decisiones en función de la evolución de los datos, optimizando la asignación de recursos para maximizar el retorno de inversión.

Por último, en el área de **salud**, el aprendizaje por refuerzo se ha utilizado para personalizar tratamientos médicos y optimizar terapias. Un modelo de RL puede aprender cuál es la mejor secuencia de decisiones para ajustar la dosis de un medicamento o definir un plan de tratamiento adaptado a cada paciente, maximizando la efectividad y minimizando los efectos adversos.

En todos estos casos, el aprendizaje por refuerzo permite que los sistemas evolucionen a partir de la experiencia, descubriendo soluciones óptimas en entornos dinámicos donde las reglas no siempre están claras desde el principio.

##### Para reflexionar...

> **¿Qué características comparten los problemas donde el aprendizaje por refuerzo ha demostrado ser más efectivo?**
>  **Clave**: Analiza cómo la interacción con el entorno y la optimización de largo plazo influyen en la toma de decisiones.

















1. **Elementos fundamentales de RL**
   - Definición del **entorno, agente, acciones, recompensas y estados**.
   - Ciclo de interacción agente-entorno.
   - Balance entre **exploración y explotación**.
2. **Formalización matemática del RL**
   - **Procesos de Decisión de Markov (MDP)**: Definición y propiedades.
   - Funciones de valor y ecuación de Bellman.
   - Métodos de solución de MDP: **Programación dinámica y Monte Carlo**.

------

