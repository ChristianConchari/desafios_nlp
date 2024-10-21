# Desafios - Procesamiento de lenguaje natural

En este repositorio se encuentran las soluciones para los desafios de la materia **Procesamiento de lenguaje natural** de la Especialización en Inteligencia Artificial (CEIA) de la Facultad de Ingeniería de la Universidad de Buenos Aires (FIUBA). 

## Desafío 1: Análisis de Similaridad entre Documentos

En este desafío, se realizó un análisis de similaridad de documentos utilizando el conjunto de datos **20 Newsgroups**. Se aplicaron las siguientes técnicas:

- **Vectorización y análisis de similaridad**: Los documentos fueron vectorizados utilizando **TF-IDF** para representar el contenido textual en un espacio de características numéricas. Posteriormente, se midió la **similaridad de coseno** entre un documento seleccionado y los 5 documentos más similares, buscando analizar si los documentos más similares pertenecen a la misma categoría.

- **Modelos de ML**: Se entrenaron modelos de **Naive Bayes** (Multinomial y ComplementNB) para realizar la tarea de clasificación de documentos. Para maximizar el rendimiento, se utilizó una búsqueda aleatoria de hiperparámetros, optimizando la métrica **f1-score macro** en el conjunto de datos de prueba.

- **Vectorización de palabras**: Se estudió la similaridad entre palabras, seleccionando 5 palabras y analizando las palabras más similares utilizando el mismo enfoque de vectorización.

El notebook con la solución se encuentra en [Desafío 1](desafio_1/solution_notebook.ipynb).

## Desafio 2: Word2Vec y Análisis de Similaridad entre Palabras

En este desafío, se trabajó con una muestra de reseñas de productos del conjunto de datos Fine Food Reviews para entrenar un modelo Word2Vec y analizar la similaridad entre palabras en el contexto de reseñas de productos alimenticios. Se trabajó en los siguiente:

- **Tokenización de Reseñas y Entrenamiento del Modelo Word2Vec**: Se tokenizaron las reseñas y se entrenó un modelo Word2Vec utilizando la librería Gensim.

- **Prueba de analogías**: Se realizaron pruebas de analogías para analizar la similaridad entre palabras en el contexto de reseñas de productos alimenticios y de esa manera, evaluar la calidad del modelo Word2Vec entrenado.

    - "service" fue identificado como similar a la combinación de "customer" y "happy", en contraste con "unhappy".

    - "refund" apareció como el concepto más cercano a la combinación de "support" y "return", lo cual tiene sentido en el contexto de atención al cliente.

    - "pricey" fue identificado como el concepto más cercano a la combinación de "product" y "expensive", en contraste con "cheap".

- **Análisis de Similaridad entre Palabras**: Se analizó la similaridad entre palabras seleccionadas y se identificaron las palabras más cercanas en el espacio vectorial de Word2Vec. Permitiendo capturar relaciones semánticas entre palabras.

- **Visualización de Embeddings**: Se visualizaron los embeddings de las palabras seleccionadas en un espacio de 3 dimensiones, aplicando previamente una [reducción de dimensionalidad con TSNE](desafio_2/utils.py).

    <img src="desafio_2/images/words_plot_1.png" alt="Desafio 2 Image 1" width="30%">

El notebook con la solución se encuentra en [Desafío 2](desafio_2/solution_notebook.ipynb).

## Desafío 3: Generación de Texto con LSTM basado en Shakespeare

En este desafío, se trabajó con una muestra de las obras de Shakespeare para entrenar un modelo LSTM de generación de texto. Se trabajó en lo siguiente:

- **Carga y Preparación de Datos**: Se cargaron los datos del archivo **shakespeare.txt** y se preprocesaron para segmentar el texto en palabras. Luego, se entrenó un `Tokenizer` para convertir las palabras en índices numéricos. Posteriormente, se dividió el conjunto de datos en entrenamiento y validación, y se aplicó padding para asegurar que todas las secuencias tuvieran la misma longitud.

- **Entrenamiento del Modelo LSTM:** Se definió y entrenó un modelo LSTM secuencial utilizando Keras. El modelo incluye una capa de Embedding con una dimensión de salida de 300, seguida por dos capas Bidirectional LSTM con 128 unidades cada una y capas Dropout para reducir el sobreajuste. Finalmente, se añadió una capa Dense con activación softmax para la clasificación. Se utilizó el optimizador Adam y se implementó EarlyStopping.

    <figure>
        <img src="desafio_3/images/training_plots.png" alt="Desafio 3 Entrenamiento" width="50%">
        <figcaption>
            (Se puede observar en una gráfica cómo el perplexity del modelo disminuye a medida que avanza el entrenamiento, lo cual indica una mejora en la capacidad del modelo para generar texto coherente).
        </figcaption>
    </figure>

- **Generación de Texto:** Tras el entrenamiento, se generaron nuevas secuencias de texto basadas en una "seed text". El modelo predijo palabras adicionales para completar las frases de manera mejorable pero con cierta coherencia.

El notebook con la solución se encuentra en [Desafío 3](desafio_3/solution_notebook.ipynb).