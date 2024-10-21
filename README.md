# Desafios - Procesamiento de lenguaje natural

En este repositorio se encuentran las soluciones para los desafios de la materia **Procesamiento de lenguaje natural** de la Especialización en Inteligencia Artificial (CEIA) de la Facultad de Ingeniería de la Universidad de Buenos Aires (FIUBA). 

## Desafío 1: Análisis de Similaridad entre Documentos

En este desafío, se realizó un análisis de similaridad de documentos utilizando el conjunto de datos **20 Newsgroups**. Se aplicaron las siguientes técnicas:

- **Vectorización y análisis de similaridad**: Los documentos fueron vectorizados utilizando **TF-IDF** para representar el contenido textual en un espacio de características numéricas. Posteriormente, se midió la **similaridad de coseno** entre un documento seleccionado y los 5 documentos más similares, buscando analizar si los documentos más similares pertenecen a la misma categoría.

- **Modelos de ML**: Se entrenaron modelos de **Naive Bayes** (Multinomial y ComplementNB) para realizar la tarea de clasificación de documentos. Para maximizar el rendimiento, se utilizó una búsqueda aleatoria de hiperparámetros, optimizando la métrica **f1-score macro** en el conjunto de datos de prueba.

- **Vectorización de palabras**: Se estudió la similaridad entre palabras, seleccionando 5 palabras y analizando las palabras más similares utilizando el mismo enfoque de vectorización.

El notebook con la solución se encuentra en [Desafío 1](desafio_1/solution_notebook.ipynb).