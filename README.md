# Clasificación de texto en 20 Newsgroups

Trabajos prácticos de 16.32 - Redes Neuronales en Bioingeniería. Todos usan el dataset **20 Newsgroups** (≈18.000 mensajes de foros clasificados en 20 categorías) y el mismo pipeline de preprocesamiento: tokenización, stopwords, stemming o lematización y vectorización TF-IDF.

## Contenido

| Carpeta | Trabajo | Modelo | Accuracy en test |
|---|---|---|---|
| [`TP1/`](TP1/) | Regresión Softmax | Una capa lineal en PyTorch, búsqueda de hiperparámetros con Optuna | 0.6568 |
| [`TP2/`](TP2/) | MLP | Perceptrón multicapa (1 a 3 capas ocultas), búsqueda con Optuna y pruning, experimento de overfitting y comparación con softmax | 0.6891 |
| `TP3/` | Regularización | MLP con L1/L2, Dropout y BatchNormalization | en proceso |

Cada carpeta tiene un notebook autocontenido, con las salidas de su ejecución guardadas.

## Cómo ejecutar

**En Google Colab (recomendado)**
1. Abrir el notebook desde Colab: *Archivo → Abrir notebook → GitHub* y pegar la URL de este repositorio.
2. Activar GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU*.
3. Ejecutar todas las celdas en orden (*Entorno de ejecución → Ejecutar todas*).

El dataset se descarga automáticamente con `sklearn.datasets.fetch_20newsgroups` y los recursos de NLTK se descargan dentro del notebook. No hace falta descargar nada a mano.

**En forma local**
```bash
pip install -r requirements.txt
jupyter notebook
```

Sin GPU los notebooks funcionan, pero el entrenamiento es más lento.

## Reproducibilidad

- Semilla fija (`SEED = 42`) para Python, NumPy, PyTorch y el sampler de Optuna.
- Partición train/validación estratificada (80/20) y fija, igual en todos los TPs.
- El test oficial de 20 Newsgroups se usa solo para la evaluación final. Está formado por mensajes posteriores a los de train, por eso los resultados en test son menores que en validación.
- Los tiempos de entrenamiento dependen del hardware y varían entre ejecuciones.
- Los experimentos se registran con MLflow (carpeta `mlruns/`, que no se sube al repositorio).

## Dependencias

Ver [`requirements.txt`](requirements.txt). Principales: PyTorch, scikit-learn, NLTK, Optuna, MLflow, pandas y matplotlib.
