# Redes Neuronales - Trabajos prácticos

Trabajos prácticos de Redes Neuronales / Deep Learning. Los TP1 a TP3 clasifican texto (**20 Newsgroups**, ≈18.000 mensajes de foros en 20 categorías) con un pipeline de preprocesamiento y vectorización TF-IDF. El TP4 clasifica imágenes de lesiones de piel (**ISIC**, 9 clases) con redes convolucionales.

## Contenido

| Carpeta | Trabajo | Modelo | Accuracy en test |
|---|---|---|---|
| [`TP1/`](TP1/) | Regresión Softmax | Una capa lineal en PyTorch, búsqueda de hiperparámetros con Optuna | 0.6568 |
| [`TP2/`](TP2/) | MLP | Perceptrón multicapa (1 a 3 capas ocultas), búsqueda con Optuna y pruning, experimento de overfitting y comparación con softmax con el mismo presupuesto | 0.6642 |
| [`TP3/`](TP3/) | Regularización | MLP con L1/L2, Dropout y BatchNormalization: experimentos controlados y búsqueda con Optuna | 0.6925 |
| [`TP4/`](TP4/) | Redes convolucionales | CNN para lesiones de piel: EDA, duplicados, MLP vs CNN, arquitectura y data augmentation | 0.646 ± 0.037 |

Cada carpeta tiene un notebook autocontenido, con las salidas de su ejecución guardadas. Las accuracies de los TP1 a TP3 son de una corrida, y la del TP4 es la media ± desvío de 5 semillas.

## Cómo ejecutar

**En Google Colab (recomendado)**
1. Abrir el notebook desde Colab: *Archivo → Abrir notebook → GitHub* y pegar la URL de este repositorio.
2. Activar GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU*.
3. Ejecutar todas las celdas en orden (*Entorno de ejecución → Ejecutar todas*).

**En forma local**
```bash
pip install -r requirements.txt
jupyter notebook
```

Sin GPU los notebooks funcionan, pero el entrenamiento es más lento.

## Datos

- **TP1 a TP3:** el dataset se descarga automáticamente con `sklearn.datasets.fetch_20newsgroups`, y los recursos de NLTK se descargan dentro del notebook.
- **TP4:** el dataset **no está en este repositorio**. Se obtiene del repositorio de la cátedra, https://github.com/cselmo/CNN-en-Skin-Dataset (carpeta `data/Split_smol`). En Colab, la primera celda del notebook lo clona y se ubica en esa carpeta. En forma local, hay que clonarlo y copiar la carpeta `data/` junto al notebook del TP4.

## Reproducibilidad

- Semillas fijas para Python, NumPy, PyTorch y el sampler de Optuna.
- TP1 a TP3: partición train/validación estratificada (80/20) y fija. El test oficial de 20 Newsgroups se usa solo para la evaluación final. Está formado por mensajes posteriores a los de train, por eso los resultados en test son menores que en validación.
- TP4: se eliminan las imágenes duplicadas y se divide en train/validación/test (60/20/20) de forma estratificada con `SEED_SPLIT = 42`. La normalización usa estadísticas calculadas solo con train y la augmentation se aplica solo a train.
- Los tiempos de entrenamiento dependen del hardware. En GPU los resultados pueden variar levemente entre ejecuciones, porque algunas operaciones no son deterministas.
- No se suben datasets, modelos entrenados ni registros de MLflow (`mlruns/`, `mlflow.db`).

## Dependencias

Ver [`requirements.txt`](requirements.txt). Principales: PyTorch, scikit-learn, NLTK, Optuna, MLflow, pandas, matplotlib y Pillow.
