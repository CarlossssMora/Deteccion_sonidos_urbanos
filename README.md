# Clasificación de Sonidos Urbanos con CRNN (UrbanSound8K)

Este proyecto implementa un sistema completo de **clasificación de sonidos ambientales** utilizando el dataset **UrbanSound8K**, técnicas de **procesamiento de audio**, **espectrogramas Mel** y un modelo **CRNN (Convolutional Recurrent Neural Network)** entrenado en TensorFlow/Keras.

El objetivo principal es construir un pipeline de Deep Learning capaz de identificar 10 categorías de audio, tales como: *claxon, ladrido de perro, taladro, sirena, motor, campana*, entre otros.

---

## Dataset Utilizado: UrbanSound8K

UrbanSound8K es un conjunto de datos compuesto por 8732 clips de audio en formato WAV, cada uno con una duración máxima de 4 segundos. Los audios se encuentran organizados en diez categorías de sonidos urbanos, como ladridos de perro, claxones, sirenas, herramientas mecánicas y motores. El dataset está dividido en diez "folds" predefinidos que permiten realizar validación cruzada y experimentos reproducibles. Cada clip cuenta con metadatos adicionales, entre ellos el nombre del archivo, su fold correspondiente y la etiqueta de clase anotada manualmente. Esta estructura lo convierte en un recurso ampliamente utilizado en experimentos de clasificación de audio y análisis acústico.

Dado que el dataset pesa aproximadamente 6 GB, puede obtenerlo y descargarlo a través de esta liga: https://www.kaggle.com/datasets/chrisfilo/urbansound8k/data

---

## Estructura del Proyecto
<img width="238" height="472" alt="image" src="https://github.com/user-attachments/assets/de8d0704-53d3-4176-9cc8-22232554c6a1" />

---

## Etapa 1: Análisis y Procesamiento del Audio

El primer notebook se enfoca en comprender el dataset y preparar los insumos necesarios para el modelo. Ahí se cargan los audios, se examina la información del archivo `UrbanSound8K.csv` y se analizan las distribuciones por clases y por fold. También se visualizan señales en formato waveform y diferentes representaciones espectrales, como MFCC, Chroma y Mel-Spectrogram.

Durante esta etapa se construye además un diccionario que relaciona `classID` con el nombre de la clase, permitiendo después trabajar únicamente con etiquetas textuales. A partir de los audios se generan espectrogramas Mel de tamaño fijo (128×128), normalizados y convertidos en tensores `.npy`, que luego quedan registrados en `mel_metadata.csv` junto con su clase correspondiente. Esta fase cubre todo el proceso de preprocesamiento y extracción de características.

---

## Etapa 2: Construcción del Dataset TF, Modelo y Evaluación

El segundo notebook recibe los espectrogramas y los vuelve un dataset de TensorFlow mediante `tf.data.Dataset`, integrando un codificador de etiquetas basado en `StringLookup` para transformar los nombres de clase en índices numéricos. En esta fase también se incluye un módulo de data augmentation directo sobre espectrogramas, con técnicas como time masking, frequency masking, variaciones de ganancia y pequeños desplazamientos temporales.

A partir de estos datos se entrena el modelo CRNN, el cual integra convoluciones, transformaciones temporales y una capa recurrente bidireccional. El entrenamiento incorpora validación continua, mecanismos de parada temprana y ajuste automático del learning rate. Al final se generan curvas de pérdida y precisión, y se realiza una evaluación completa que incluye métricas globales y por clase, matriz de confusión, curvas ROC multiclase y los valores de AUC promedio. El modelo final queda exportado en `crnn_urbansound8k.h5`.

---

## Arquitectura CRNN Utilizada

El modelo implementado combina convoluciones bidimensionales con una etapa recurrente pensada para capturar variaciones temporales del sonido. El flujo inicia con tres bloques convolucionales con batch normalization y max pooling, que reducen progresivamente la resolución espacial del espectrograma y aumentan la riqueza de sus características. Una vez comprimido el mapa de características, se reorganiza su estructura mediante `Permute` y una operación `TimeDistributed(GlobalAveragePooling1D)`, lo que transforma la salida convolucional en una secuencia temporal interpretable por la parte recurrente del modelo.

La red incorpora una capa **Bidirectional LSTM** para capturar dependencias temporales en ambas direcciones, seguida de una capa densa intermedia con dropout para mejorar la generalización. Finalmente, una capa softmax produce la probabilidad asociada a cada una de las diez clases del dataset.

La arquitectura completa utilizada es la siguiente:

Diagrama

---

## Cómo Ejecutar el Proyecto

### 1. Crear y activar entorno virtual
Este paso es esencial si se desea ejecutar el notebook `1_analisis.ipynb`, puesto que las bibliotecas de librosa y relacionadas funcionan de manera estable en la versión 3.11.0, la cual usted debe de contar también.

```bash
python -m venv venv311
```

#### Activación del entorno en Linux/Mac
```bash
source venv311/bin/activate
```

#### Activación del entorno en Windows
```bash
venv311\Scripts\activate
```

### 2. Instalar dependencias
Instale las dependencias **solo para el entorno venv311**
```bash
pip install -r requirements311.txt
```


### 3. Ejecutar notebooks

####1. Abrir `1_analisis.ipynb`
Para ejecutar este notebook es necesario que haya creado el entorno virtual venv311, descargado y acomodado el dataset como se muestra en la sección de estructura del proyecto. Una vez hecho este ejecute el siguiente comando para tomar el entorno virtual como kernel para **solo este notebook**:
```bash
python -m ipykernel install --user --venv311 --display-name "Python 3.11 (Audio)"
```
Finalmente seleccione el kernel creado y ejecute el notebook. Con esto verá un Análisis Exploratorio de los audios y la generación de los espectrogramas de los mismos, así como un archivo csv con la metada de estos últimos.

####2. Abrir `2_CRNN.ipynb`
Para la ejecución de este notebook no es necesario utilizar el entorno virtual. Posteriormente, ejecute el siguiente comando para asegurarse de que disponga de las librerías para el proyecto:
```bash
pip install -r requirements.txt
```
Para la ejecución de este notebook puede utilizar la versión de Python que desee. Para el caso de los autores se utilizó la versión 3.13.7. Tras su ejecución se realiza el entrenamiento y evaluación del modelo

---

## Autores

Proyecto académico para la materia:

**Redes Neuronales y Aprendizaje Profundo**  

Equipo:

- *Juan Carlos Flores Mora*  
- *Sergio de Jesús Castillo Molano*  
- *Guillermo Carreto Sánchez*  

---
