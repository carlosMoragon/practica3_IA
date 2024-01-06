# practica3_IA

### ENUNCIADO: Práctica de aprendizaje automático

#### Pasos a realizar

Los pasos a realizar son los que se muestran a continuación:

1. **Elegir el problema**. La predicción de resultados de partidos en la ATP
o en la WTA de tenis, una liga deportiva, una competición de deporte
electrónico, o cualquier otro problema del que se dispongan de datos
suficientes como para aplicar algoritmos de aprendizaje automático.

2. **Identificar la fuente de datos**. Es necesario disponer de una serie de
datos históricos que sirvan para que el sistema aprenda.

3. **Identificar las características relevantes de los hechos.** Por ejemplo, en
la predicción del resultado de un encuentro de tenis, algunas características importantes pueden ser la posición en el ranking de cada jugador,
la superficie en que se juega, la edad de cada jugador, etc.

4. **Obtener un fichero .arff** con los hechos codificados de acuerdo con las
características anteriormente elegidas. Este fichero servirá como entrada para la herramienta Weka2.

5. **Evaluar distintos algoritmos de aprendizaje automático** con los datos
obtenidos. Este paso se llevará a cabo con la herramienta Weka, y
tendrá como salida el algoritmo con mejor rendimiento para los datos
datos.

6. **Generar en Java un objeto persistente** con el algoritmo obtenido en el
paso 5. También se realizará con Weka.

7. **Implementar un prototipo de aplicación que consulte el objeto persistente** generado en el paso 6. La aplicación cargará en memoria el objeto
persistente, que tendrá como responsabilidad la resolución del problema propuesto (p.ej. el pronóstico de un resultado deportivo), e interactuará con el usuario a través de una interfaz (que puede ser de texto)
(véase la figura 1.2).

#### Material a entregar

Deberá ser el siguiente:

1. Cógido fuente en Java.
2. Los ficheros adicionales necesarios para poder compilar dicho código.
3. El fichero .arf


### Resolución de la práctica

#### Elección del problema:

El problema a resolver de cara a esta práctica es la **clasificación de imagenes en números del 0-9**.


#### Elección de fuentes de datos:

Para la elección de los datos se han tomado fotografías de números del 0-9.

Los datos escogidos constan con un total de 42000 fotografias.

![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/grafica_datos.png)

#### Preprocesamiento de datos:

Al trabajar con imagenes, se ha llevado a cabo un preprocesamiento de datos cuyo principal objetivo era separar las imagenes por pixeles, siendo estos pixeles las columnas del archivo .csv.

Este preprocesamiento se ha hecho por medio de python, teniendo como resultado todos los datos en un archivo .csv.

```{python}
   from PIL import Image
   import pandas as pd
   import os
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Tras ejecutarlo hay que tratar el dataset
   
   directory = ".\\trainingSet\\"
   numbers = list(range(10))
   paths = list(map(lambda x: directory + str(x), numbers))
   
   datos_imagenes = []
   for ruta_directorio in paths:

       # Iterar sobre las imágenes en el directorio
       for nombre_archivo in os.listdir(ruta_directorio):

           if nombre_archivo.endswith(('.jpg', '.png', '.jpeg')):

               # Construir la ruta completa al archivo de imagen
               ruta_imagen = os.path.join(ruta_directorio, nombre_archivo)
   
               # Abrir la imagen con Pillow
               imagen = Image.open(ruta_imagen)
   
               # Convertir la imagen a una serie de números (píxeles)
               datos_pixeles = list(imagen.getdata())
   
               # Agregar datos de la imagen a la lista
               datos_imagenes.append({'numero': int(ruta_directorio.split("\\trainingSet\\")[1][0]), 'pixeles': datos_pixeles})
   
   
   df = pd.DataFrame(datos_imagenes)
   
   df.to_csv('dataset_imagenes.csv', index=False)
   
   print("El dataset de imágenes ha sido guardado en 'dataset_imagenes.csv'.")

```

Al empezar a entrenar el modelo con los datos me percate de que hacia falta realizar una modificación en el fichero. Esto se debe a que tenia como cabecera "numero, pixeles", por lo que el dataset era de 2 
columnas.

Para solucionarlo investigué como lo habian realizado otras personas, y me encontré varios casos que seguian la notación: 

label,1x1,1x2,1x3,1x4,1x5,1x6,1x7,1x8,1x9,1x10,1x11,1x12, ... ,28x26,28x27,28x28

Siendo **label** la columna utilizada como clase, y las demás los píxeles. En mi caso, trato con imagenes de 28x28 pixeles.

![archivo csv](https://https://github.com/carlosMoragon/practica3_IA/blob/main/python/dataset_imagenes.csv)

![archivo .zip con las imagenes](https://https://github.com/carlosMoragon/practica3_IA/blob/main/python/dataset_imagenes.zip)



#### Identificar las características relevantes de los hechos:

Al estar trabajando con clasificación de imagenes, los parametros más relevantes son:

- Los píxeles de la imagen.
- El tipo de la imagen (Imágenes a color o blanco y negro).
- El tamaño de las imagenes.
- El número que se representa en la imagen.

Tras el preprocesamiento de datos, se han establecido todas las imagenes del mismo tamaño y del mismo tipo, quedando como caractirísticas relevantes:

- Los píxeles de la imagen.
- El número que se representa en la imagen.


#### Obtener un fichero .arff:

Después del preprocesamiento de datos, se creó un archivo .arff, el cual se utilizará para evaluar distintos algoritmos.

![archivo .arff](https://github.com/carlosMoragon/practica3_IA/blob/main/python/dataset_imagenes.csv.arff)

#### Evaluar distintos algoritmos de aprendizaje automático:

En este paso obtuve un inconveniente, debido a la cantidad de datos utilizados.

Mi conjunto de datos consta de 42000 registros, los cuales tiene 785 columnas, de las cuales 1 es 'label', es decir, el número que se representa en la imagen, y las otras 784 son cada uno de los píxeles.

Al tener tantos datos y una capacidad de cómputo limitada, la evaluación de los algortimos no ha sido del todo satisfactoria.

##### Random Forest:

Random Forest es un algoritmo que crea multiples árboles de decisión durante su entrenamiento, convinandolos para tener un árbol más robusto y preciso.

Random forest es utilizado en problemas de clasificación y regresión.

![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/randomForest.png)

Al evaluar los datos con Random Forest, observamos:

(Ignoramos el coeficiente de correlación, debido a que no es un problema de regresión, sino de clasificación.)

- **Error absoluto medio:**
Nos sale un valor bajo (0,0035), lo que es deseable.

Lo que indica que las predicciones que realiza el modelo, difieren muy poco del valor real.
  
- **Error cuadrático medio:**
El Error cuadrático medio es más grande que el Error cuadrático medio.

Puede deberse a la penalización adicional de los errores más grandes. 
  
- **Error absoluto relativo:**
El error absoluto relativo es de 90,0223%, lo que es relativamente alto.

Sugiere que las predicciones pueden tener un margen de error considerable en relación con los valores reales.

- **Error cuadrático relativo:**
Igual que el Error absoluto Relativo, indica un alto error cuadrático en las predicciones del modelo.



Al obtener estos datos, se podrían ajustar el número de árboles creados por el modelo, pero se ha optado por otra opción.


##### Regresión Lineal:

![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/regresionLineal.png)

El modelo de regresión lineal fue descartado tras un largo tiempo de procesamiento/entrenamiento del modelo.

Es de suponer que el alto número de datos (42000) y la alta cantidad de características (785), han hecho que este algoritmo tarde mucho tiempo en el entrenamiento.

Al trascurrir varias horas, se decidio interrumpir el procesamiento del algoritmo, siendo una variable crítica el rápido entrenamiento de un modelo.


##### Perceptron Multicapa / Red Neuronal:

![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/MultiLayerPerceptron.png)

Buscando una mejora del tiempo de procesamiento, se decidio probar con el algoritmo Perceptrón Multicapa.

Tras 3 horas de procesamiento, se decidio interrumpir la ejecución del algoritmo.

Las posibles razones del largo tiempo de procesamiento pueden ser:

- Alta complejidad del modelo, es decir, gran numero de capas y/o gran numero de neuronas por capa.
- Una mala optimización de los hiperparámetros, es decir:
     - Mala tasa de aprendizaje.
     - Tamaño de lote alto.
     - Función de activación no adecuada.

##### Red neuronal convolucional:

Tras un análisis de los problemas anteriormente comentados, se decidió crear manualmente una red neuronal en un sistema con un alto procesamiento.

Para la creación de esta red neuronal se han usado las siguientes herramientas:

- **Google Colab:** Entorno de desarrollo en la nube, con una capacidad de computo superior a la utilizada anteriormente.
- **Librerias de python:**
     - **pandas:** Utilizado para el manejo de datos en forma de datasets.
     - **tensorflow:** Utilizado para la creación de la red neuronal.
     - **sklearn:** Utilizado para el entrenamiento de la red neuronal.
     - **matplotlib:** Utilizado para evaluar de forma gráfica el entrenamiento del modelo.



#### Construcción de una red convolucional:


#### Generar en Java un objeto persistente:


#### Prototipo de aplicación que consulte al modelo:

