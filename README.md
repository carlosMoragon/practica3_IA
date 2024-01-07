practica3_IA
=======

## Realizada por Carlos Moragón Corella

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

```python
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


**Random Forest:**
   
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
   
   
**Regresión Lineal:**
   
   ![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/regresionLineal.png)
   
   El modelo de regresión lineal fue descartado tras un largo tiempo de procesamiento/entrenamiento del modelo.
   
   Es de suponer que el alto número de datos (42000) y la alta cantidad de características (785), han hecho que este algoritmo tarde mucho tiempo en el entrenamiento.
   
   Al trascurrir varias horas, se decidio interrumpir el procesamiento del algoritmo, siendo una variable crítica el rápido entrenamiento de un modelo.
   

**Perceptron Multicapa / Red Neuronal:**
   
   ![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/MultiLayerPerceptron.png)
   
   Buscando una mejora del tiempo de procesamiento, se decidio probar con el algoritmo Perceptrón Multicapa.
   
   Tras 3 horas de procesamiento, se decidio interrumpir la ejecución del algoritmo.
   
   Las posibles razones del largo tiempo de procesamiento pueden ser:
   
   - Alta complejidad del modelo, es decir, gran numero de capas y/o gran numero de neuronas por capa.
   - Una mala optimización de los hiperparámetros, es decir:
        - Mala tasa de aprendizaje.
        - Tamaño de lote alto.
        - Función de activación no adecuada.


**Red neuronal convolucional:**
   
   Tras un análisis de los problemas anteriormente comentados, se decidió crear manualmente una red neuronal en un sistema con un alto procesamiento.
   
   Para la creación de esta red neuronal se han usado las siguientes herramientas:
   
   - **Google Colab:** Entorno de desarrollo en la nube, con una capacidad de computo superior a la utilizada anteriormente.
   - **Librerias de python:**
        - **pandas:** Utilizado para el manejo de datos en forma de datasets.
        - **tensorflow:** Utilizado para la creación de la red neuronal.
        - **sklearn:** Utilizado para el entrenamiento de la red neuronal.
        - **matplotlib:** Utilizado para evaluar de forma gráfica el entrenamiento del modelo.
   
   Tras la construcción del modelo y su entrenamiento, se decidio evaluar según lo exacto que es el modelo y según sus perdidas:
   
   
   **Precisión:**
   
   Tras entrenar el modelo, podemos observar una precisión que ronda el 98-99%, pudiendo decir que es un modelo bastante preciso.
   
   ![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/exactitud.png)
   
   
   **Perdidas:**
   
   Tras entrenar el modelo, podemos observar un valor de la función de perdida entorno al 5%, pudiendo concluir que está haciendo conclusiones bastantes precisas.
   
   ![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/perdidas.png)
   
   
   Al analizar como de preciso es, podemos concluir que es el algoritmo que mejor se ajusta al problema que afrontamos.
   

#### Construcción de una red convolucional:

La construcción de la red convolucional se puede dividir en varias partes:

1. **Separación de datos de entrenamiento y datos de prueba**

   Descargamos los datos del archivo .csv:
   ```python
      df = pd.read_csv("dataset_imagenes.csv")
   ```
   
   Separamos la solución de las caractirísticas:
   ```python
      # La primera columna es la etiqueta: lo que va a ser la salida
      # Estamos cogiendo los datos de las distintas columnas
      X = df.iloc[:, 1:].values  # Características (píxeles)
      y = df.iloc[:, 0].values   # Etiquetas
   ```
   
   Normalizamos los datos, para optimizar el entrenamiento:
   ```python
      # Normalizar los valores de píxeles:
      # Los pixeles van de 0 a 255 asi que si dividimos entre 255 nos saldrá entre 0 y 1 los valores.
      X = X / 255.0
   ```
   
   Dividimos los datos en el entrenamiento y las soluciones:
   ```python
      # Dividir los datos en conjuntos de entrenamiento y prueba
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```
   
   Redimensionamos las fotografias a un tamaño 28x28 píxeles
   ```python
      # Asumimos que la imagen está en blanco y negro, y es de 28x28 (como en los datos de entrenamiento)
      # EL -1 es para que se redimensione automáticamente el número de imagenes
      X_train = X_train.reshape(-1, 28, 28, 1)
      X_test = X_test.reshape(-1, 28, 28, 1)
   ```
   
   Cambiamos las etiquetas de solucion a one-hot, para poder compararlo posteriormente.
   ```python
      # El modelo me exige que las etiquetas sean en modelo one-hot. 
      # Siendo más expecifico es la función softmax de la última la que me lo exige.
      # El 10 es por que hay 10 categorias entonces será [1,0,0,0,0,0,0,0,0,0] por ejemplo si es el número 0.
      y_train = tf.keras.utils.to_categorical(y_train, 10)
      y_test = tf.keras.utils.to_categorical(y_test, 10)
   ```

2. **Creación de una red neuronal:**

   Definimos un modelo de red secuencial, es decir, no saltos de conexiones entre capas, cada capa se conecta con la siguiente capa.
   ```python
   # Definición de la red neuronal

   # definimos un modelo de red neuronal secuencial
   modelo = tf.keras.models.Sequential()
   ```

   En la capa de entrada metemos 32 neuronas, de las cuales deja entrar a 3x3 píxeles por neurona.
   Utilizamos la función de activación 'relu', la cual es más idonea para una red convolucional.
   ```python
   # Una primera capa:
   # Metemos neuronas en 2 dimensiones, las cuales son muy comunes en redes convolucionales, para imagenes.
   # Estás van a aprender 32 características.
   # un kernel_size de 3x3, lo que significa que deja entrar 3x3 pixeles por neurona.
   # En vez de utilizar la función sigmoide utilizamos la función relu
   # la cual, he investigado y, es mejor para este tipo de red.
   # el input_shape corresponde al tamaño de la imagen 64x64 y 1 es debido a que es solo en blanco y negro
   # si fuese en color podria 3 debido a rgb (3 dimensiones; red, green, blue)
   # Relu: lo que hace es activar la neurona si la entrada es positiva y pasar de ella si es negativa.
   
   modelo.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

   ```

   Después de cada capa realizamos una agrupación, reduciendo la información a tratar y optimizando el entrenamiento del modelo.
   Para dicha agrupación hemos utilizado "MaxPooling", un agrupamiento por el mayor valor, funcionando muy bien con datos relevantes.
   ```python
   # He añadido 3 capas por que es el estandar para redes CNN (Convolucionales)
   
   # Es una capa de agrupación, lo que hace es resumir la información agrupandola para reducir la cantidad
   # de parametros en la red.
   # Utilizo MaxPooling por que se supone que es mejor para datos reelevantes,
   # en cambio AveragePooling, es mejor para una representación más general.
   modelo.add(tf.keras.layers.MaxPooling2D((2, 2)))
   ```

   Como capas intermedias se ha implementado 3 capas, 2 de ellas con un tamaño del kernel de 3x3 píxeles, utilizando como función de activación 'relu'.
   ```python
   modelo.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
   
   modelo.add(tf.keras.layers.MaxPooling2D((2, 2)))
   
   modelo.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

   ```

   La 3ª capa intermedia es una densa, lo que significa que cada neurona de la capa tiene una relación con cada neurona de la capa anterior.
   La elección de una capa densa es para recoger toda la información de la capa anterior y minimizar las perdidas.
   Antes de la capa densa, aplanamos los datos a una dimensión, en nuestro caso tenemos altura y anchura, y lo transformamos en un número.
   ```python
   # Aplanar la salida de la última capa convolucional
   # Esto se hace por que se tiene que transformar una salida 3 dimensional (altura, anchura y dimensiones)
   # a una salida unidimensional para que se pueda utilizar una capa densa para la clasificación.
   modelo.add(tf.keras.layers.Flatten())
   
   
   # Una capa densa que cada neurona está totalmente conectada con la capa anterior para hacer la clasificación.
   # El valor 64 es el numero de neuronas, dependerá de la complejidad del modelo.
   modelo.add(tf.keras.layers.Dense(64, activation='relu'))

   ```

   La capa de salida es también una capa densa, en la que tenemos 10 neurona, una por cada elemento de la clasificación (0-9).
   También cambiamos la función de activación a 'softmax', por que te realiza una clasificación no binaria, en nuestro caso en 10 elementos.
   ```python
   # En este caso utilizaria sigmoide si quisiese una clasificación binaria.
   # Como tengo que clasificar en 10 casos añado 10 neuronas de salida.
   # He utilizado la función softmax, debido a que te da la probabilidad de cada cosa.
   modelo.add(tf.keras.layers.Dense(10, activation='softmax'))

   ```

   Tras establecer las capas compilamos el modelo, declarando las distintas funciones a utilizar en nuestro modelo.
   Utilizamos 'adam' para ajustar los pesos en el momento del entrenamiento.
   Utilizamos 'categorical_crossentropy' como función de pérdida, debido a que tenemos un problema de clasificación multiclase.
   Utilizamos 'accuracy' (precisión), como variable para evaluar el modelo.
   ```python
   # Compilar el modelo
   # Adam: es el algoritmo que se encarga de ajustar los pesos en el entrenamiento.
   # La tasa de aprendizaje se adapta en cada momento, no es fija, aprendiendo de los gradientes previamente calculados.
   # He utilizado una función de perdida: categorical_crossentropy debido al problema multiclase.
   # La métrica mide la fracción de muestras correctamente clasificadas.
   # He utilizado accurancy (Precisión) por que he visto que es muy común para problemas de clasificación de imagenes.
   modelo.compile(optimizer='adam',
                 loss='categorical_crossentropy', # si fuese binaria la solución: binary_crossentropy
                 metrics=['accuracy']) # Nosotros lo vimos como precisión.
   ```

3. **Entrenamiento del modelo:**

   Entrenamos el modelo durante 10 épocas.

   Durante cada época, el modelo realizará el proceso de retropropagación y ajustará sus pesos para minimizar la función de pérdida en el conjunto de
   entrenamiento. La retropropagación implica calcular el gradiente de la función de pérdida con respecto a los pesos del modelo y luego utilizar un algoritmo de
   optimización para actualizar los pesos en la dirección que minimiza la pérdida.
   ```python
   # Entrenamos el modelo: lo guardamos en un history para ver como ha ido aprendiendo
   history = modelo.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
   
   # Acceso a las métricas para ver como ha ido aprendiendo
   print(history.history['accuracy'])
   print(history.history['val_accuracy'])
   
   ```


#### Generar un objeto persistente:

Para generar un objeto persistente del modelo se ha guardado en 2 formatos distintos:

- **.h5**
- **.keras**

El archivo .keras, aunque no se utiliza en la aplicación, se ha generado con motivo de esta advertencia:

UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.

Esta advertencia nos comunica que la extensión .h5 es *legacy*, es decir, que hace uso de dependencias antiguas de la libreria HDF5, la cual guarda el modelo.


#### Prototipo de aplicación que consulte al modelo

Para probar el buen funcionamiento del modelo, se ha realizado un prototipo de aplicación, en el cual se pueden dibujar números y se evaluarán por medio del modelo entrenado.

Esta aplicación ha sido desarrollada en python, en el fichero ![app.py](https://github.com/carlosMoragon/practica3_IA/blob/main/src/app.py).

La ejecución de la aplicación se ha pensado de 2 formas distintas:

1. **Docker:**
   Si desea ejecutarlo mediante docker, deberá:
   - Tener instalado docker
   - Dar permisos de ejecución al fichero: ![ejecución.sh](https://github.com/carlosMoragon/practica3_IA/blob/main/ejecucion.sh), por ejemplo: chmod 777 ejecucion.sh
   - Ejecutar el fichero 'ejecución.sh', el cual le desactivará por unos instantes la variable $DISPLAY, para poder desplegar la interfaz de usuario, pero al final de la ejecución se restablecerá.

2. **Python:**
   Si desea ejecutar con su entorno python, deberá instalarse las dependencias indicadas en el fichero ![requierements.txt](https://github.com/carlosMoragon/practica3_IA/blob/main/requirements.txt), ejecutando el siguiente comando:
   pip install -r requirements.txt

   A continuación, ya podrá ejecutar el fichero app.py, con el siguiente comando:
   python3 ./src/app.py

**Ejemplos:**

**0:**
![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/Captura%20de%20pantalla%20de%202024-01-07%2018-29-06.png)


**1:**
![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/Captura%20de%20pantalla%20de%202024-01-07%2018-49-09.png)


**2:**
![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/Captura%20de%20pantalla%20de%202024-01-07%2018-30-04.png)


**3:**
![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/Captura%20de%20pantalla%20de%202024-01-07%2018-53-10.png)


**4:**
![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/Captura%20de%20pantalla%20de%202024-01-07%2018-30-42.png)


**5:**
![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/Captura%20de%20pantalla%20de%202024-01-07%2018-30-53.png)


**6:**
![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/Captura%20de%20pantalla%20de%202024-01-07%2018-49-39.png)


**7:**
![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/Captura%20de%20pantalla%20de%202024-01-07%2018-51-41.png)


**8:**
![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/Captura%20de%20pantalla%20de%202024-01-07%2018-31-37.png)


**9:**
![](https://github.com/carlosMoragon/practica3_IA/blob/main/imgs_readme/Captura%20de%20pantalla%20de%202024-01-07%2018-31-51.png)
