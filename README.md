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


#### Evaluar distintos algoritmos de aprendizaje automático:


#### Construcción de una red convolucional:


#### Generar en Java un objeto persistente:


#### Prototipo de aplicación que consulte al modelo:

