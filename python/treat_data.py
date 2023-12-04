
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

