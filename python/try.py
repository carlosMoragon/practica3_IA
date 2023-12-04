'''
from PIL import Image
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


# Ruta al directorio que contiene las imágenes
directory = ".\\trainingSet\\"
numbers = list(range(10))
paths = list(map(lambda x: directory + str(x), numbers))

# Lista para almacenar los datos de las imágenes
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


# Crear un DataFrame con los datos de las imágenes
df = pd.DataFrame(datos_imagenes)

# Guardar el DataFrame en un archivo CSV
df.to_csv('dataset_imagenes.csv', index=False)

print("El dataset de imágenes ha sido guardado en 'dataset_imagenes.csv'.")

'''

'''
# Cargar el DataFrame desde el archivo CSV
df = pd.read_csv('dataset_imagenes.csv')

# Iterar sobre las filas del DataFrame
for index, row in df.iterrows():
    # Obtener el nombre del archivo y los datos de los píxeles
    nombre_archivo = row['archivo']

    # Convertir la cadena de píxeles a una lista de enteros
    datos_pixeles = ast.literal_eval(row['pixeles'])
    
    # Reformatear los datos a una matriz cuadrada (asumiendo que es una imagen cuadrada)
    lado_imagen = int(np.sqrt(len(datos_pixeles)))
    matriz_pixeles = np.array(datos_pixeles).reshape((lado_imagen, lado_imagen))

    # Mostrar la imagen
    plt.imshow(matriz_pixeles, cmap='gray')  # cmap='gray' para imágenes en escala de grises
    plt.title(f'Imagen: {nombre_archivo}')
    plt.show()

'''

'''
from PIL import Image
def determinar_tipo_imagen(ruta_imagen):
    # Abrir la imagen
    imagen = Image.open(ruta_imagen)
    
    # Obtener el modo de la imagen (L para blanco y negro, RGB para color)
    modo = imagen.mode
    
    if modo == 'L':
        return 'Blanco y Negro'
    elif modo == 'RGB':
        return 'Color'
    else:
        return 'Desconocido'

# Ruta de tu imagen
ruta_imagen = ".\\trainingSet\\0\\img_1.jpg"

# Determinar el tipo de imagen
tipo_imagen = determinar_tipo_imagen(ruta_imagen)
size = Image.open(ruta_imagen).size
print(f'Tipo de imagen: {size}')
'''
'''
# n = [3, 0, 0, 3, 7, 3, 0, 3, 0, 11, 0, 0, 3, 0, 0, 3, 8, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 0, 12, 0, 16, 0, 0, 4, 0, 2, 8, 3, 0, 4, 8, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 1, 12, 0, 8, 0, 0, 6, 0, 11, 0, 0, 6, 7, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 2, 3, 0, 0, 0, 12, 0, 0, 23, 0, 0, 0, 0, 11, 3, 0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 6, 0, 25, 27, 136, 135, 188, 89, 84, 25, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 3, 88, 247, 236, 255, 249, 250, 227, 240, 136, 37, 1, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 4, 27, 193, 251, 253, 255, 255, 255, 255, 240, 254, 255, 213, 89, 0, 0, 14, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 18, 56, 246, 255, 253, 243, 251, 255, 245, 255, 255, 254, 255, 231, 119, 7, 0, 5, 0, 0, 0, 0, 4, 0, 0, 12, 13, 0, 65, 190, 246, 255, 255, 251, 255, 109, 88, 199, 255, 247, 250, 255, 234, 92, 0, 0, 0, 0, 0, 0, 0, 10, 1, 0, 0, 18, 163, 248, 255, 235, 216, 150, 128, 45, 6, 8, 22, 212, 255, 255, 252, 172, 0, 15, 0, 0, 0, 0, 0, 1, 4, 5, 0, 0, 187, 255, 254, 94, 57, 7, 1, 0, 6, 0, 0, 139, 242, 255, 255, 218, 62, 0, 0, 0, 0, 0, 5, 2, 0, 0, 11, 56, 252, 235, 253, 20, 5, 2, 5, 1, 0, 1, 2, 0, 97, 249, 248, 249, 166, 8, 0, 0, 0, 0, 0, 0, 2, 0, 0, 70, 255, 255, 245, 25, 10, 0, 0, 1, 0, 4, 10, 0, 10, 255, 246, 250, 155, 0, 0, 0, 0, 0, 2, 0, 7, 12, 0, 87, 226, 255, 184, 0, 3, 0, 10, 5, 0, 0, 0, 9, 0, 183, 251, 255, 222, 15, 0, 0, 0, 0, 0, 5, 1, 0, 19, 230, 255, 243, 255, 35, 2, 0, 0, 0, 0, 9, 8, 0, 0, 70, 245, 242, 255, 14, 0, 0, 0, 0, 0, 4, 3, 0, 19, 251, 239, 255, 247, 30, 1, 0, 4, 4, 14, 0, 0, 2, 0, 47, 255, 255, 247, 21, 0, 0, 0, 0, 6, 0, 2, 2, 0, 173, 247, 252, 250, 28, 10, 0, 0, 8, 0, 0, 0, 8, 0, 67, 249, 255, 255, 12, 0, 0, 0, 0, 0, 0, 6, 3, 0, 88, 255, 251, 255, 188, 21, 0, 15, 0, 8, 2, 16, 0, 35, 200, 247, 251, 134, 4, 0, 0, 0, 0, 0, 3, 3, 1, 0, 11, 211, 247, 249, 255, 189, 76, 0, 0, 4, 0, 2, 0, 169, 255, 255, 247, 47, 0, 0, 0, 0, 0, 0, 6, 0, 0, 2, 0, 59, 205, 255, 240, 255, 182, 41, 56, 28, 33, 42, 239, 246, 251, 238, 157, 0, 1, 0, 0, 0, 0, 2, 1, 0, 0, 2, 10, 0, 104, 239, 255, 240, 255, 253, 247, 237, 255, 255, 250, 255, 239, 255, 100, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 7, 0, 4, 114, 255, 255, 255, 255, 247, 249, 253, 251, 254, 237, 251, 89, 0, 0, 1, 0, 0, 0, 0, 0, 0, 9, 0, 0, 1, 13, 0, 14, 167, 255, 246, 253, 255, 255, 254, 242, 255, 244, 61, 0, 19, 0, 1, 0, 0, 0, 0, 2, 1, 7, 0, 0, 4, 0, 14, 0, 27, 61, 143, 255, 255, 252, 255, 149, 21, 6, 16, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

from PIL import Image
import numpy as np

# Tu lista de valores
pixel_values = [3, 0, 0, 3, 7, 3, 0, 3, 0, 11, 0, 0, 3, 0, 0, 3, 8, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 0, 12, 0, 16, 0, 0, 4, 0, 2, 8, 3, 0, 4, 8, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 1, 12, 0, 8, 0, 0, 6, 0, 11, 0, 0, 6, 7, 2, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 2, 3, 0, 0, 0, 12, 0, 0, 23, 0, 0, 0, 0, 11, 3, 0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 6, 0, 25, 27, 136, 135, 188, 89, 84, 25, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 3, 88, 247, 236, 255, 249, 250, 227, 240, 136, 37, 1, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 4, 27, 193, 251, 253, 255, 255, 255, 255, 240, 254, 255, 213, 89, 0, 0, 14, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 18, 56, 246, 255, 253, 243, 251, 255, 245, 255, 255, 254, 255, 231, 119, 7, 0, 5, 0, 0, 0, 0, 4, 0, 0, 12, 13, 0, 65, 190, 246, 255, 255, 251, 255, 109, 88, 199, 255, 247, 250, 255, 234, 92, 0, 0, 0, 0, 0, 0, 0, 10, 1, 0, 0, 18, 163, 248, 255, 235, 216, 150, 128, 45, 6, 8, 22, 212, 255, 255, 252, 172, 0, 15, 0, 0, 0, 0, 0, 1, 4, 5, 0, 0, 187, 255, 254, 94, 57, 7, 1, 0, 6, 0, 0, 139, 242, 255, 255, 218, 62, 0, 0, 0, 0, 0, 5, 2, 0, 0, 11, 56, 252, 235, 253, 20, 5, 2, 5, 1, 0, 1, 2, 0, 97, 249, 248, 249, 166, 8, 0, 0, 0, 0, 0, 0, 2, 0, 0, 70, 255, 255, 245, 25, 10, 0, 0, 1, 0, 4, 10, 0, 10, 255, 246, 250, 155, 0, 0, 0, 0, 0, 2, 0, 7, 12, 0, 87, 226, 255, 184, 0, 3, 0, 10, 5, 0, 0, 0, 9, 0, 183, 251, 255, 222, 15, 0, 0, 0, 0, 0, 5, 1, 0, 19, 230, 255, 243, 255, 35, 2, 0, 0, 0, 0, 9, 8, 0, 0, 70, 245, 242, 255, 14, 0, 0, 0, 0, 0, 4, 3, 0, 19, 251, 239, 255, 247, 30, 1, 0, 4, 4, 14, 0, 0, 2, 0, 47, 255, 255, 247, 21, 0, 0, 0, 0, 6, 0, 2, 2, 0, 173, 247, 252, 250, 28, 10, 0, 0, 8, 0, 0, 0, 8, 0, 67, 249, 255, 255, 12, 0, 0, 0, 0, 0, 0, 6, 3, 0, 88, 255, 251, 255, 188, 21, 0, 15, 0, 8, 2, 16, 0, 35, 200, 247, 251, 134, 4, 0, 0, 0, 0, 0, 3, 3, 1, 0, 11, 211, 247, 249, 255, 189, 76, 0, 0, 4, 0, 2, 0, 169, 255, 255, 247, 47, 0, 0, 0, 0, 0, 0, 6, 0, 0, 2, 0, 59, 205, 255, 240, 255, 182, 41, 56, 28, 33, 42, 239, 246, 251, 238, 157, 0, 1, 0, 0, 0, 0, 2, 1, 0, 0, 2, 10, 0, 104, 239, 255, 240, 255, 253, 247, 237, 255, 255, 250, 255, 239, 255, 100, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 7, 0, 4, 114, 255, 255, 255, 255, 247, 249, 253, 251, 254, 237, 251, 89, 0, 0, 1, 0, 0, 0, 0, 0, 0, 9, 0, 0, 1, 13, 0, 14, 167, 255, 246, 253, 255, 255, 254, 242, 255, 244, 61, 0, 19, 0, 1, 0, 0, 0, 0, 2, 1, 7, 0, 0, 4, 0, 14, 0, 27, 61, 143, 255, 255, 252, 255, 149, 21, 6, 16, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Convertir la lista a un array de numpy
pixel_array = np.array(pixel_values, dtype=np.uint8)

# Reshape el array para que tenga la forma (28, 28)
pixel_array = pixel_array.reshape((28, 28))

# Crear una imagen usando Pillow
image = Image.fromarray(pixel_array)

# Guardar la imagen o mostrarla
image.save("imagen_resultante.png")
image.show()
'''

import pandas as pd
# Tratamiento de los datos:

# Lee el archivo CSV en un DataFrame
df = pd.read_csv('dataset_imagenes.csv')

X = df['pixeles']
print(df.info())

# y = df['numero']

# X =  df.drop('numero', axis=1)

y = df['numero']

