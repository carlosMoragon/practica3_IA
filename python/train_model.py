import pandas as pd
import numpy as np
import ast
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("dataset_imagenes.csv")

# La primera columna es la etiqueta: lo que va a ser la salida
# Estamos cogiendo los datos de las distintas columnas
X = df.iloc[:, 1:].values  # Características (píxeles)
y = df.iloc[:, 0].values   # Etiquetas

# Normalizar los valores de píxeles:
# Los pixeles van de 0 a 255 asi que si dividimos entre 255 nos saldrá entre 0 y 1 los valores.
X = X / 255.0

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Asumimos que la imagen está en blanco y negro, y es de 28x28 (como en los datos de entrenamiento)
# EL -1 es para que se redimensione automáticamente el número de imagenes
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# El modelo me exige que las etiquetas sean en modelo one-hot. 
# Siendo más expecifico es la función softmax de la última la que me lo exige.
# El 10 es por que hay 10 categorias entonces será [1,0,0,0,0,0,0,0,0,0] por ejemplo si es el número 0.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# Definición de la red neuronal

# definimos un modelo de red neuronal secuencial
modelo = tf.keras.models.Sequential()

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

# He añadido 3 capas por que es el estandar para redes CNN (Convolucionales)

# Es una capa de agrupación, lo que hace es resumir la información agrupandola para reducir la cantidad
# de parametros en la red.
# Utilizo MaxPooling por que se supone que es mejor para datos reelevantes,
# en cambio AveragePooling, es mejor para una representación más general.
modelo.add(tf.keras.layers.MaxPooling2D((2, 2)))

modelo.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

modelo.add(tf.keras.layers.MaxPooling2D((2, 2)))

modelo.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

# Aplanar la salida de la última capa convolucional
# Esto se hace por que se tiene que transformar una salida 3 dimensional (altura, anchura y dimensiones)
# a una salida unidimensional para que se pueda utilizar una capa densa para la clasificación.
modelo.add(tf.keras.layers.Flatten())


# Una capa densa que cada neurona está totalmente conectada con la capa anterior para hacer la clasificación.
# El valor 64 es el numero de neuronas, dependerá de la complejidad del modelo.
modelo.add(tf.keras.layers.Dense(64, activation='relu'))

# En este caso utilizaria sigmoide si quisiese una clasificación binaria.
# Como tengo que clasificar en 10 casos añado 10 neuronas de salida.
# He utilizado la función softmax, debido a que te da la probabilidad de cada cosa.
modelo.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compilar el modelo
# Adam: es el algoritmo que se encarga de ajustar los pesos en el entrenamiento.
# La tasa de aprendizaje se adapta en cada momento, no es fija, aprendiendo de los gradientes previamente calculados.
# He utilizado una función de perdida: categorical_crossentropy debido al problema multiclase.
# La métrica mide la fracción de muestras correctamente clasificadas.
# He utilizado accurancy (Precisión) por que he visto que es muy común para problemas de clasificación de imagenes.
modelo.compile(optimizer='adam',
              loss='categorical_crossentropy', # si fuese binaria la solución: binary_crossentropy
              metrics=['accuracy']) # Nosotros lo vimos como precisión.


# Entrenamos el modelo: lo guardamos en un history para ver como ha ido aprendiendo
history = modelo.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Acceso a las métricas para ver como ha ido aprendiendo
print(history.history['accuracy'])
print(history.history['val_accuracy'])


# A continuación está sacado de chatgpt para ver como de bien ha aprendido mi modelo 

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Evalua el modelo con test.
test_loss, test_accuracy = modelo.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# ------------ Hasta aquí sacado de chatgpt -----------

# Lo guardo en 2 tipos por que me sale un warning diciendo que .h5 es legacy

# Salvar el modelo en el fichero: modelo.h5
modelo.save("modelo.h5")
# Salvar el modelo en el fichero: my_model.keras
modelo.save('my_model.keras')