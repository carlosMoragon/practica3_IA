import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Especifica la ruta del archivo CSV
ruta_csv = 'dataset.csv'

# Carga el archivo CSV en un DataFrame de pandas
dataset = pd.read_csv(ruta_csv)

# Muestra las primeras filas del DataFrame
# print(dataset.head())

print(dataset["path"].__class__)

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

# Resumen del modelo
modelo.summary()

# Tratamiento de los datos:

# Lee el archivo CSV en un DataFrame
df = pd.read_csv('archivo.csv')

# x = df['pixeles']

# y = df['numero']

X = df.drop('numero', axis=1)

y = df['numero']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características (opcional pero a menudo recomendado)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenamiento del modelo:

modelo.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


# Salvar el modelo en el fichero: modelo.h5
modelo.save("modelo.h5")

