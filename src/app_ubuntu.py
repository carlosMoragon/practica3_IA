import tkinter as tk
from PIL import Image, ImageOps  # Importa ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import os
from mss import mss

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paint App")

        self.canvas = tk.Canvas(root, bg="black", width=400, height=400)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        # COnfiguración del evento del ratón
        self.canvas.bind("<B1-Motion>", self.paint)

        # Agregamos un botón para capturar la imagen
        capture_button = tk.Button(root, text="Capturar Imagen", command=self.capture_image)
        capture_button.pack(side=tk.BOTTOM)

        # Creamos una variable para guardar el resultado del modelo
        self.inference_result = tk.StringVar()
        result_label = tk.Label(root, textvariable=self.inference_result, fg="white", bg="black")
        result_label.pack(side=tk.BOTTOM)

        # Cargamos el modelo entrenado
        self.model = load_model('modelo.h5')

    def paint(self, event):
        # Ajustamos el tamaño del trazo
        trazo_mas_grande = 5
        x1, y1 = (event.x - trazo_mas_grande), (event.y - trazo_mas_grande)
        x2, y2 = (event.x + trazo_mas_grande), (event.y + trazo_mas_grande)

        # Ajustamos el color del trazo
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")

    def capture_image(self):
        # Usamos mss para realizar una captura de pantalla.
        with mss() as sct:
            # Ajustamos la captura de pantalla al lienzo.
            x = self.root.winfo_rootx() + self.canvas.winfo_x()
            y = self.root.winfo_rooty() + self.canvas.winfo_y()
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()

            monitor = {'top': y, 'left': x, 'width': width, 'height': height}

            # Realizamos la captura de pantalla.
            captured_image = np.array(sct.grab(monitor))

        # Guarda la imagen capturada
        # captured_image_path = "captured_image.png"
        # self.save_image(captured_image, captured_image_path)

        # Realiza la inferencia en la imagen utilizando el modelo
        # self.perform_inference(captured_image_path)
        self.perform_inference(captured_image)
    '''
    def save_image(self, image, path):
        # Invierte los colores de la imagen RGB
        inverted_image = np.invert(image)

        # Convierte la imagen a escala de grises
        grayscale_image = ImageOps.grayscale(Image.fromarray(inverted_image))

        # Guarda la imagen
        grayscale_image.save(path)
    '''

    def perform_inference(self, captured_image):
            # Lee la imagen capturada
            # captured_image = Image.open(image_path)

            # Redimensionamos la imagen a un tamaño 28x28 pixeles.
            resized_image = captured_image.resize((28, 28))

            # Guarda la imagen redimensionada (opcional, solo para referencia)
            # resized_image_path = "resized_image.png"
            # self.save_image(np.array(resized_image), resized_image_path)

            # Preprocesamos la imagen, para adaptarla al modelo.
            img_array = np.array(resized_image) 
            img_array = img_array / 255.0  # Normalizamos los píxeles.
            img_array = np.expand_dims(img_array, axis=-1)  # Añadimos una dimensión adicional para el canal de color.
            img_array = np.expand_dims(img_array, axis=0)  # Añadimos una dimensión adicional para el lote.

            # Realizamos las predicciones.
            predictions = self.model.predict(img_array)
            print("Resultado de la inferencia:", predictions)

            # Imprimimos el número del [0-9] según la predicción.
            predicted_number = np.argmax(predictions)
            print("Número predicho:", predicted_number)

            # Sacamos por pantalla el número predicho.
            self.inference_result.set(f"Número predicho: {predicted_number}")

    '''
    def perform_inference(self, image_path):
        # Lee la imagen capturada
        captured_image = Image.open(image_path)

        # Redimensiona la imagen al tamaño esperado por el modelo (28x28)
        resized_image = captured_image.resize((28, 28))

        # Guarda la imagen redimensionada (opcional, solo para referencia)
        resized_image_path = "resized_image.png"
        self.save_image(np.array(resized_image), resized_image_path)

        # Preprocesa la imagen para adaptarla al modelo
        img_array = np.array(resized_image)  # Ajusta al tamaño esperado por el modelo
        img_array = img_array / 255.0  # Normaliza los valores de píxeles a [0,1]
        img_array = np.expand_dims(img_array, axis=-1)  # Añade una dimensión adicional para el canal de color
        img_array = np.expand_dims(img_array, axis=0)  # Añade una dimensión adicional para el lote

        # Imprime el resultado en la consola (puedes ajustar según tu lógica)
        predictions = self.model.predict(img_array)  # Ajusta según tu lógica
        print("Resultado de la inferencia:", predictions)

        # Imprime el número predicho
        predicted_number = np.argmax(predictions)
        print("Número predicho:", predicted_number)

        # Actualiza la etiqueta en la interfaz gráfica
        self.inference_result.set(f"Número predicho: {predicted_number}")

        '''
if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()


