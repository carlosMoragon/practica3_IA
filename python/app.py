import tkinter as tk
from PIL import ImageGrab, Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paint App")

        self.canvas = tk.Canvas(root, bg="white", width=400, height=400)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        # Configura el evento del ratón
        self.canvas.bind("<B1-Motion>", self.paint)

        # Agrega un botón para capturar la imagen
        capture_button = tk.Button(root, text="Capturar Imagen", command=self.capture_image)
        capture_button.pack(side=tk.BOTTOM)

        # Variable para almacenar la imagen capturada
        self.captured_image = None

        # Cargar el modelo desde el archivo .h5
        self.model = tf.keras.models.load_model("modelo.h5")

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=2)

    def capture_image(self):
        # Captura la imagen actual del Canvas
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        captured_image = ImageGrab.grab(bbox=(x, y, x1, y1))

        # Procesar la imagen y realizar la clasificación
        processed_image = self.process_image(captured_image)
        prediction = self.classify_image(processed_image)

        # Mostrar la predicción en la consola
        print(f"Predicción: {prediction}")

    def process_image(self, image):
        # Convertir la imagen capturada a un formato adecuado para el modelo
        image = image.resize((28, 28))  # Ajustar al tamaño esperado por el modelo
        image = image.convert("L")  # Convertir a escala de grises
        image_array = np.array(image) / 255.0  # Normalizar los valores de píxeles
        image_array = image_array.reshape((1, 28, 28, 1))  # Ajustar la forma para el modelo
        return image_array

    def classify_image(self, processed_image):
        # Realizar la clasificación utilizando el modelo
        prediction = self.model.predict(processed_image)
        # Obtener la clase predicha (índice con la probabilidad más alta)
        predicted_class = np.argmax(prediction)
        return predicted_class

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()
