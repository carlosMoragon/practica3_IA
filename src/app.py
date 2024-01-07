import tkinter as tk
from PIL import Image, ImageOps  
import numpy as np
from tensorflow.keras.models import load_model
import os
from mss import mss

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paint App")

        # Declaramos el color de fondo, y las dimensiones de la pantalla.
        self.canvas = tk.Canvas(root, bg="black", width=400, height=400)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
    
        # Configuramos el evento del ratón
        self.canvas.bind("<B1-Motion>", self.paint)

        # Agregamos un botón para capturar la imagen
        capture_button = tk.Button(root, text="Capturar Imagen", command=self.capture_image)
        capture_button.pack(side=tk.BOTTOM)

        # Agregamos un botón para borrar el dibujo
        clear_button = tk.Button(root, text="Borrar", command=self.clear_canvas)
        clear_button.pack(side=tk.BOTTOM)

        # Declaramos una variable para almacenar el resultado de la inferencia
        self.inference_result = tk.StringVar()
        result_label = tk.Label(root, textvariable=self.inference_result, fg="white", bg="black")
        result_label.pack(side=tk.BOTTOM)

        # Cargamos el modelo previamente entrenado
        self.model = load_model('./src/modelo.h5')

    def paint(self, event):
        # Ajustamos el tamaño del trazo
        trazo_mas_grande = 5
        x1, y1 = (event.x - trazo_mas_grande), (event.y - trazo_mas_grande)
        x2, y2 = (event.x + trazo_mas_grande), (event.y + trazo_mas_grande)

        # Declaramos el color para pintar
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")

    def capture_image(self):
        # Capturamos la imagen de la ventana de la aplicación usando mss
        with mss() as sct:
            x = self.root.winfo_rootx() + self.canvas.winfo_x()
            y = self.root.winfo_rooty() + self.canvas.winfo_y()
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()

            # Definimos la región de captura como una tupla
            monitor = {'top': y, 'left': x, 'width': width, 'height': height}

            # Realizamos la captura a la imagen
            captured_image = np.array(sct.grab(monitor))

        # Guardamos la imagen capturada
        captured_image_path = "captured_image.png"
        self.save_image(captured_image, captured_image_path)

        # Evaluamos la imagen en el modelo
        self.perform_inference(captured_image_path)

    def save_image(self, image, path):
        # Invierte los colores de la imagen RGB
        inverted_image = np.invert(image)

        # Convierte la imagen a escala de grises
        grayscale_image = ImageOps.grayscale(Image.fromarray(inverted_image))

        # Guarda la imagen
        grayscale_image.save(path)

    def perform_inference(self, image_path):
        # Leemos la imagen capturada
        captured_image = Image.open(image_path)

        # Redimensionamos la imagen al tamaño esperado por el modelo (28x28)
        resized_image = captured_image.resize((28, 28))

        # Guarda la imagen redimensionada
        # resized_image_path = "resized_image.png"
        # self.save_image(np.array(resized_image), resized_image_path)

        # Preprocesamos la imagen para adaptarla al modelo
        img_array = np.array(resized_image)  
        img_array = img_array / 255.0  # Normalizamos los pixeles a [0,1]
        img_array = np.expand_dims(img_array, axis=-1)  # Añade una dimensión adicional para el canal de color
        img_array = np.expand_dims(img_array, axis=0)  # Añade una dimensión adicional para el lote

        # Guardamos las predicciones
        predictions = self.model.predict(img_array)
        print("Resultado de la inferencia:", predictions)

        # Asignamos un numero del 0-9 según las predicciones
        predicted_number = np.argmax(predictions)
        print("Número predicho:", predicted_number)

        # Actualizamos la interfaz gráfica
        self.inference_result.set(f"Número predicho: {predicted_number}")

    def clear_canvas(self):
        # Borra todo el contenido del lienzo
        self.canvas.delete("all")

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()

