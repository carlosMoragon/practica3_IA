import tkinter as tk
from PIL import ImageGrab

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

        self.captured_image = ImageGrab.grab(bbox=(x, y, x1, y1))

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()
