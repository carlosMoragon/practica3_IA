import os
import csv

numbers = list(range(10))
directory = ".\\trainingSet\\"

paths = list(map(lambda x: directory + str(x), numbers))

# print(paths)


image_names = []

for path in paths:
    for names in os.listdir(path):
        image_names.append(path + "\\" + names)

# Imprimir la lista de nombres de archivos
# print(len(image_names))


# Ejemplo de construcción de un dataset simple
dataset = []

for idx, name in enumerate(image_names, start=1):

    data_entry = {
        "path": name,
        "number": int(name.split("\\trainingSet\\")[1][0]),  # Convertir el número a entero
        "id": idx  # Utilizar el índice como identificador
    }

    dataset.append(data_entry)


# Imprimir el dataset
# print("Dataset construido:")
# print(dataset)

csv_file_path = "dataset.csv"

# Escribir el dataset en un archivo CSV
with open(csv_file_path, mode='w', newline='') as csv_file:
    fieldnames = ["id", "number", "path"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    # Escribir la fila de encabezado
    writer.writeheader()

    # Escribir cada entrada del dataset en el archivo CSV
    for entry in dataset:
        writer.writerow(entry)

print(f"El dataset ha sido guardado en {csv_file_path}.")


