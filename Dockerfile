FROM python:3.9

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    tk \
    libxcb-xinerama0 \
    libfontconfig1 \
    libxkbcommon-x11-0 \
    libxcb-render0 \
    libxcb-shm0 \
    libxcb1 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libice6 \
    x11-xserver-utils

# Limpiar la imagen
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./src/app.py"]

