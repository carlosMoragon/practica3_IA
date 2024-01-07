#!/bin/bash

# Nombre de la imagen en Docker Hub
DOCKER_IMAGE="cmoragon15/practica3_ia:v1"

# Guardar el valor actual de DISPLAY
ORIGINAL_DISPLAY=$DISPLAY

# Descargar la imagen desde Docker Hub
docker pull $DOCKER_IMAGE

# Comprobar si el servidor X11 está en ejecución
if [ -z "$DISPLAY" ]; then
    echo "Error: El servidor X11 no está en ejecución."
    exit 1
fi

# Configurar las autorizaciones X11
xhost +

# Ejecutar el contenedor Docker
sudo docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --device /dev/snd \
    $DOCKER_IMAGE

# Restaurar las configuraciones originales de DISPLAY
xhost -
export DISPLAY=$ORIGINAL_DISPLAY

echo "Configuraciones de DISPLAY restauradas."

