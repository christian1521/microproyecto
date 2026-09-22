# ---------------------------------------------------------------------------
# Imagen integrada: un solo contenedor sirve los dos servicios del proyecto.
#
#   - API    (app/api.py)   -> /api/v1/...   (FastAPI + SciBERT)
#   - Tablero (app/cite/)   -> /cite/        (montado por app/main.py)
#
# Ambos viven en la carpeta app/ y los expone el mismo proceso uvicorn
# (app.main:app), por lo que no hacen falta dos imagenes ni dos servicios.
# Basado en example-docker-api/Dockerfile y example-docker-dash/Dockerfile.
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# Crear usuario que ejecuta la app (API + tablero)
RUN adduser --disabled-password --gecos '' api-user

# Instalar compiladores C/C++ y build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Definir directorio de trabajo
WORKDIR /opt/microproyecto

# Instalar dependencias
# requirements.txt se copia primero para aprovechar la cache de capas:
# cambiar el codigo de app/ no reinstala torch ni transformers.
COPY ./requirements.txt /opt/microproyecto/requirements.txt
RUN pip install --upgrade pip
# torch desde el indice CPU-only: las ruedas por defecto de PyPI traen CUDA
# (~2.5 GB extra) y la inferencia de SciBERT aqui corre en CPU.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /opt/microproyecto/requirements.txt

# Copiar el codigo de los dos servicios (API + tablero estatico) y el arranque
COPY ./app /opt/microproyecto/app
COPY ./run.sh /opt/microproyecto/run.sh

# El modelo NO se hornea en la imagen: pesa ~440 MB y esta versionado con DVC.
# En Railway se monta aqui un Volume; en local se monta con -v (ver README).
RUN mkdir -p /opt/microproyecto/data

# Hacer el script de arranque ejecutable
RUN chmod +x /opt/microproyecto/run.sh
# Cambiar propiedad de la carpeta a api-user
RUN chown -R api-user:api-user ./

USER api-user

# Variables de la app (Railway inyecta PORT y puede sobreescribir el resto)
ENV PYTHONUNBUFFERED=1 \
    MODEL_DIR=/opt/microproyecto/data/model_scibert_citing_sentences \
    MAX_LENGTH=128 \
    PORT=8001

# Puerto unico para la API y el tablero
EXPOSE 8001

# Comandos a ejecutar al correr el contenedor
CMD ["bash", "./run.sh"]
