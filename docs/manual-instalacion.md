# Manual de instalación

El API y el tablero se ejecutan en el **mismo proceso Uvicorn** y en la **misma imagen Docker**. El
API responde en `/api/v1` y el tablero en `/cite/`. Hay tres formas de instalarlos. La opción A es la
más sencilla y la que se recomienda.

## Requisitos previos

| **OPCIONES** | **Docker ≥ 24** | **Python ≥ 3.9** | **Git** | **DVC con S3 y credenciales AWS** | **Cuenta  Railway** | **RAM disponible** |
| --- | --- | --- | --- | --- | --- | --- |
| **Opción A (DockerHub, localhost)** | ✅️ | - | - | - | - | ~3 GB (dos modelos de ~440 MB) |
| **Opción B1 (código fuente, venv)** | - | ✅️ | opcional | ✅️ | - | ~3 GB |
| **Opción B2 (código fuente, Dockerfile)** | ✅️ | ✅️ ** | opcional | ✅️ | - | ~3 GB |
| **Opción C (Railway)** | - | - | - | - | ✅️ | plan con ≥ 3 GB |

`**`  Python solo se usa para instalar DVC (`pip install "dvc[s3]"`) y bajar los modelos. El modelo corre dentro del contenedor.




## Opción A - Ejecutar la imagen publicada en DockerHub

La imagen ya trae el código, las dependencias (PyTorch CPU) y los dos modelos con *fine-tuning*.
Las versiones publicadas se pueden ver en <https://hub.docker.com/repository/docker/fredygamez/cite-api/tags>. La más reciente es `v0.9`.

```bash
# 1. Descargar la imagen
docker pull fredygamez/cite-api:v0.9

# 2. Ejecutar el contenedor (el servicio escucha en la variable PORT)
docker run -d --name cite-api -p 8001:8001 -e PORT=8001 fredygamez/cite-api:v0.9

# 3. Revisar los logs hasta que aparezca "Uvicorn running on http://0.0.0.0:8001"
docker logs -f cite-api
```

Accesos locales:

- Inicio: <http://localhost:8001/>
- Tablero: <http://localhost:8001/cite/>
- API: <http://localhost:8001/docs>

## Opción B - Instalar desde el código fuente

La versión estable del código es el release [**v0.2**](https://github.com/christian1521/microproyecto/releases/tag/v0.2). En su sección *Assets* se puede descargar el **Source code** (zip o tar.gz). El código incluye el `Dockerfile`, el
`.dockerignore` y el `run.sh`, así que hay dos formas de ejecutarlo:

- **B.1 - Entorno virtual de Python:** Uvicorn se ejecuta directamente en la máquina.
- **B.2 - Contenedor Docker:** se construye una imagen propia con el `Dockerfile` del repositorio.

Los pasos 1 y 2 son iguales para las dos formas.

### Paso 1 - Obtener el código fuente (release v0.2)

Use alguna de estas dos formas.

**a) Descargar y descomprimir el Source code (zip):**

```bash
curl -L -o microproyecto-v0.2.zip \
https://github.com/christian1521/microproyecto/archive/refs/tags/v0.2.zip
unzip microproyecto-v0.2.zip
cd microproyecto-0.2
```

Para el tar.gz, la URL termina en `v0.2.tar.gz` y se descomprime con `tar -xzf microproyecto-v0.2.tar.gz`.

**b) Clonar con Git directamente en el tag v0.2:**

```bash
git clone --branch v0.2 --depth 1 https://github.com/christian1521/microproyecto.git
cd microproyecto
```

### Paso 2 - Descargar los modelos versionados con DVC

Los modelos (~440 MB cada uno) no están en Git. Se descargan del remoto S3 con DVC y quedan en `data/`. Las dos formas (B.1 y B.2) los necesitan.

Instale DVC y configure las credenciales de AWS. En lugar de `aws configure`, también puede exportar `AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY`:

```bash
pip install "dvc[s3]"
aws configure
```

Solo si usó la opción **a)** (zip o tar.gz), ejecute además:

```bash
dvc config --local core.no_scm true
```

Descargue los modelos:

```bash
dvc pull data/model_scibert_citing_sentences.dvc data/model_bert_citing_sentences.dvc
```

> [!NOTE]
> El *Source code* (opción a) no trae la carpeta `.git`, y DVC necesita un repositorio Git para funcionar. Por eso, con esa opción hay que ejecutar `dvc config --local core.no_scm true` antes de `dvc pull`. Con la opción b (`git clone`) no hace falta.

### B.1 - Ejecutar con un entorno virtual de Python

#### Paso 3 - Crear y activar un entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

En Windows se activa con `.venv\Scripts\activate`.

#### Paso 4 - Instalar las dependencias

Torch se instala en su versión CPU:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

#### Paso 5 - Levantar el API y el tablero

```bash
bash run.sh
```

`run.sh` usa la variable `PORT`, o `8080` si no está definida. Otra forma de levantarlo:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### B.2 - Construir y ejecutar la imagen con el Dockerfile del repositorio

Esta forma usa los modelos que el paso 2 dejó en `data/`. El `Dockerfile` construye una sola imagen
para el API y el tablero, usa `python:3.12-slim` e instala torch en su versión CPU. No hace falta
instalar Python ni dependencias en la máquina.

#### Paso 3 - Construir la imagen

```bash
docker build -t cite-api:local .
```

#### Paso 4 - Ejecutar el contenedor

```bash
docker run -d --name cite-api-local -p 8001:8001 -e PORT=8001 cite-api:local
```

#### Paso 5 - Revisar los logs

Espere hasta que aparezca `Uvicorn running on http://0.0.0.0:8001`:

```bash
docker logs -f cite-api-local
```

> [!NOTE]
> La imagen copia `data/` con los modelos. Por eso `data/` **no** puede estar excluida en `.dockerignore`: ahí solo se excluyen los punteros `.dvc` y los notebooks. Ver [Retos](retos.md#4-construcción-de-la-imagen-con-los-modelos).

#### Paso 6 (opcional) - Publicar una nueva versión en DockerHub

```bash
docker login
docker tag cite-api:local fredygamez/cite-api:v0.10
docker push fredygamez/cite-api:v0.10
```

### Accesos locales

Con cualquiera de las dos formas:

- Inicio: <http://localhost:8001/>
- Tablero: <http://localhost:8001/cite/>
- API: <http://localhost:8001/docs>

Si en B.1 usó `run.sh` sin definir `PORT`, cambie `8001` por `8080`.


## Opción C - Desplegar en Railway (PaaS)

1. En Railway, crear un proyecto y agregar un servicio con **Deploy → Docker Image**.
2. Escribir la imagen `fredygamez/cite-api:v0.9`.
3. En **Variables**, definir `PORT=8080`. Opcionalmente, definir `MAX_LENGTH=128`.
4. En **Settings → Networking**, generar un dominio público apuntando al puerto `8080`.
5. Para actualizar, cambiar el *tag* de la imagen y volver a desplegar. Railway conserva el historial
   de despliegues.

El repositorio incluye además `railway.json`, que sirve para construir desde el `Dockerfile`, con
*healthcheck* en `/` y reintentos automáticos en caso de falla.

Despliegue actual: <https://cite-api-production.up.railway.app/>

## Verificar la instalación

```bash
curl http://localhost:8001/api/v1/health
# {"status":"ok","device":"cpu","model_version":"..."}

curl -X POST http://localhost:8001/api/v1/classify-citation \
  -H "Content-Type: application/json" \
  -d '{"citing_sentence":"Similar to the method proposed in [CITATION], we adopt a transformer-based architecture.","type_model":"scibert"}'
```

Si `health` devuelve `"status":"degraded"`, el servicio no encontró los modelos. Revise que
`data/model_scibert_citing_sentences/` y `data/model_bert_citing_sentences/` contengan `config.json`.

## Variables de entorno

| Variable | Propósito | Valor por defecto |
| --- | --- | --- |
| `PORT` | Puerto de Uvicorn (Railway lo inyecta) | `8001` en la imagen, `8080` en `run.sh` |
| `MAX_LENGTH` | Longitud máxima de tokens de la entrada | `128` |

## Pruebas automatizadas

```bash
pip install tox
tox -e test_app        # pytest sobre app/tests/
```

> [!CAUTION]
> Verificar que las credenciales de AWS y las llaves `.pem` **nunca** se encuentren en repositorio Git.

[Volver al inicio](index.md)

