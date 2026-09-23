# Vista de desarrollo

## Estructura del repositorio

Repositorio: <https://github.com/christian1521/microproyecto>

```apl
https://github.com/christian1521/microproyecto
```

```
.
├── .dvc/config                    # Remoto DVC: s3://christian1521-dvcstore
├── app/                           # API + tablero (un solo servicio)
│   ├── main.py                    # FastAPI: página de inicio, router /api/v1, monta /cite
│   ├── api.py                     # Endpoints, catálogo de funciones, carga e inferencia de modelos
│   ├── config.py                  # Settings (/api/v1, CORS, logging)
│   ├── cite/                      # Tablero web
│   │   ├── index.html
│   │   └── favicon.svg
│   └── tests/                     # Pruebas pytest (conftest.py, test_api.py)
├── data/
│   ├── final_labelled_citing_sentences_all.csv.dvc # DVC dataset entrenamiento SCDF
│   ├── model_scibert_citing_sentences.dvc          # DVC del modelo SciBERT (~440 MB)
│   └── model_bert_citing_sentences.dvc             # DVC del modelo BERT (~439 MB)
├── experimentos-mlflow/           # Scripts de experimentos registrados en MLflow
│   ├── scibert_finetuning_v1.py … v4.py
│   ├── experimento_BERT.py
│   ├── experimento_MISTRAL.py
│   └── experimento_QWEN.py
├── dist/                          # Paquete distribuible (.whl, .tar.gz)
├── Dockerfile                     # Imagen única API + tablero (python:3.12)
├── .dockerignore
├── run.sh                         # Arranque Docker con uvicorn app.main:app ${PORT}
├── railway.json                   # Build/deploy en Railway (healthcheck, reintentos)
├── Procfile                       # Arranque estilo PaaS
├── pyproject.toml                 # Paquete citing-sentences-api
├── requirements.txt, test_requirements.txt, typing_requirements.txt
├── tox.ini, mypy.ini
└── README.md
```





## Diagrama de componentes

![diagramacomponentes](./images/diagramacomponentes.png)



> **Esquema de desarrollo y despliegue del servicio `\cite-api\`:**
>
> El diagrama ilustra el flujo de desarrollo, despliegue y operación del servicio *cite-api*, alojado en **Railway**. El usuario interactúa vía **HTTPS** con un contenedor Docker (`fredygamez/cite-api:v0.9`), que ejecuta una aplicación **FastAPI** con **Uvicorn**. Esta expone un *tablero* (`/cite/`) y una **API REST** (`/api/v1`), que consume dos modelos de lenguaje *fine-tuned*: **SciBERT** y **BERT**. El tablero recupera datos en formato JSON desde la API.
>
> El código fuente y la configuración residen en un repositorio **GitHub** (`christian1521/microproyecto`), donde se construye la imagen Docker y se sube a **DockerHub**. Los modelos y datasets se gestionan con **DVC** en un bucket de **AWS S3** (`christian1521-dvcstore`), mientras que el registro de experimentos de *MLOps* se realiza con **MLFlow** en una instancia **AWS EC2**. Este esquema garantiza un despliegue escalable y un desarrollo colaborativo.



## Diagrama de secuencia del uso

![](./images/diagramasecuencia.png)

> **Esquema de uso de la aplicación para usuarios finales:**
>
> Este diagrama de secuencias describe el flujo de interacción del usuario con la aplicación *cite-api*. El proceso inicia cuando el **usuario** ingresa un texto (y opcionalmente un párrafo citado) en la interfaz **Web**. Luego, selecciona un modelo de clasificación y hace clic en el botón **"Analizar Cita"**.
>
> La aplicación **Web** envía una consulta a la **API**, especificando el modelo seleccionado. La **API** procesa la solicitud y la reenvía al **Modelo** correspondiente, que analiza el texto y devuelve un **porcentaje de clasificación**. La **API** traduce este resultado en una **descripción de categoría** y lo envía de vuelta a la interfaz **Web**, que finalmente muestra los resultados al usuario. Este flujo garantiza una experiencia intuitiva y eficiente para el análisis de citas.

## Stack tecnológico

| Capa | Tecnología |
| --- | --- |
| Lenguaje | Python 3.12 (imagen); ≥ 3.9 (paquete) |
| Modelos | Transformers ≥ 4.46, PyTorch ≥ 2.13 (CPU) |
| API | FastAPI ≥ 0.115, Uvicorn ≥ 0.32, Pydantic 2 |
| Tablero | HTML + CSS + JavaScript, servido con `StaticFiles` |
| Experimentos | MLflow en AWS EC2 (puerto 8050) |
| Versionado de datos | DVC + S3 |
| Contenedor / registro | Docker / DockerHub |
| Despliegue | Railway (PaaS) |
| Pruebas | pytest vía tox |

## Modelos servidos

| Modelo | Run en MLflow | F1 macro | Exactitud |
| --- | --- | --- | --- |
| SciBERT (`allenai/scibert_scivocab_uncased`) fine-tuned | `scibert-citfunc-seed42` | 0,883 | 0,907 |
| BERT fine-tuned | `bert-citfunc-seed42` | 0,828 | 0,862 |

Los modelos se cargan de forma diferida en la primera petición. La entrada se trunca a `MAX_LENGTH=128`
tokens. Si llega `cited_paragraphs`, el texto de entrada se arma así:
`"<citing_sentence> CONTEXT: In the text, the [CITATION] tag refers to: <cited_paragraphs>"`.

## Flujo de trabajo

1. Los datos y los modelos se versionan con DVC en S3, y el código se versiona en GitHub.
2. Los experimentos se registran en MLflow.
3. La imagen se construye localmente con los modelos en `data/` y se publica en DockerHub con *tags*
   `v0.x`.
4. Railway despliega el *tag* elegido.

[Volver al inicio](index.md)
