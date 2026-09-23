# Retos y evidencias de la entrega final

**Grupo 24**: Christian Alberto Torres Manigua / Fredy Alexander Gamez Rodriguez

> [!NOTE]
> Esta bitácora reúne las evidencias y los retos técnicos que se presentaron al probar los modelos, al construir la imagen Docker y al desplegarla en Railway. No es un instructivo. Los pasos oficiales están en el [Manual de instalación](manual-instalacion.md).

## Tabla de contenido

1. [Pruebas de modelos en MLflow](#1-pruebas-de-modelos-en-mlflow)
2. [Pruebas locales del contenedor Docker](#2-pruebas-locales-del-contenedor-docker)
3. [Primer despliegue en Railway y asignación del puerto](#3-primer-despliegue-en-railway-y-asignación-del-puerto)
4. [Construcción de la imagen con los modelos](#4-construcción-de-la-imagen-con-los-modelos)
5. [Publicación en DockerHub y despliegue desde la imagen](#5-publicación-en-dockerhub-y-despliegue-desde-la-imagen)
6. [Soporte de varios modelos y pruebas funcionales](#6-soporte-de-varios-modelos-y-pruebas-funcionales)
7. [Versiones finales](#7-versiones-finales)

---

## 1. Pruebas de modelos en MLflow

Se entrenaron de nuevo los dos *encoders* que usa el tablero con la misma semilla (`seed42`), y ambos quedaron registrados en el servidor MLflow de AWS EC2. SciBERT (`scibert-citfunc-seed42`) llegó a **0,883 de F1 macro** y **0,907 de exactitud**. BERT (`bert-citfunc-seed42`) llegó a **0,828** y **0,862**. Con esto se confirma lo que se vio en la Entrega 2: un *encoder* especializado en
literatura científica rinde más.

#### Justificación de resultados de metricas

Debido a que el dataset proviene de un proceso de etiquetado automatizado, este tiende a producir fronteras de decisión más regulares y aprendibles, precisamente porque el propio proceso que generó las etiquetas probablemente se apoyó en las mismas señales superficiales (léxicas, sintácticas o de patrones de frase) que un modelo de lenguaje preentrenado como BERT o SciBERT también es capaz de capturar con alta fidelidad. Esto no invalida el resultado, pero sí exige matizarlo: el valor de F1-macro refleja qué tan bien el modelo aproxima el criterio automatizado de etiquetado, no necesariamente qué tan bien reproduciría el juicio de un experto humano sobre la función retórica real de la cita.



![Comparación de runs BERT vs SciBERT en MLflow](images/image-20260921224026517.png)

La vista comparativa reúne 19 corridas de 5 experimentos: TF-IDF, *encoders* congelados, Mistral, Qwen y *fine-tuning*.

![Comparación de 19 runs en MLflow](images/image-20260921225325456.png)

## 2. Pruebas locales del contenedor Docker

Antes de desplegar, la imagen se probó en local. Primero se identificó el ID de la imagen, luego se
ejecutó el contenedor y por último se verificó el acceso en `localhost:8001`.

![Identificando la imagen](images/image-20260922100708842.png)

![Ejecutando el contenedor](images/image-20260922100643792.png)

![Logs de arranque](images/image-20260922101002049.png)

![Acceso local 1](images/image-20260922101043670.png)

![Acceso local 2](images/image-20260922101109036.png)

## 3. Primer despliegue en Railway y asignación del puerto

Se escogió **Railway (PaaS) con DockerHub** en lugar de AWS ECR/ECS Fargate, porque es más fácil de
configurar y los costos son bajos.

![Despliegue en Railway](images/image-20260922003405556.png)

![Configuración en Railway](images/image-20260922003423547.png)

> [!CAUTION]
> **Reto:** el primer despliegue falló. Según los logs, Railway no lograba enrutar el tráfico hacia el puerto donde escuchaba Uvicorn.

![Error de puerto](images/error2.png)

Se reintentó con otra configuración de puerto y el despliegue funcionó:

![Reintento](images/image-20260922011311749.png)

![Despliegue con ajustes](images/image-20260922011916731.png)

**Solución robusta:** `run.sh` ahora lee `PORT` y usa `8080` si no está definido (`--port "${PORT:-8080}"`), y en Railway se fijó `PORT=8080`.

![Puerto 8080](images/image-20260922012428640.png)

## 4. Construcción de la imagen con los modelos

Los modelos (~440 MB cada uno) se incluyeron en la imagen, en `/data`, para que el contenedor no dependa de DVC ni de credenciales AWS al arrancar.

![Imagen con el modelo en /data](images/image-20260922023735855.png)

> [!CAUTION]
> **Reto:** `docker build` no copiaba los modelos porque `data/` estaba en `.dockerignore`. **Solución:** en `.dockerignore` ahora solo se excluyen los punteros `.dvc` y los notebooks de `data/`, así que la carpeta de modelos entra a la imagen. Además, torch se instala desde el índice CPU. Así se evitan unos 2,5 GB de ruedas CUDA.

![Rechazo por .dockerignore](images/image-20260922023648035.png)

![Build de la imagen](images/image-20260922023951875.png)

## 5. Publicación en DockerHub y despliegue desde la imagen

![docker push](images/image-20260922024810474.png)

![Imagen en DockerHub](images/image-20260922025009254.png)

![Desplegando desde DockerHub](images/image-20260922025412386.png)

Railway asignó la URL pública <https://cite-api-production.up.railway.app>, y se revisaron los pasos del despliegue:

![Verificación del despliegue](images/image-20260922033311578.png)

## 6. Soporte de varios modelos y pruebas funcionales

Se cambiaron el `Dockerfile` y `app/api.py` para que las rutas de los modelos se resuelvan en relación con el directorio de instalación. Así se sirven SciBERT y BERT al mismo tiempo, y el parámetro `type_model` escoge cuál usar. En el historial de Railway quedan una versión fallida (`v0.3`) y la corrección (`v0.4`).

![Historial de despliegues en Railway](images/image-20260922121522327.png)

Se compararon varios ejemplos en el tablero ya desplegado:

![Ejemplo Gap](images/image-20260922124049583.png)

![Ejemplo 2](images/image-20260922124253377.png)

## 7. Versiones finales

Se agregó una página de inicio con accesos al API y al tablero, y se publicaron varias versiones con ajustes menores hasta llegar a `v0.9`.

![Página de inicio](images/image-20260922135617600.png)

![Versiones en DockerHub](images/image-20260922140818622.png)

![Actualización de versiones](images/image-20260922141002592.png)

## Lecciones aprendidas

- En plataformas PaaS, el puerto lo define la plataforma: el servicio debe leer `PORT`.
- `.dockerignore` puede dejar por fuera artefactos que la imagen necesita. Es necesario verificarlo cuando en la contruccion del build se asume que se copia algo que no está presente.
- Usar torch CPU reduce mucho el tamaño de la imagen cuando la inferencia no necesita GPU.
- Usar *tags* versionados en DockerHub permite volver a una versión anterior en Railway sin reconstruir la imagen.

[Volver al inicio](index.md)
