# CiteAnalyzer: clasificación de la función de cita

**Proyecto Desarrollo de Soluciones - MAIA Uniandes **

## Grupo 24 - Entrega 3 

```
Christian Alberto Torres Manigua 
Fredy Alexander Gamez Rodriguez
```

**CiteAnalyzer** clasifica automáticamente el propósito discursivo de una cita académica en una de 
**9 categorías**: `Background`, `Gap`, `Basis`, `Comparison`, `Application`, `Improvement/Modification`,
`Evidence`, `Identification of the Originator` y `Further Reading`. Uso de modelos *encoder* (BERT y SciBERT) con *fine-tuning*. Se despliega con un API FastAPI y un tablero web, en un mismo contenedor Docker desplegado via Railway.

## Accesos al prototipo desplegado

| Recurso | URL |
| --- | --- |
| Página de inicio | <https://cite-api-production.up.railway.app/> |
| Tablero (aplicación) | <https://cite-api-production.up.railway.app/cite/> |
| API | <https://cite-api-production.up.railway.app/docs/> |
| Imagen Docker | <https://hub.docker.com/repository/docker/fredygamez/cite-api/tags> (`fredygamez/cite-api:v0.9`) |
| Repositorio código en Github | <https://github.com/christian1521/microproyecto> <br>[Ver detalles](desarrollo.md) |

## Manuales

- [Manual de usuario del tablero](manual-usuario.md): cómo analizar una cita e interpretar el resultado.
- [Manual de instalación](manual-instalacion.md): instalación con Docker (DockerHub), desde el código fuente y despliegue en Railway.

## Documentación complementaria

- [Vista de desarrollo](desarrollo.md): estructura del repositorio, stack tecnológico, componentes y
  flujo de uso.
- [Retos y evidencias](retos.md): bitácora de pruebas de modelos, construcción de la imagen Docker y
  despliegue en Railway.

## Stack tecnológico

| Categoría | Tecnología |
| --- | --- |
| Modelos | BERT y SciBERT con *fine-tuning* |
| Experimentos | MLflow en AWS EC2 |
| Datos y modelos versionados | DVC con remoto AWS-S3 (`s3://christian1521-dvcstore`) |
| API | FastAPI + Uvicorn (`/api/v1`) |
| Tablero | HTML/JS por FastAPI (`/cite/`) |
| Contenedor | Docker (`python:3.12-slim`), una imagen para API y Tablero |
| Registro de imágenes | DockerHub (`fredygamez/cite-api`) |
| Despliegue (PaaS) | Railway |
| Código fuente | Repositorio GitHub |
