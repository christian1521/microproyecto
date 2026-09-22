# CiteAnalyzer - Clasificación de la función de cita

Proyecto Desarrollo de Soluciones - MAIA Uniandes - **Grupo 24**

```ini
Christian Alberto Torres Manigua - ca.torres@uniandes.edu.co
Fredy Alexander Gamez Rodriguez - fredygamez@uniandes.edu.co
```

Cuando se escribe un artículo científico, no siempre queda claro para qué se cita un trabajo previo. 

El proyecto **CiteAnalyzer** responde esta pregunta: 

>  ***¿Cómo clasificar automáticamente el propósito de una cita usando modelos de lenguaje (NLP)?***
>
>  Para ello, asigna a cada contexto de cita una de nueve funciones: `Background`, `Gap`, `Basis`, `Comparison`, `Application`, `Improvement/Modification`, `Evidence`, `Identification of the Originator` y `Further Reading`.

Se incorporan estos componentes para crear una solución integral para uso de usuario final:

- **Datos:** Dataset SDCF (arXiv) con 707.225 oraciones citantes después de limpieza. Versionado con DVC en AWS-S3.
- **Modelos:** SciBERT y BERT con *fine-tuning*, con experimentos registrados en MLflow. 
- **Solución Web Interactiva:** API FastAPI y Tablero Web desplegados con Docker via Railway.

| Recurso | URL |
| --- | --- |
| Tablero | <https://cite-api-production.up.railway.app/cite/> |
| API | <https://cite-api-production.up.railway.app/docs#> |
| Imagen | `docker pull fredygamez/cite-api:v0.9` |
| Código | <https://github.com/christian1521/microproyecto> |

### Inicio rápido

```bash
docker pull fredygamez/cite-api:v0.9
docker run -d -p 8001:8001 -e PORT=8001 fredygamez/cite-api:v0.9
# http://localhost:8001/cite/
```

### Documentación y manuales

1. [Indice](index.md)  
2. [Manual de usuario](manual-usuario.md) 
3. [Manual de instalación](manual-instalacion.md) 
4. [Vista de desarrollo](desarrollo.md) 
5. [Retos](retos.md)



---

Enlaces: [Indice](index.md) · [Manual de usuario](manual-usuario.md) · [Manual de instalación](manual-instalacion.md) · [Vista de desarrollo](desarrollo.md) · [Retos](retos.md)

