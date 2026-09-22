import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api import api_router
from app.config import settings, setup_app_logging

setup_app_logging(config=settings)


app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

root_router = APIRouter()

@root_router.get("/")
def index(request: Request) -> Any:
    body = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CiteAnalyzer - Inicio</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f8f9fa;
                color: #212529;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                background-color: #ffffff;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                text-align: center;
                max-width: 600px;
                width: 90%;
            }
            h1 { color: #2c3e50; margin-bottom: 5px; }
            h2 { color: #34495e; font-size: 1.1rem; font-weight: normal; margin-top: 0; margin-bottom: 25px; }
            .info {
                color: #6c757d;
                font-size: 1rem;
                line-height: 1.6;
                margin-bottom: 35px;
            }
            .buttons {
                display: flex;
                justify-content: center;
                gap: 20px;
                flex-wrap: wrap;
            }
            .btn {
                display: inline-flex;
                align-items: center;
                gap: 10px;
                text-decoration: none;
                padding: 12px 25px;
                border-radius: 6px;
                color: white;
                font-weight: bold;
                transition: background-color 0.3s, transform 0.2s;
            }
            .btn-icon {
                width: 22px;
                height: 22px;
            }
            .btn:hover { transform: translateY(-2px); }
            .btn-api { background-color: #3b82f6; }
            .btn-api:hover { background-color: #2563eb; }
            .btn-dash { background-color: #10b981; }
            .btn-dash:hover { background-color: #059669; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CiteAnalyzer</h1>
            <h2>Clasificación de la función de cita (Citation Function Classification)</h2>
            
            <div class="info">
                <strong>Grupo 24 - Proyecto Desarrollo de Soluciones</strong><br>
                Christian Alberto Torres Manigua<br>
                Fredy Alexander Gamez Rodriguez
            </div>
            
            <div class="buttons">
                <a href="/docs" class="btn btn-api">
                    <!-- Icono SVG integrado para el API -->
                    <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="16 18 22 12 16 6"></polyline>
                        <polyline points="8 6 2 12 8 18"></polyline>
                    </svg>
                    Abrir Documentación del API
                </a>
                <a href="/cite/" class="btn btn-dash">
                    <!-- Imagen favicon.svg para el tablero -->
                    <img src="favicon.svg" alt="Icono Tablero" class="btn-icon">
                    Abrir el Tablero (Aplicación)
                </a>
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=body)


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

BASE_DIR = Path(__file__).resolve()
CITE_DIR = BASE_DIR.parent / "cite"

if CITE_DIR.exists():
    app.mount("/cite", StaticFiles(directory=CITE_DIR, html=True), name="cite_app")
else:
    logger.warning(f"El directorio '{CITE_DIR}' no existe. La aplicación frontend no se montará.")


if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


if __name__ == "__main__":
    logger.warning("Running in development mode. Do not run like this in production.")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")