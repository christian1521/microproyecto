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
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Citation Function Classification API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "<div style='margin-top: 15px; font-weight: bold;'>"
        "Citation Function Classification: <a href='/cite/'>Open App</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)


app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(root_router)

CITE_DIR = Path("cite")

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