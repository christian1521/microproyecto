#!/usr/bin/env bash
# Arranque unico para los dos servicios: uvicorn publica la API en /api/v1
# y el tablero estatico en /cite (ambos montados en app/main.py).
set -e

# PORT lo inyecta Railway en tiempo de ejecucion; 8001 es el default local.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
