import os
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import __version__

logger = logging.getLogger("citation_api")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./data/model_scibert_citing_sentences"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))

# --------------------------------------------------------------------
# Estado del modelo: se carga de forma perezosa (lazy), no al importar
# este módulo. Esto evita que 'import app.api' obligue a tener torch y
# transformers instalados solo para poder probar la estructura de la API.
# --------------------------------------------------------------------
_tokenizer = None
_model = None
_id2label: Optional[dict] = None
_device: str = "cpu"


def _load_model_if_needed():
    """Carga el modelo real la primera vez que se necesita. Los tests
    pueden monkeypatchear esta función para evitar requerir torch."""
    global _tokenizer, _model, _id2label, _device

    if _model is not None:
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(f"No se encontró 'config.json' en {MODEL_DIR}.")

    logger.info(f"Cargando modelo desde: {MODEL_DIR}")
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    _model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    _model.to(_device)
    _model.eval()

    label_mapping_path = MODEL_DIR / "label_mapping.json"
    if label_mapping_path.exists():
        with open(label_mapping_path, "r", encoding="utf-8") as f:
            _id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}
    else:
        _id2label = {int(k): v for k, v in _model.config.id2label.items()}

    logger.info(f"Modelo cargado en dispositivo: {_device}")
    logger.info(f"Etiquetas: {list(_id2label.values())}")


def predict(citing_sentence: str, cited_paragraphs: Optional[list] = None) -> dict:
    """
    Punto único de inferencia. Encapsulada como función independiente
    (en vez de código inline dentro del endpoint) para que los tests puedan
    reemplazarla directamente vía monkeypatch, sin necesitar torch/transformers
    instalados ni el modelo real presente.
    """
    _load_model_if_needed()

    import torch  # ya cargado por _load_model_if_needed(); solo referencia local

    inputs = _tokenizer(
        citing_sentence,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

    pred_id = int(torch.argmax(probs).item())

    return {
        "predicted_label": _id2label[pred_id],
        "confidence": float(probs[pred_id].item()),
        "used_context": bool(cited_paragraphs),
        "all_probabilities": {
            _id2label[i]: float(probs[i].item()) for i in range(len(_id2label))
        },
    }


# --------------------------------------------------------------------
# Esquemas de request
# --------------------------------------------------------------------
class CitationRequest(BaseModel):
    citing_sentence: str


api_router = APIRouter()


@api_router.get("/health", status_code=200)
def health():
    try:
        _load_model_if_needed()
    except Exception as e:
        return {"status": "degraded", "device": None, "model_version": __version__, "error": str(e)}

    return {
        "status": "ok",
        "device": _device,
        "model_version": __version__,
        "labels": list(_id2label.values()),
    }


@api_router.post("/classify-citation", status_code=200)
async def classify_citation(payload: CitationRequest):
    try:
        result = predict(payload.citing_sentence)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return result
