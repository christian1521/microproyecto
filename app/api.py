import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

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


# --------------------------------------------------------------------
# Catálogo de las 9 funciones de cita: definición y criterios.
# Se modela como una clase (CitationFunctionInfo) + un diccionario de
# clase (CITATION_FUNCTIONS) que actúa como listado/registro de las 9
# categorías soportadas por el clasificador.
# --------------------------------------------------------------------
class CitationFunctionInfo(BaseModel):
    definition: str
    criteria: List[str]

class CitationFunctionCatalog:
    """Contiene el listado completo de las 9 funciones de cita soportadas por el modelo."""

    CITATION_FUNCTIONS: Dict[str, CitationFunctionInfo] = {
        "Background": CitationFunctionInfo(
            definition=(
                "Citations used to provide context, summarize the general "
                "background of a research topic, or trace the history of a "
                "field or idea. These citations lay the foundation for "
                "understanding the focal study."
            ),
            criteria=[
                "Reviews existing literature to ensure comprehensiveness.",
                "Cites review papers or relevant previous studies to depict "
                "the current state of the field.",
                "Highlights important or prevailing findings in the field.",
                "Traces the historical development or presents key ideas in "
                "the subject area.",
            ],
        ),
        "Gap": CitationFunctionInfo(
            definition=(
                "Citations that help identify research gaps or unexplored "
                "areas, justifying the author's choice of research topic."
            ),
            criteria=[
                "Highlights what has or has not been done in the research area.",
                "Identifies gaps in knowledge or areas for further study.",
                "Justifies the relevance and importance of the current "
                "research by contrasting it with existing work.",
            ],
        ),
        "Basis": CitationFunctionInfo(
            definition=(
                "Citations that provide a foundation for the current "
                "research. This can operate at a macro level (broad "
                "influence) or a micro level (specific key contributions)."
            ),
            criteria=[
                "Serves as the intellectual starting point for the focal research.",
                "The cited work significantly shapes the research idea or hypothesis.",
                "The focal research builds upon, continues, or expands on "
                "the cited work.",
            ],
        ),
        "Comparison": CitationFunctionInfo(
            definition=(
                "Citations used to compare the current work with cited "
                "studies or to draw comparisons between cited studies."
            ),
            criteria=[
                "Highlights similarities or differences between the current "
                "research and cited works.",
                "Compares methodologies, findings, algorithms, data, or "
                "theoretical concepts.",
                "May point out advantages of the current study over previous "
                "ones or establish links among different cited works.",
            ],
        ),
        "Application": CitationFunctionInfo(
            definition=(
                "Citations that directly employ a method, technique, or "
                "tool from the cited work without modification."
            ),
            criteria=[
                "Utilizes existing methods, algorithms, instruments, or data "
                "for the current research.",
                "Employs the cited work as a practical tool (e.g., "
                "equations, analysis methods) for calculation or "
                "experimentation.",
                "Applies the cited method without modification.",
            ],
        ),
        "Improvement / Modification": CitationFunctionInfo(
            definition=(
                "Citations in which methods or tools from the cited work "
                "are adapted, expanded, or modified for the current research."
            ),
            criteria=[
                "The cited method is adapted, improved, or extended to fit "
                "new experimental conditions or research objectives.",
                "Uses the cited work as a foundation but modifies it to "
                "enhance precision or scope.",
            ],
        ),
        "Evidence": CitationFunctionInfo(
            definition=(
                "Citations used to support claims, hypotheses, or findings "
                "in the current research."
            ),
            criteria=[
                "Supports arguments, hypotheses, or factual statements.",
                "Justifies research design, methodologies, or experimental "
                "procedures.",
                "Substantiates or explains findings, especially in the "
                "discussion section.",
                "Provides evidence to mitigate limitations or support "
                "further research suggestions.",
            ],
        ),
        "Identification of the Originator": CitationFunctionInfo(
            definition=(
                "Citations used to acknowledge the original source of an "
                "idea, concept, method, or theory."
            ),
            criteria=[
                "Identifies the original publication where a key idea or "
                "method was first introduced.",
                "Acknowledges pioneers in the field.",
                "Gives credit to the priority of a cited work, demonstrating "
                "intellectual indebtedness.",
            ],
        ),
        "Further Reading": CitationFunctionInfo(
            definition=(
                "Citations that direct readers to additional or "
                "supplementary literature for more detailed information or "
                "context."
            ),
            criteria=[
                "Alerts readers to new, different, or relevant sources of "
                "information.",
                "Provides more complete details on data, methods, or "
                "background not fully covered in the current work.",
                "Often used to refer readers to external sources for deeper "
                "insights.",
            ],
        ),
    }

    @classmethod
    def get(cls, label: str) -> Optional[CitationFunctionInfo]:
        return cls.CITATION_FUNCTIONS.get(label)

    @classmethod
    def list_labels(cls) -> List[str]:
        return list(cls.CITATION_FUNCTIONS.keys())


def predict(citing_sentence: str, type_model: Optional[str] = None, cited_paragraphs: Optional[str] = None) -> dict:
    """
    Punto único de inferencia.

    El resultado incluye, la etiqueta y confianza, la definition y criterio de la función de cita predicha (según CitationFunctionCatalog).
    """

    if cited_paragraphs and "[CITATION]" not in citing_sentence:
        raise ValueError("El contexto de cita debe contener la etiqueta '[CITATION]' cuando se proporciona el párrafo del documento citado.")

    _load_model_if_needed()

    import torch

    if cited_paragraphs:
        text_to_classify = (
            f"{citing_sentence} CONTEXT: In the text, the [CITATION] tag "
            f"refers to: {cited_paragraphs}"
        )
    else:
        text_to_classify = citing_sentence

    inputs = _tokenizer(
        text_to_classify,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

    pred_id = int(torch.argmax(probs).item())
    predicted_label = _id2label[pred_id]

    result = {
        "predicted_label": predicted_label,
        "confidence": float(probs[pred_id].item()),
        "citing_sentence": citing_sentence,
        "all_probabilities": {
            _id2label[i]: float(probs[i].item()) for i in range(len(_id2label))
        },
    }

    citation_info = CitationFunctionCatalog.get(predicted_label)
    if citation_info is not None:
        result["definition"] = citation_info.definition
        result["criteria"] = citation_info.criteria

    return result


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
    }


@api_router.get("/citation-functions", status_code=200)
def list_citation_functions():
    """Devuelve el catálogo completo de las 9 funciones de cita soportadas,
    con su Definition y Criteria -- útil para que la app web muestre esta
    información sin necesidad de duplicarla en el frontend."""
    return {
        label: info.model_dump()
        for label, info in CitationFunctionCatalog.CITATION_FUNCTIONS.items()
    }


@api_router.post("/classify-citation", status_code=200)
async def classify_citation(payload: CitationRequest):
    try:
        result = predict(payload.citing_sentence)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result