import os
import json
from pathlib import Path

import torch
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app import __version__

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./data/model_scibert_citing_sentences"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))

if not (MODEL_DIR / "config.json").exists():
    raise FileNotFoundError(
        f"No se encontró 'config.json' en {MODEL_DIR}."
    )

print(f"Cargando modelo desde: {MODEL_DIR}")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to(device)
model.eval()

label_mapping_path = MODEL_DIR / "label_mapping.json"
if label_mapping_path.exists():
    with open(label_mapping_path, "r", encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}
else:
    id2label = {int(k): v for k, v in model.config.id2label.items()}

print(f"Modelo cargado en dispositivo: {device}")
print(f"Etiquetas: {list(id2label.values())}")

# --------------------------------------------------------------------
# Esquemas de request
# --------------------------------------------------------------------
class CitationRequest(BaseModel):
    citing_sentence: str


api_router = APIRouter()


@api_router.get("/health", status_code=200)
def health():
    return {"status": "ok", "device": device, "model_version": __version__, "labels": list(id2label.values())}


@api_router.post("/classify-citation", status_code=200)
async def classify_citation(payload: CitationRequest):
    inputs = tokenizer(
        payload.citing_sentence,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)

    pred_id = int(torch.argmax(probs).item())

    return {
        "predicted_label": id2label[pred_id],
        "confidence": float(probs[pred_id].item()),
        "all_probabilities": {
            id2label[i]: float(probs[i].item()) for i in range(len(id2label))
        },
    }

app = FastAPI(title="Citation Function Classification API")
app.include_router(api_router)
