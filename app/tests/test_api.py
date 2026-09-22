from pathlib import Path
 
import pandas as pd
import pytest
 
from app.api import predict, MODEL_DIR

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_DIR_SCIBERT_AVAILABLE = BASE_DIR / "data" / "model_scibert_citing_sentences"
MODEL_DIR_BERT_AVAILABLE = BASE_DIR / "data" / "model_bert_citing_sentences"

@pytest.mark.skipif(
    not MODEL_DIR_SCIBERT_AVAILABLE.exists() and not MODEL_DIR_BERT_AVAILABLE.exists(),
    reason=(
        f"No se encontró el directorio de los moelos BERT o SCIBERT."
    ),
)
def test_make_prediction(test_data: pd.DataFrame) -> None:
    # Given
    sample = test_data.iloc[0]
 
    # When
    prediction_data = predict(citing_sentence=sample["citing_sentence"], type_model="scibert")
 
    # Then
    assert prediction_data["predicted_label"]
    assert 0.0 <= prediction_data["confidence"] <= 1.0