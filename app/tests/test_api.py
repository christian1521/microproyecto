from pathlib import Path
 
import pandas as pd
import pytest
 
from app.api import predict, MODEL_DIR
 
_MODEL_AVAILABLE = (Path(MODEL_DIR) / "config.json").exists()
 
 
@pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason=(
        f"No se encontró el modelo en '{MODEL_DIR}'. "
        f"Define MODEL_DIR o coloca el modelo ahí para correr esta prueba."
    ),
)
def test_make_prediction(test_data: pd.DataFrame) -> None:
    # Given
    sample = test_data.iloc[0]
 
    # When
    prediction_data = predict(citing_sentence=sample["citing_sentence"])
 
    # Then
    assert prediction_data["predicted_label"]
    assert 0.0 <= prediction_data["confidence"] <= 1.0