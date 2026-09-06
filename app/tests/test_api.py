import math

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    # Given
    sample = test_data.iloc[0]
    payload = {"citing_sentence": sample["citing_sentence"]}

    # When
    response = client.post("/classify-citation", json=payload)

    # Then
    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["predicted_label"]
    assert 0.0 <= prediction_data["confidence"] <= 1.0