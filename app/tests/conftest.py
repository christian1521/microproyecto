from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app import api as api_module


_RULES = {
    "adopt": "Application",
    "confirms": "Evidence",
    "fails": "Gap",
    "compare": "Comparison",
}


def _fake_predict(citing_sentence: str, cited_paragraphs=None) -> dict:
    predicted_label = "Background"
    for keyword, label in _RULES.items():
        if keyword in citing_sentence.lower():
            predicted_label = label
            break

    return {
        "predicted_label": predicted_label,
        "confidence": 0.87,
        "used_context": bool(cited_paragraphs),
        "all_probabilities": {
            predicted_label: 0.87,
            "Background": 0.08,
            "Comparison": 0.05,
        },
    }


def _fake_load_model_if_needed():
    api_module._device = "cpu"
    api_module._id2label = {
        0: "Background",
        1: "Further Reading",
        2: "Evidence",
        3: "Basis",
        4: "Application",
        5: "Improvement / Modification",
        6: "Identification of the Originator",
        7: "Gap",
        8: "Comparison",
    }
    api_module._model = object()
    api_module._tokenizer = object()


@pytest.fixture(scope="module")
def test_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "citing_sentence": "Similar to the method proposed in [CITATION], we adopt a transformer-based architecture.",
                "expected_label": "Application",
            },
            {
                "citing_sentence": "This confirms earlier findings by [CITATION] regarding attention sparsity.",
                "expected_label": "Evidence",
            },
            {
                "citing_sentence": "However, [CITATION] fails to generalize to low-resource settings.",
                "expected_label": "Gap",
            },
            {
                "citing_sentence": "We compare our results against those reported in [CITATION].",
                "expected_label": "Comparison",
            },
        ]
    )


@pytest.fixture()
def client(monkeypatch) -> Generator:
    monkeypatch.setattr(api_module, "predict", _fake_predict)
    monkeypatch.setattr(api_module, "_load_model_if_needed", _fake_load_model_if_needed)

    with TestClient(main_module.app) as _client:
        yield _client
