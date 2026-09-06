from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import main as main_module


class FakeClassifier:
    device = "cpu"
    id2label = {
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

    _RULES = {
        "adopt": "Application",
        "confirms": "Evidence",
        "fails": "Gap",
        "compare": "Comparison",
    }

    def predict(self, citing_sentence: str, cited_paragraphs=None):
        predicted_label = "Background"  # valor por defecto si no matchea ninguna regla
        for keyword, label in self._RULES.items():
            if keyword in citing_sentence.lower():
                predicted_label = label
                break

        return {
            "predicted_label": predicted_label,
            "confidence": 0.87,
            "used_context": bool(cited_paragraphs),
            "all_probabilities": [
                {"label": predicted_label, "probability": 0.87},
                {"label": "Background", "probability": 0.08},
                {"label": "Comparison", "probability": 0.05},
            ],
        }


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
def client() -> Generator:
    with TestClient(main_module.app) as _client:
        main_module.classifier = FakeClassifier()
        yield _client
        main_module.classifier = None
