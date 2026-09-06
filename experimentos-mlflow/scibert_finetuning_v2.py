# -*- coding: utf-8 -*-
"""
outputv2.py — Barrido de experimentos de clasificación de función de cita,
registrado en el servidor MLflow del equipo.

Grupo 24 — MAIA Uniandes — Proyecto: Desarrollo de Soluciones
Derivado de `modelos/scibert_eda_cleaning_finetuning.py`.

IDEA CENTRAL
------------
El fine-tuning completo de SciBERT tarda horas y produce **una** corrida, lo
que hace imposible comparar configuraciones. Aquí se invierte el planteamiento:
en vez de reentrenar el encoder, se usa SciBERT **congelado** como extractor de
representaciones. El costoso paso hacia adelante se hace UNA sola vez por cada
longitud de secuencia y se cachea en disco; a partir de ahí, cada experimento
entrena únicamente una cabeza clásica sobre esas representaciones, lo que toma
segundos. Así se pueden registrar y comparar decenas de corridas en MLflow.

Se barren dos familias de modelos, en el mismo experimento, para que la UI
permita compararlas lado a lado:

  familia = tfidf           TF-IDF + clasificador lineal. Línea base sin redes
                            neuronales, entrena en segundos.
  familia = scibert_frozen  Embeddings de SciBERT congelado + cabeza clásica.
                            Mide cuánto aporta la representación científica
                            preentrenada sin pagar el costo del fine-tuning.

PARÁMETROS QUE SE VARÍAN (los que hacen que una corrida dure segundos)
----------------------------------------------------------------------
  muestra_por_clase   Submuestreo balanceado del dataset (500/clase = 4.500
                      filas frente a las 718.172 originales). Es la palanca
                      que más reduce el tiempo.
  max_length          64 o 128 tokens. Recortar a la mitad la secuencia casi
                      divide por dos el costo del paso hacia adelante.
  pooling             cls o mean sobre la última capa oculta. Ambas se obtienen
                      del mismo paso hacia adelante, así que comparar pooling
                      es gratis.
  clasificador        logreg | linear_svc | complement_nb | mlp
  C / alpha           Regularización de la cabeza.
  class_weight        balanced o None, para medir el efecto del desbalance.
  ngram_range,        Solo para la familia tfidf.
  max_features

PARTICIÓN SIN FUGA DE INFORMACIÓN
---------------------------------
El enunciado del proyecto exige que no haya fuga a nivel de documento. La
partición se hace con GroupShuffleSplit agrupando por `paper-id`: ninguna
frase de un mismo artículo aparece simultáneamente en train y en test. Por eso
la proporción de test no cae exactamente en el 20 % y se registra el valor real.

Uso:
    python outputv2.py                # las dos familias, ~12 corridas
    python outputv2.py tfidf          # solo la familia rápida
    python outputv2.py scibert
"""

import os
import sys
import time
import platform
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

# =============================================================================
# 1. Servidor MLflow
# =============================================================================

# El servidor responde en HTTP plano (mlflow server --host 0.0.0.0 --port 8050
# no habilita TLS por sí solo). https:// queda como candidato de respaldo para
# cuando se ponga un proxy con certificado delante.
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://54.208.154.153:8050")
URIS_CANDIDATAS = [
    MLFLOW_URI,
    MLFLOW_URI.replace("https://", "http://", 1),
    MLFLOW_URI.replace("http://", "https://", 1),
]
os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")

EXPERIMENT_NAME = "citation-function-experimentos"


def conectar_mlflow():
    """Fija la primera URI que responda y devuelve (uri, experiment)."""
    from mlflow.tracking import MlflowClient

    intentos, vistas = [], set()
    for uri in URIS_CANDIDATAS:
        if uri in vistas:
            continue
        vistas.add(uri)
        try:
            mlflow.set_tracking_uri(uri)
            MlflowClient().search_experiments(max_results=1)
            experiment = mlflow.set_experiment(EXPERIMENT_NAME)
            print("Servidor MLflow: {} | experimento '{}' (id={})".format(
                uri, EXPERIMENT_NAME, experiment.experiment_id))
            return uri, experiment
        except Exception as exc:  # noqa: BLE001
            intentos.append((uri, "{}: {}".format(type(exc).__name__, exc)))

    detalle = "\n".join("  - {}\n      {}".format(u, m[:300]) for u, m in intentos)
    raise ConnectionError(
        "No fue posible conectarse a ningún servidor MLflow.\n" + detalle +
        "\n\nRevisar: instancia EC2 encendida, IP pública vigente, puerto 8050 "
        "abierto en el security group, y servidor lanzado con --host 0.0.0.0."
    )


# =============================================================================
# 2. Datos
# =============================================================================

BASE_NAME = "final_labelled_citing_sentences_all"
INPUT_CSV = Path(os.getenv(
    "INPUT_CSV",
    Path(__file__).resolve().parent.parent / "notebook-datos" / (BASE_NAME + ".csv"),
))

TEXT_COL = "citing-sentence"
LABEL_COL = "label"
GROUP_COL = "paper-id"

MIN_TEXT_LENGTH = 15
MAX_TEXT_LENGTH = 2000
RANDOM_SEED = 42

# Palanca principal de velocidad: 500 x 9 = 4.500 filas balanceadas.
MUESTRA_POR_CLASE = int(os.getenv("MUESTRA_POR_CLASE", "500"))
TEST_SIZE = 0.20

MODEL_NAME = "allenai/scibert_scivocab_uncased"

# Caché de embeddings: el paso hacia adelante de SciBERT se paga una sola vez.
CACHE_DIR = Path(os.getenv(
    "CACHE_DIR",
    Path(tempfile.gettempdir()) / "outputv2-cache-embeddings",
))

EXPECTED_LABELS = {
    "Background", "Further Reading", "Evidence", "Basis", "Application",
    "Improvement / Modification", "Identification of the Originator",
    "Gap", "Comparison",
}


def preparar_datos():
    """Carga, limpia, submuestrea de forma balanceada y parte por documento.

    Devuelve (train_df, test_df, info) con `info` listo para registrarse como
    parámetros y métricas comunes a todas las corridas.
    """
    import re
    from sklearn.model_selection import GroupShuffleSplit

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            "No se encontró el CSV '{}'. Se genera desde "
            "notebook-datos/base-{}.txt aplicando el mapeo de 18 a 9 "
            "etiquetas del notebook del equipo.".format(INPUT_CSV, BASE_NAME)
        )

    print("Cargando {} ...".format(INPUT_CSV.name))
    df = pd.read_csv(INPUT_CSV, dtype={GROUP_COL: str, "line-number": str},
                     usecols=[GROUP_COL, TEXT_COL, LABEL_COL])
    n_inicial = len(df)

    # Limpieza (mismos criterios que el script de fine-tuning)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str).apply(
        lambda t: re.sub(r"\s+", " ", t.replace("\n", " ").replace("\r", " ")).strip())
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df[df[TEXT_COL] != ""]
    df = df.drop_duplicates(subset=[TEXT_COL, LABEL_COL])
    largo = df[TEXT_COL].str.len()
    df = df[(largo >= MIN_TEXT_LENGTH) & (largo <= MAX_TEXT_LENGTH)]
    df = df[df[LABEL_COL].isin(EXPECTED_LABELS)]
    n_limpio = len(df)

    # Submuestreo balanceado: el dataset del proyecto debe quedar equilibrado
    # y, de paso, es lo que vuelve viables los experimentos en minutos.
    n_por_clase = min(MUESTRA_POR_CLASE, int(df[LABEL_COL].value_counts().min()))
    df = pd.concat(
        [g.sample(n=n_por_clase, random_state=RANDOM_SEED)
         for _, g in df.groupby(LABEL_COL)]
    ).reset_index(drop=True)

    # Partición agrupada por documento: sin fuga entre train y test.
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                            random_state=RANDOM_SEED)
    idx_train, idx_test = next(gss.split(df, groups=df[GROUP_COL]))
    train_df = df.iloc[idx_train].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)

    papers_compartidos = len(set(train_df[GROUP_COL]) & set(test_df[GROUP_COL]))

    info = {
        "filas_csv_original": n_inicial,
        "filas_tras_limpieza": n_limpio,
        "muestra_por_clase": n_por_clase,
        "n_total": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_size_real": len(test_df) / len(df),
        "num_clases": int(df[LABEL_COL].nunique()),
        "papers_unicos": int(df[GROUP_COL].nunique()),
        "papers_compartidos_train_test": papers_compartidos,
    }

    print("  {} filas limpias -> muestra balanceada de {} ({}/clase)".format(
        n_limpio, len(df), n_por_clase))
    print("  train={} test={} ({:.1%}) | papers únicos={} | fuga={}".format(
        len(train_df), len(test_df), info["test_size_real"],
        info["papers_unicos"], papers_compartidos))
    return train_df, test_df, info


# =============================================================================
# 3. Representaciones
# =============================================================================

def embeddings_scibert(textos, max_length, cache_key):
    """Paso hacia adelante de SciBERT congelado. Devuelve (emb_cls, emb_mean).

    Sin gradientes y en modo evaluación: el encoder no se modifica. El
    resultado se cachea en disco, de modo que solo la primera corrida de cada
    `max_length` paga el costo.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ruta_cls = CACHE_DIR / "{}_len{}_cls.npy".format(cache_key, max_length)
    ruta_mean = CACHE_DIR / "{}_len{}_mean.npy".format(cache_key, max_length)

    if ruta_cls.exists() and ruta_mean.exists():
        print("  embeddings en caché (max_length={})".format(max_length))
        return np.load(ruta_cls), np.load(ruta_mean), 0.0

    torch.set_num_threads(os.cpu_count() or 4)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("  calculando embeddings SciBERT (max_length={}, {} textos)...".format(
        max_length, len(textos)))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    modelo = AutoModel.from_pretrained(MODEL_NAME)
    modelo.eval()

    t0 = time.time()
    lote = 32
    salidas_cls, salidas_mean = [], []
    with torch.no_grad():
        for ini in range(0, len(textos), lote):
            batch = textos[ini:ini + lote]
            enc = tokenizer(batch, truncation=True, padding="max_length",
                            max_length=max_length, return_tensors="pt")
            out = modelo(**enc).last_hidden_state           # [B, T, 768]

            salidas_cls.append(out[:, 0, :].numpy())        # token [CLS]

            # Media enmascarada: el padding no debe contaminar el promedio.
            mascara = enc["attention_mask"].unsqueeze(-1).float()
            media = (out * mascara).sum(1) / mascara.sum(1).clamp(min=1e-9)
            salidas_mean.append(media.numpy())

            if (ini // lote) % 20 == 0:
                print("    {}/{} textos".format(min(ini + lote, len(textos)),
                                                len(textos)), flush=True)

    emb_cls = np.vstack(salidas_cls)
    emb_mean = np.vstack(salidas_mean)
    segundos = time.time() - t0
    np.save(ruta_cls, emb_cls)
    np.save(ruta_mean, emb_mean)
    print("  listo en {:.1f} s -> {}".format(segundos, emb_cls.shape))
    return emb_cls, emb_mean, segundos


def construir_cabeza(nombre, C, class_weight):
    """Instancia el clasificador de la cabeza según su nombre.

    `class_weight` llega como el texto que se registra en MLflow, donde la
    ausencia de ponderación se escribe "none"; sklearn espera None.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import ComplementNB
    from sklearn.neural_network import MLPClassifier

    if class_weight in ("none", "None", ""):
        class_weight = None

    if nombre == "logreg":
        return LogisticRegression(C=C, max_iter=2000, class_weight=class_weight,
                                  random_state=RANDOM_SEED)
    if nombre == "linear_svc":
        return LinearSVC(C=C, class_weight=class_weight, random_state=RANDOM_SEED)
    if nombre == "complement_nb":
        return ComplementNB(alpha=C)
    if nombre == "mlp":
        return MLPClassifier(hidden_layer_sizes=(256,), max_iter=40,
                             random_state=RANDOM_SEED)
    raise ValueError("Clasificador desconocido: {}".format(nombre))


# =============================================================================
# 4. Registro de una corrida
# =============================================================================

def registrar_corrida(experiment, nombre, params, Xtr, ytr, Xte, yte,
                      clases, info, segundos_features):
    """Entrena la cabeza, evalúa y registra todo en MLflow."""
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 classification_report)

    with mlflow.start_run(experiment_id=experiment.experiment_id,
                          run_name=nombre) as run:
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("muestra_por_clase", info["muestra_por_clase"])
        mlflow.log_param("random_seed", RANDOM_SEED)

        mlflow.set_tag("equipo", "Grupo 24")
        mlflow.set_tag("familia", params["familia"])
        mlflow.set_tag("dataset", INPUT_CSV.name)
        mlflow.set_tag("host", platform.node())

        for k, v in info.items():
            mlflow.log_metric(k, float(v))
        mlflow.log_metric("n_features", Xtr.shape[1])
        mlflow.log_metric("segundos_features", segundos_features)

        modelo = construir_cabeza(params["clasificador"],
                                  params.get("C", 1.0),
                                  params.get("class_weight") or None)

        t0 = time.time()
        modelo.fit(Xtr, ytr)
        segundos = time.time() - t0

        preds = modelo.predict(Xte)
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
            yte, preds, average="macro", zero_division=0)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(
            yte, preds, average="weighted", zero_division=0)

        metricas = {
            "accuracy": accuracy_score(yte, preds),
            "f1_macro": f1_mac,
            "precision_macro": p_mac,
            "recall_macro": r_mac,
            "f1_weighted": f1_w,
            "precision_weighted": p_w,
            "recall_weighted": r_w,
            "segundos_entrenamiento": segundos,
        }
        for k, v in metricas.items():
            mlflow.log_metric(k, float(v))

        reporte = classification_report(yte, preds, labels=clases,
                                        target_names=clases, zero_division=0)
        with tempfile.TemporaryDirectory() as tmp:
            ruta = Path(tmp) / "classification_report.txt"
            ruta.write_text(reporte, encoding="utf-8")
            mlflow.log_artifact(str(ruta))

        print("  {:42s} f1_macro={:.4f}  acc={:.4f}  ({:.1f} s)".format(
            nombre, f1_mac, metricas["accuracy"], segundos))
        return {"run": nombre, "f1_macro": f1_mac,
                "accuracy": metricas["accuracy"], "segundos": segundos}


# =============================================================================
# 5. Familia tfidf
# =============================================================================

# Cada entrada varía un solo factor respecto de la primera, para que la
# comparación en MLflow sea interpretable.
GRID_TFIDF = [
    {"ngram_range": (1, 1), "max_features": 50000, "clasificador": "logreg",
     "C": 1.0, "class_weight": "balanced"},
    {"ngram_range": (1, 2), "max_features": 50000, "clasificador": "logreg",
     "C": 1.0, "class_weight": "balanced"},
    {"ngram_range": (1, 2), "max_features": 50000, "clasificador": "logreg",
     "C": 10.0, "class_weight": "balanced"},
    {"ngram_range": (1, 2), "max_features": 50000, "clasificador": "logreg",
     "C": 1.0, "class_weight": None},
    {"ngram_range": (1, 2), "max_features": 50000, "clasificador": "linear_svc",
     "C": 1.0, "class_weight": "balanced"},
    {"ngram_range": (1, 2), "max_features": 50000, "clasificador": "complement_nb",
     "C": 1.0, "class_weight": None},
]


def familia_tfidf(experiment, train_df, test_df, info, clases):
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("\n--- Familia tfidf ({} corridas) ---".format(len(GRID_TFIDF)))
    resultados = []
    for cfg in GRID_TFIDF:
        t0 = time.time()
        vec = TfidfVectorizer(ngram_range=cfg["ngram_range"],
                              max_features=cfg["max_features"],
                              sublinear_tf=True)
        Xtr = vec.fit_transform(train_df[TEXT_COL].values)
        Xte = vec.transform(test_df[TEXT_COL].values)
        segundos_features = time.time() - t0

        params = {
            "familia": "tfidf",
            "representacion": "tfidf",
            "ngram_range": str(cfg["ngram_range"]),
            "max_features": cfg["max_features"],
            "sublinear_tf": True,
            "clasificador": cfg["clasificador"],
            "C": cfg["C"],
            "class_weight": cfg["class_weight"] or "none",
        }
        nombre = "tfidf-{}-ng{}-C{}-{}".format(
            cfg["clasificador"], cfg["ngram_range"][1], cfg["C"],
            "bal" if cfg["class_weight"] else "nobal")

        resultados.append(registrar_corrida(
            experiment, nombre, params, Xtr, train_df[LABEL_COL].values,
            Xte, test_df[LABEL_COL].values, clases, info, segundos_features))
    return resultados


# =============================================================================
# 6. Familia scibert_frozen
# =============================================================================

GRID_SCIBERT = [
    {"max_length": 128, "pooling": "mean", "clasificador": "logreg",
     "C": 1.0, "class_weight": "balanced"},
    {"max_length": 128, "pooling": "cls", "clasificador": "logreg",
     "C": 1.0, "class_weight": "balanced"},
    {"max_length": 128, "pooling": "mean", "clasificador": "logreg",
     "C": 10.0, "class_weight": "balanced"},
    {"max_length": 128, "pooling": "mean", "clasificador": "linear_svc",
     "C": 1.0, "class_weight": "balanced"},
    {"max_length": 128, "pooling": "mean", "clasificador": "mlp",
     "C": 1.0, "class_weight": None},
    {"max_length": 64, "pooling": "mean", "clasificador": "logreg",
     "C": 1.0, "class_weight": "balanced"},
]


def familia_scibert(experiment, train_df, test_df, info, clases):
    print("\n--- Familia scibert_frozen ({} corridas) ---".format(len(GRID_SCIBERT)))

    textos_train = train_df[TEXT_COL].tolist()
    textos_test = test_df[TEXT_COL].tolist()
    clave = "n{}_seed{}".format(info["n_total"], RANDOM_SEED)

    # Un paso hacia adelante por cada max_length; ambos poolings salen del mismo.
    cache = {}
    for max_length in sorted({c["max_length"] for c in GRID_SCIBERT}):
        tr_cls, tr_mean, s1 = embeddings_scibert(
            textos_train, max_length, clave + "_train")
        te_cls, te_mean, s2 = embeddings_scibert(
            textos_test, max_length, clave + "_test")
        cache[max_length] = {
            "cls": (tr_cls, te_cls), "mean": (tr_mean, te_mean),
            "segundos": s1 + s2,
        }

    resultados = []
    for cfg in GRID_SCIBERT:
        entrada = cache[cfg["max_length"]]
        Xtr, Xte = entrada[cfg["pooling"]]

        params = {
            "familia": "scibert_frozen",
            "representacion": "scibert-embeddings",
            "modelo_base": MODEL_NAME,
            "encoder_congelado": True,
            "max_length": cfg["max_length"],
            "pooling": cfg["pooling"],
            "clasificador": cfg["clasificador"],
            "C": cfg["C"],
            "class_weight": cfg["class_weight"] or "none",
        }
        nombre = "scibert-{}-{}-len{}-C{}".format(
            cfg["clasificador"], cfg["pooling"], cfg["max_length"], cfg["C"])

        resultados.append(registrar_corrida(
            experiment, nombre, params, Xtr, train_df[LABEL_COL].values,
            Xte, test_df[LABEL_COL].values, clases, info,
            entrada["segundos"]))
    return resultados


# =============================================================================
# 7. Punto de entrada
# =============================================================================

def main():
    familias = (sys.argv[1] if len(sys.argv) > 1 else "todas").lower()

    print("=" * 78)
    print("outputv2.py — barrido de experimentos  ({:%Y-%m-%d %H:%M})".format(
        datetime.now()))
    print("=" * 78)

    _, experiment = conectar_mlflow()
    train_df, test_df, info = preparar_datos()
    clases = sorted(train_df[LABEL_COL].unique().tolist())

    resultados = []
    t0 = time.time()
    if familias in ("todas", "tfidf"):
        resultados += familia_tfidf(experiment, train_df, test_df, info, clases)
    if familias in ("todas", "scibert"):
        resultados += familia_scibert(experiment, train_df, test_df, info, clases)

    print("\n" + "=" * 78)
    print("Ranking por f1_macro ({} corridas, {:.1f} s en total)".format(
        len(resultados), time.time() - t0))
    print("=" * 78)
    for r in sorted(resultados, key=lambda x: -x["f1_macro"]):
        print("  {:42s} f1_macro={:.4f}  acc={:.4f}".format(
            r["run"], r["f1_macro"], r["accuracy"]))
    print("\nExperimento: {}/#/experiments/{}".format(
        mlflow.get_tracking_uri(), experiment.experiment_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
