# -*- coding: utf-8 -*-
"""experimento_BERT.py

# BERT large uncased para Clasificación de Función de Cita Científica

`bert-large-uncased` se usa congelado, como extractor de representaciones. 
Sobre esas representaciones se ajusta una
cabeza lineal (regresión logística), que es el procedimiento estándar de
*linear probe*: mide cuánta información sobre la función de cita ya está
presente en el modelo preentrenado, sin entrenarlo.

Por qué hace falta la cabeza: `bert-large-uncased` es un modelo de lenguaje
enmascarado, no un clasificador. No tiene 9 salidas ni sabe nada de las
categorías del proyecto, así que asi como viene, no puede emitir una
predicción por sí solo. La cabeza lineal es la forma mínima de leer sus
representaciones sin tocar el encoder: 0 de los 335 M de parámetros de BERT se
actualizan; solo se ajustan los ~7 K coeficientes de la regresión logística.

Proceso:
1. **Extracción de embeddings** con BERT large congelado (`torch.no_grad()`).
2. **Cabeza lineal** sobre los embeddings y evaluación.

Registro en MLflow (experimento `modelos-openweight `):
- Parámetros: `model_base`, `balance_strategy`, `max_length`, `pooling`,
  `cabeza`, `C`, `train_size`, `val_size`, `test_size`, `num_labels`,
  `encoder_congelado`, `muestra_por_clase`.
- Métricas: `accuracy`, `f1_macro`, `precision_macro`, `recall_macro`,
  `f1_weighted`, `precision_weighted`, `recall_weighted`, más
  `segundos_embeddings` y `segundos_cabeza`.

## 1. Importar librerías
"""

# !pip install transformers torch scikit-learn pandas matplotlib seaborn mlflow --quiet

import re
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import mlflow

import torch
from transformers import AutoTokenizer, AutoModel, set_seed

sns.set_theme(style="whitegrid")
pd.set_option("display.max_colwidth", 120)

"""## 2. Configuración"""

BASE_NAME = "final_labelled_citing_sentences_all"
#INPUT_CSV = Path(f"/content/drive/MyDrive/Colab Notebooks/microproyecto/data/{BASE_NAME}.csv")
INPUT_CSV = Path(f"/content/{BASE_NAME}.csv")

TEXT_COL = "citing-sentence"
LABEL_COL = "label"

# Modelo tal como viene del hub, sin fine-tuning.
MODEL_NAME = "bert-large-uncased"
OUTPUT_DIR = Path("/content/data/bert-large-sin-finetuning")

# Umbrales de limpieza por longitud del texto (en caracteres)
MIN_TEXT_LENGTH = 15
MAX_TEXT_LENGTH = 2000

# Submuestreo balanceado. BERT large son 335 M de parámetros: con GPU, 2000 por
# clase (18.000 textos) tarda unos minutos; en CPU conviene bajar a 500.
MUESTRA_POR_CLASE = 500

# Longitud de secuencia para el paso hacia adelante.
MAX_LENGTH = 128
# Agregación de la última capa oculta: "mean" (media enmascarada) o "cls".
# En un modelo sin fine-tuning el token [CLS] no está entrenado para resumir la
# oración, así que "mean" suele funcionar bastante mejor.
POOLING = "mean"
BATCH_SIZE_EMB = 32

# Cabeza lineal sobre los embeddings congelados.
CABEZA = "logreg"
C_REG = 1.0
CLASS_WEIGHT = "balanced"

RANDOM_SEED = 42
set_seed(RANDOM_SEED)

# --- Registro de experimentos en MLflow ---
# Servidor del equipo (instancia EC2). Responde en HTTP plano: `mlflow server
# --host 0.0.0.0 --port 8050` no habilita TLS por sí solo, así que https://
# fallaría. Ojo: la IP pública cambia cada vez que la instancia se reinicia.
MLFLOW_TRACKING_URI = "http://3.89.93.186:8050"
# Los tres experimentos (BERT, Qwen, Mistral) escriben en el mismo experimento
# para poder compararlos lado a lado en la UI.
MLFLOW_EXPERIMENT_NAME = "modelos-openweigth"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
print(f"MLflow: {MLFLOW_TRACKING_URI} | experimento "
      f"'{MLFLOW_EXPERIMENT_NAME}' (id={experiment.experiment_id})")

"""## 3. Carga del dataset"""

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo '{INPUT_CSV}'."
    )

df_raw = pd.read_csv(INPUT_CSV, dtype={"paper-id": str, "line-number": str})

print(f"Filas cargadas: {len(df_raw)}")
print(f"Columnas: {list(df_raw.columns)}")
df_raw.head(5)

"""## 4. Análisis exploratorio (EDA)

### 4.1 Nulos, duplicados, tipos de dato
"""

print("Info general del DataFrame:")
df_raw.info()

print("\nValores nulos por columna:")
print(df_raw.isnull().sum())

print(f"\nFilas totalmente duplicadas: {df_raw.duplicated().sum()}")
print(f"Duplicados exactos en '{TEXT_COL}': {df_raw[TEXT_COL].duplicated().sum()}")

"""### 4.2 Distribución de clases (label)"""

label_counts = df_raw[LABEL_COL].value_counts()
print(label_counts)

plt.figure(figsize=(9, 5))
sns.barplot(x=label_counts.values, y=label_counts.index, palette="viridis")
plt.title("Distribución de categorías de función de cita (antes de limpiar)")
plt.xlabel("Registros")
plt.ylabel("Categoría")
plt.tight_layout()
plt.show()

print(f"\nRatio desbalance (clase mayoritaria / clase minoritaria): "
      f"{label_counts.max() / label_counts.min():.2f}x")

"""### 4.3 Distribución de longitud del texto"""

df_raw["text_length"] = df_raw[TEXT_COL].astype(str).apply(len)
df_raw["word_count"] = df_raw[TEXT_COL].astype(str).apply(lambda t: len(t.split()))

print("Estadísticas de longitud de texto (caracteres):")
print(df_raw["text_length"].describe())

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

sns.histplot(df_raw["text_length"], bins=50, ax=axes[0], color="steelblue")
axes[0].axvline(MIN_TEXT_LENGTH, color="red", linestyle="--", label=f"min={MIN_TEXT_LENGTH}")
axes[0].axvline(MAX_TEXT_LENGTH, color="red", linestyle="--", label=f"max={MAX_TEXT_LENGTH}")
axes[0].set_title("Distribución de longitud (caracteres)")
axes[0].legend()

sns.boxplot(x=LABEL_COL, y="text_length", data=df_raw, ax=axes[1])
axes[1].set_title("Longitud de texto por categoría")
axes[1].tick_params(axis="x", rotation=75)

plt.tight_layout()
plt.show()

print(f"\nRegistros con longitud < {MIN_TEXT_LENGTH}: "
      f"{(df_raw['text_length'] < MIN_TEXT_LENGTH).sum()}")
print(f"Registros con longitud > {MAX_TEXT_LENGTH}: "
      f"{(df_raw['text_length'] > MAX_TEXT_LENGTH).sum()}")

"""## 5. Limpieza de datos

Pasos aplicados:
1. Eliminar filas con texto o etiqueta nula/vacía.
2. Normalizar espacios en blanco (colapsar espacios múltiples, quitar saltos de línea sobrantes).
3. Eliminar duplicados exactos en `citing-sentence` (mismo texto + misma etiqueta).
4. Filtrar textos demasiado cortos (probable ruido/truncamiento) o demasiado largos (probable error de parseo).
5. Verificar que las etiquetas correspondan exactamente a las 9 categorías esperadas.
"""

EXPECTED_LABELS = {
    "Background",
    "Further Reading",
    "Evidence",
    "Basis",
    "Application",
    "Improvement / Modification",
    "Identification of the Originator",
    "Gap",
    "Comparison",
}

def normalize_whitespace(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


df = df_raw.copy()
n_inicial = len(df)

# 1. Nulos / vacíos
df = df.dropna(subset=[TEXT_COL, LABEL_COL])
df = df[df[TEXT_COL].astype(str).str.strip() != ""]
n_tras_nulos = len(df)

# 2. Normalizar texto
df[TEXT_COL] = df[TEXT_COL].apply(normalize_whitespace)
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

# 3. Duplicados exactos (texto + etiqueta)
df = df.drop_duplicates(subset=[TEXT_COL, LABEL_COL])
n_tras_duplicados = len(df)

# 4. Filtrar por longitud
df["text_length"] = df[TEXT_COL].apply(len)
df = df[(df["text_length"] >= MIN_TEXT_LENGTH) & (df["text_length"] <= MAX_TEXT_LENGTH)]
n_tras_longitud = len(df)

# 5. Verificar etiquetas válidas
etiquetas_invalidas = set(df[LABEL_COL].unique()) - EXPECTED_LABELS
if etiquetas_invalidas:
    print(f"ADVERTENCIA: etiquetas inesperadas encontradas: {etiquetas_invalidas}")
    df = df[df[LABEL_COL].isin(EXPECTED_LABELS)]
n_final = len(df)

print("Resumen:")
print(f"  Filas iniciales:                    {n_inicial}")
print(f"  Tras eliminar nulos/vacíos:         {n_tras_nulos}  (-{n_inicial - n_tras_nulos})")
print(f"  Tras eliminar duplicados:           {n_tras_duplicados}  (-{n_tras_nulos - n_tras_duplicados})")
print(f"  Tras filtrar por longitud:          {n_tras_longitud}  (-{n_tras_duplicados - n_tras_longitud})")
print(f"  Tras validar etiquetas:             {n_final}  (-{n_tras_longitud - n_final})")
print(f"\n  Total descartado: {n_inicial - n_final} filas ({(n_inicial - n_final) / n_inicial * 100:.1f}%)")

df = df.reset_index(drop=True)
df[[TEXT_COL, LABEL_COL]].head(10)

"""### 5.1 Distribución de clases después de la limpieza"""

label_counts_clean = df[LABEL_COL].value_counts()
print(label_counts_clean)

plt.figure(figsize=(9, 5))
sns.barplot(x=label_counts_clean.values, y=label_counts_clean.index, palette="mako")
plt.title("Distribución de categorías después de la limpieza")
plt.xlabel("Registros")
plt.ylabel("Categoría")
plt.tight_layout()
plt.show()

## 6. Balanceo de categorías

BALANCE_STRATEGY = "submuestreo_balanceado + class_weights_en_cabeza"

n_por_clase = min(MUESTRA_POR_CLASE, int(df[LABEL_COL].value_counts().min()))

df_balanced = pd.concat(
    [g.sample(n=n_por_clase, random_state=RANDOM_SEED)
     for _, g in df.groupby(LABEL_COL)]
).reset_index(drop=True)

print(f"Submuestreo balanceado: {n_por_clase} ejemplos por clase "
      f"-> {len(df_balanced)} filas en total")
print(df_balanced[LABEL_COL].value_counts())

"""## 7. Mapeo de etiquetas a IDs numéricos"""

labels_sorted = sorted(df_balanced[LABEL_COL].unique().tolist())
label2id = {label: idx for idx, label in enumerate(labels_sorted)}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(labels_sorted)

print(f"Esquema de etiquetas ({num_labels} clases):")
for label, idx in label2id.items():
    print(f"  {idx}: {label}")

df_balanced["label_id"] = df_balanced[LABEL_COL].map(label2id)

"""## 8. División train / validation / test (estratificada)
"""

TEST_SIZE = 0.15
EVAL_SIZE = 0.15

train_df, temp_df = train_test_split(
    df_balanced,
    test_size=(TEST_SIZE + EVAL_SIZE),
    stratify=df_balanced["label_id"],
    random_state=RANDOM_SEED,
)

relative_test_size = TEST_SIZE / (TEST_SIZE + EVAL_SIZE)
val_df, test_df = train_test_split(
    temp_df,
    test_size=relative_test_size,
    stratify=temp_df["label_id"],
    random_state=RANDOM_SEED,
)

print(f"Train:      {len(train_df)} ejemplos")
print(f"Validation: {len(val_df)} ejemplos")
print(f"Test:       {len(test_df)} ejemplos")

"""## 9. Extracción de embeddings con BERT large congelado

El modelo se carga en modo evaluación y todo el cálculo va dentro de
`torch.no_grad()`: no se construye grafo de gradientes y ningún peso se
actualiza. El resultado se guarda en disco, de modo que reajustar la cabeza con
otros hiperparámetros no obliga a repetir el paso hacia adelante.
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {device}")

print(f"Cargando tokenizer y modelo: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME)
encoder.eval()
encoder.to(device)

# Confirmación explícita de que el encoder está congelado.
for parametro in encoder.parameters():
    parametro.requires_grad = False

n_params = sum(p.numel() for p in encoder.parameters())
n_entrenables = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
print(f"Parámetros del encoder: {n_params:,} | entrenables: {n_entrenables:,}")

CACHE_DIR = OUTPUT_DIR / "embeddings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def extraer_embeddings(textos, nombre_cache):
    """Devuelve la matriz [N, hidden] de embeddings del encoder congelado."""
    ruta = CACHE_DIR / f"{nombre_cache}_{POOLING}_len{MAX_LENGTH}_n{len(textos)}.npy"
    if ruta.exists():
        print(f"  embeddings en caché: {ruta.name}")
        return np.load(ruta), 0.0

    t0 = time.time()
    salidas = []
    with torch.no_grad():
        for inicio in range(0, len(textos), BATCH_SIZE_EMB):
            lote = textos[inicio:inicio + BATCH_SIZE_EMB]
            enc = tokenizer(lote, truncation=True, padding="max_length",
                            max_length=MAX_LENGTH, return_tensors="pt").to(device)
            oculto = encoder(**enc).last_hidden_state       # [B, T, hidden]

            if POOLING == "cls":
                vector = oculto[:, 0, :]
            else:
                # Media enmascarada: el padding no debe contaminar el promedio.
                mascara = enc["attention_mask"].unsqueeze(-1).float()
                vector = (oculto * mascara).sum(1) / mascara.sum(1).clamp(min=1e-9)

            salidas.append(vector.cpu().numpy())

            if (inicio // BATCH_SIZE_EMB) % 20 == 0:
                print(f"    {min(inicio + BATCH_SIZE_EMB, len(textos))}/{len(textos)}",
                      flush=True)

    matriz = np.vstack(salidas)
    segundos = time.time() - t0
    np.save(ruta, matriz)
    print(f"  {nombre_cache}: {matriz.shape} en {segundos:.1f} s")
    return matriz, segundos


print("\nExtrayendo embeddings...")
X_train, s_train = extraer_embeddings(train_df[TEXT_COL].tolist(), "train")
X_val, s_val = extraer_embeddings(val_df[TEXT_COL].tolist(), "val")
X_test, s_test = extraer_embeddings(test_df[TEXT_COL].tolist(), "test")

y_train = train_df["label_id"].values
y_val = val_df["label_id"].values
y_test = test_df["label_id"].values

segundos_embeddings = s_train + s_val + s_test
print(f"Tiempo total de extracción: {segundos_embeddings:.1f} s")

"""## 10. Ajuste de la cabeza lineal

Único componente que se entrena. El encoder queda intacto.
"""

# Se abre la corrida de MLflow aquí, que es cuando ya están definidos todos los
# parámetros que se quieren registrar. Se usa la forma imperativa
# (start_run / end_run) para no tener que indentar el resto del script.
mlflow.start_run(
    experiment_id=experiment.experiment_id,
    run_name=f"bert-large-{POOLING}-len{MAX_LENGTH}-{CABEZA}",
)

mlflow.log_param("model_base", MODEL_NAME)
mlflow.log_param("fine_tuning", False)
mlflow.log_param("encoder_congelado", True)
mlflow.log_param("balance_strategy", BALANCE_STRATEGY)
mlflow.log_param("max_length", MAX_LENGTH)
mlflow.log_param("pooling", POOLING)
mlflow.log_param("cabeza", CABEZA)
mlflow.log_param("C", C_REG)
mlflow.log_param("class_weight", CLASS_WEIGHT)
mlflow.log_param("muestra_por_clase", n_por_clase)
mlflow.log_param("train_size", len(train_df))
mlflow.log_param("val_size", len(val_df))
mlflow.log_param("test_size", len(test_df))
mlflow.log_param("num_labels", num_labels)

mlflow.set_tag("equipo", "Grupo 24")
mlflow.set_tag("tipo_modelo", "encoder sin fine-tuning")

print("\nAjustando la cabeza lineal sobre los embeddings congelados...")
t0 = time.time()
cabeza = LogisticRegression(
    C=C_REG,
    max_iter=2000,
    class_weight=CLASS_WEIGHT,
    random_state=RANDOM_SEED,
)
cabeza.fit(X_train, y_train)
segundos_cabeza = time.time() - t0
print(f"Cabeza ajustada en {segundos_cabeza:.1f} s")

# El set de validación no interviene en el ajuste; sirve de control.
val_preds = cabeza.predict(X_val)
print(f"Accuracy en validación: {accuracy_score(y_val, val_preds):.4f}")

"""## 11. Evaluación sobre el conjunto de test"""

preds = cabeza.predict(X_test)
true_labels = y_test

target_names = [id2label[i] for i in range(num_labels)]
report = classification_report(true_labels, preds, target_names=target_names, zero_division=0)

print("Reporte de clasificación (test):\n")
print(report)

# --- Registro de las métricas de test en MLflow ---
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    true_labels, preds, average="macro", zero_division=0
)
precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
    true_labels, preds, average="weighted", zero_division=0
)

metricas = {
    "accuracy": accuracy_score(true_labels, preds),
    "f1_macro": f1_macro,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_weighted": f1_w,
    "precision_weighted": precision_w,
    "recall_weighted": recall_w,
    "segundos_embeddings": segundos_embeddings,
    "segundos_cabeza": segundos_cabeza,
}

print("\nMétricas registradas en MLflow:")
for nombre, valor in metricas.items():
    mlflow.log_metric(nombre, float(valor))
    print(f"  {nombre}: {valor:.4f}")

"""### 11.1 Matriz de confusión"""

cm = confusion_matrix(true_labels, preds)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matriz de confusión — {MODEL_NAME} sin fine-tuning")
plt.ylabel("Etiqueta real")
plt.xlabel("Etiqueta predicha")
plt.xticks(rotation=75)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

"""## 12. Guardar artefactos y metadatos"""

final_dir = OUTPUT_DIR / "bert_large_linear_probe"
final_dir.mkdir(parents=True, exist_ok=True)

with open(final_dir / "label_mapping.json", "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

with open(final_dir / "test_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

metadata = {
    "model_base": MODEL_NAME,
    "fine_tuning": False,
    "balance_strategy": BALANCE_STRATEGY,
    "max_length": MAX_LENGTH,
    "pooling": POOLING,
    "cabeza": CABEZA,
    "C": C_REG,
    "muestra_por_clase": n_por_clase,
    "train_size": len(train_df),
    "val_size": len(val_df),
    "test_size": len(test_df),
    "num_labels": num_labels,
}
with open(final_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

mlflow.log_artifact(str(final_dir / "test_classification_report.txt"))

print(f"Artefactos guardados en: {final_dir.resolve()}")

"""## 13. Cierre de la corrida en MLflow"""

mlflow.end_run(status="FINISHED")

run_id = mlflow.last_active_run().info.run_id
print(f"Corrida registrada en MLflow: {MLFLOW_TRACKING_URI}"
      f"/#/experiments/{experiment.experiment_id}/runs/{run_id}")
