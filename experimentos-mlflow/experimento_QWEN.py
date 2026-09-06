# -*- coding: utf-8 -*-
"""experimento_QWEN.py

# Qwen3 14B — clasificación zero-shot / few-shot


Qwen3 14B es un modelo generativo instruido, describe la taxonomía de 
9 funciones de cita en el prompt y se le
pide que devuelva la categoría de cada *citing sentence*. No se ajustan pesos
ni se entrena una cabeza; el "aprendizaje" ocurre únicamente dentro del prompt.

Esto corresponde a la actividad A5 del proyecto (evaluación de LLMs en modo
inferencia comparando estrategias de prompting). La taxonomía del prompt son
las definiciones operativas de `problema.md`, sección 2.

Dos backends, seleccionables con `BACKEND`:

  "transformers"    Pesos locales desde el hub. Qwen3 14B necesita ~28 GB en
                    bf16, o ~10 GB con carga en 4 bits (`LOAD_IN_4BIT`).
                    Requiere GPU: en CPU cada ejemplo tardaría minutos.
  "openai_compat"   Endpoint compatible con la API de OpenAI (vLLM, Ollama,
                    TGI, DashScope). No descarga pesos; solo hace peticiones
                    HTTP. Es la opción práctica si el modelo ya está servido.

Registro en MLflow (experimento `modelos-openweight`):

- Parámetros: `model_base`, `backend`, `n_shot`, `temperature`,
  `max_new_tokens`, `enable_thinking`, `balance_strategy`, `max_length`,
  `train_size`, `val_size`, `test_size`, `num_labels`, `eval_por_clase`.
- Métricas: `accuracy`, `f1_macro`, `precision_macro`, `recall_macro`,
  `f1_weighted`, `precision_weighted`, `recall_weighted`, más `tasa_parseo`,
  `latencia_media_s`, `latencia_p95_s` y `segundos_totales`.

## 1. Importar librerías
"""

# !pip install transformers torch accelerate scikit-learn pandas matplotlib seaborn mlflow requests --quiet
# !pip install bitsandbytes --quiet   # solo si se usa LOAD_IN_4BIT

import re
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import mlflow

sns.set_theme(style="whitegrid")
pd.set_option("display.max_colwidth", 120)

"""## 2. Configuración"""

BASE_NAME = "final_labelled_citing_sentences_all"
#INPUT_CSV = Path(f"/content/drive/MyDrive/Colab Notebooks/microproyecto/data/{BASE_NAME}.csv")
INPUT_CSV = Path(f"/content/{BASE_NAME}.csv")

TEXT_COL = "citing-sentence"
LABEL_COL = "label"

# Modelo tal como viene del hub, sin fine-tuning.
MODEL_NAME = "Qwen/Qwen3-14B"
OUTPUT_DIR = Path("/content/data/qwen3-14b-sin-finetuning")

# "transformers" (pesos locales, requiere GPU) | "openai_compat" (endpoint HTTP)
BACKEND = "transformers"
LOAD_IN_4BIT = True

# Solo para BACKEND = "openai_compat".
API_BASE_URL = os.getenv("QWEN_API_BASE", "http://localhost:8000/v1")
API_KEY = os.getenv("QWEN_API_KEY", "EMPTY")

# Umbrales de limpieza por longitud del texto (en caracteres)
MIN_TEXT_LENGTH = 15
MAX_TEXT_LENGTH = 2000

# Submuestreo balanceado del dataset completo.
MUESTRA_POR_CLASE = 500

# Ejemplos por clase que se envían al modelo. La inferencia generativa es
# lenta: 30 por clase son 270 peticiones, unos minutos en GPU. Subirlo encarece
# el experimento de forma lineal.
EVAL_POR_CLASE = 30

# Estrategia de prompting. N_SHOT = 0 es zero-shot; con N_SHOT = 1 se añade un
# ejemplo por clase (9 ejemplos) tomados del conjunto de entrenamiento.
N_SHOT = 0
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 16

# Qwen3 admite modo de razonamiento explícito. Para clasificación conviene
# desactivarlo: alarga la respuesta y no mejora una etiqueta de una palabra.
ENABLE_THINKING = False

# Longitud máxima del texto de entrada en caracteres (recorte defensivo).
MAX_LENGTH = 2000

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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

"""## 6. Balanceo de categorías
"""

BALANCE_STRATEGY = "submuestreo_balanceado (sin entrenamiento)"

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

# Subconjunto balanceado del test sobre el que se hará la inferencia.
n_eval = min(EVAL_POR_CLASE, int(test_df[LABEL_COL].value_counts().min()))
eval_df = pd.concat(
    [g.sample(n=n_eval, random_state=RANDOM_SEED)
     for _, g in test_df.groupby(LABEL_COL)]
).reset_index(drop=True)

print(f"\nSubconjunto de evaluación: {n_eval} por clase -> {len(eval_df)} peticiones")

"""## 9. Taxonomía y construcción del prompt

Las definiciones son las de `problema.md`, sección 2. Se le entregan al modelo
en el prompt porque es lo único que sabrá de la tarea: no hay entrenamiento.
"""

TAXONOMIA = {
    "Background": "Citations used to provide context, summarize the general background of a research topic, or trace the history of a field or idea.",
    "Gap": "Citations that help identify research gaps or unexplored areas, justifying the author's choice of research topic.",
    "Basis": "Citations that provide a foundation for the current research; the cited work shapes the research idea or hypothesis, and the focal research builds upon or continues it.",
    "Comparison": "Citations used to compare the current work with cited studies, or to draw comparisons between cited studies (similarities, differences, advantages).",
    "Application": "Citations that directly employ a method, technique, tool or data from the cited work WITHOUT modification.",
    "Improvement / Modification": "Citations in which methods or tools from the cited work are adapted, extended or modified for the current research.",
    "Evidence": "Citations used to support claims, hypotheses or findings, or to justify a research design or experimental procedure.",
    "Identification of the Originator": "Citations used to acknowledge the original source of an idea, concept, method or theory, or to credit pioneers in the field.",
    "Further Reading": "Citations that direct readers to additional or supplementary literature for more detail or context.",
}

BLOQUE_TAXONOMIA = "\n".join(
    f"- {nombre}: {definicion}" for nombre, definicion in TAXONOMIA.items()
)

SYSTEM_PROMPT = (
    "You are an expert annotator of scientific citation functions. "
    "Given a sentence from a scientific paper containing a citation marker, "
    "you classify the rhetorical function of that citation into exactly one of "
    "the following nine categories:\n\n"
    f"{BLOQUE_TAXONOMIA}\n\n"
    "Answer with the category name only, exactly as written above. "
    "Do not explain, do not add punctuation, do not output anything else."
)


def construir_ejemplos_few_shot(n_shot, train_pool):
    """Devuelve n_shot ejemplos por clase, tomados del conjunto de train."""
    if n_shot <= 0:
        return []
    ejemplos = []
    for etiqueta, grupo in train_pool.groupby(LABEL_COL):
        k = min(n_shot, len(grupo))
        for _, fila in grupo.sample(n=k, random_state=RANDOM_SEED).iterrows():
            ejemplos.append((fila[TEXT_COL], etiqueta))
    # Se barajan para no inducir un sesgo por el orden de las categorías.
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(ejemplos)
    return ejemplos


EJEMPLOS_FEW_SHOT = construir_ejemplos_few_shot(N_SHOT, train_df)
print(f"Estrategia de prompting: {'zero-shot' if N_SHOT == 0 else f'{N_SHOT}-shot'} "
      f"({len(EJEMPLOS_FEW_SHOT)} ejemplos en el prompt)")


def construir_mensajes(texto):
    """Arma la lista de mensajes del chat para una citing sentence."""
    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ejemplo_texto, ejemplo_etiqueta in EJEMPLOS_FEW_SHOT:
        mensajes.append({"role": "user", "content": f"Sentence: {ejemplo_texto[:MAX_LENGTH]}"})
        mensajes.append({"role": "assistant", "content": ejemplo_etiqueta})
    mensajes.append({"role": "user", "content": f"Sentence: {texto[:MAX_LENGTH]}"})
    return mensajes

"""## 10. Backend de inferencia

"""

if BACKEND == "transformers":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Cargando {MODEL_NAME} (4 bits: {LOAD_IN_4BIT})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    kwargs_modelo = {"device_map": "auto"}
    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs_modelo["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs_modelo["dtype"] = torch.bfloat16

    modelo = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs_modelo)
    modelo.eval()

    n_params = sum(p.numel() for p in modelo.parameters())
    print(f"Parámetros cargados: {n_params:,} (ninguno se va a actualizar)")

    def aplicar_plantilla(mensajes):
        """Aplica la plantilla de chat, desactivando el modo de razonamiento.

        `enable_thinking` es específico de Qwen3; si la versión instalada de
        transformers o la plantilla del modelo no lo aceptan, se ignora.
        """
        try:
            return tokenizer.apply_chat_template(
                mensajes, tokenize=False, add_generation_prompt=True,
                enable_thinking=ENABLE_THINKING,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                mensajes, tokenize=False, add_generation_prompt=True,
            )

    def clasificar(texto):
        """Devuelve (respuesta_cruda, tokens_generados)."""
        prompt = aplicar_plantilla(construir_mensajes(texto))
        entradas = tokenizer(prompt, return_tensors="pt").to(modelo.device)

        with torch.no_grad():
            salida = modelo.generate(
                **entradas,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=TEMPERATURE > 0,
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )

        generados = salida[0][entradas["input_ids"].shape[1]:]
        return tokenizer.decode(generados, skip_special_tokens=True), len(generados)

elif BACKEND == "openai_compat":
    import requests

    print(f"Usando endpoint compatible con OpenAI: {API_BASE_URL}")

    def clasificar(texto):
        """Devuelve (respuesta_cruda, tokens_generados)."""
        cuerpo = {
            "model": MODEL_NAME,
            "messages": construir_mensajes(texto),
            "temperature": TEMPERATURE,
            "max_tokens": MAX_NEW_TOKENS,
        }
        respuesta = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=cuerpo,
            timeout=120,
        )
        respuesta.raise_for_status()
        datos = respuesta.json()
        texto_salida = datos["choices"][0]["message"]["content"]
        tokens = datos.get("usage", {}).get("completion_tokens", 0)
        return texto_salida, tokens

else:
    raise ValueError(f"BACKEND desconocido: {BACKEND}")

"""## 11. Parseo de la respuesta

Un modelo generativo puede responder con variantes ("Improvement", "the
category is Basis", "1. Background"). Se normaliza la respuesta y se busca la
categoría; lo que no se pueda mapear se cuenta aparte y se trata como error,
que es lo honesto: en producción esa respuesta tampoco serviría.
"""

NO_PARSEADO = -1

# Claves de búsqueda: nombre completo normalizado y algunos alias frecuentes.
ALIAS = {}
for nombre in TAXONOMIA:
    ALIAS[nombre.lower()] = nombre
ALIAS["improvement"] = "Improvement / Modification"
ALIAS["modification"] = "Improvement / Modification"
ALIAS["improvement/modification"] = "Improvement / Modification"
ALIAS["originator"] = "Identification of the Originator"
ALIAS["identification"] = "Identification of the Originator"
ALIAS["further reading"] = "Further Reading"


def parsear_etiqueta(respuesta):
    """Convierte la respuesta del modelo en un label_id, o NO_PARSEADO."""
    if not respuesta:
        return NO_PARSEADO

    # Qwen3 puede emitir un bloque de razonamiento aunque se le pida no hacerlo.
    texto = re.sub(r"<think>.*?</think>", " ", respuesta, flags=re.DOTALL)
    texto = texto.strip().strip(".\"'` \n")
    normalizado = re.sub(r"\s+", " ", texto.lower())

    if normalizado in ALIAS:
        return label2id[ALIAS[normalizado]]

    # Coincidencia por contención, de la clave más larga a la más corta para no
    # confundir "Improvement / Modification" con "Improvement".
    for clave in sorted(ALIAS, key=len, reverse=True):
        if clave in normalizado:
            return label2id[ALIAS[clave]]

    return NO_PARSEADO

"""## 12. Inferencia sobre el subconjunto de evaluación"""

# Se abre la corrida de MLflow aquí, que es cuando ya están definidos todos los
# parámetros que se quieren registrar. Se usa la forma imperativa
# (start_run / end_run) para no tener que indentar el resto del script.
mlflow.start_run(
    experiment_id=experiment.experiment_id,
    run_name=f"qwen3-14b-{'zeroshot' if N_SHOT == 0 else f'{N_SHOT}shot'}-{BACKEND}",
)

mlflow.log_param("model_base", MODEL_NAME)
mlflow.log_param("fine_tuning", False)
mlflow.log_param("backend", BACKEND)
mlflow.log_param("load_in_4bit", LOAD_IN_4BIT)
mlflow.log_param("n_shot", N_SHOT)
mlflow.log_param("temperature", TEMPERATURE)
mlflow.log_param("max_new_tokens", MAX_NEW_TOKENS)
mlflow.log_param("enable_thinking", ENABLE_THINKING)
mlflow.log_param("balance_strategy", BALANCE_STRATEGY)
mlflow.log_param("max_length", MAX_LENGTH)
mlflow.log_param("muestra_por_clase", n_por_clase)
mlflow.log_param("eval_por_clase", n_eval)
mlflow.log_param("train_size", len(train_df))
mlflow.log_param("val_size", len(val_df))
mlflow.log_param("test_size", len(test_df))
mlflow.log_param("num_labels", num_labels)

mlflow.set_tag("equipo", "Grupo 24")
mlflow.set_tag("tipo_modelo", "LLM generativo sin fine-tuning")
mlflow.set_tag("estrategia", "zero-shot" if N_SHOT == 0 else f"{N_SHOT}-shot")

print(f"\nClasificando {len(eval_df)} ejemplos con {MODEL_NAME}...\n")

predicciones, respuestas_crudas, latencias = [], [], []
tokens_generados = 0
t_inicio = time.time()

for posicion, fila in eval_df.iterrows():
    t0 = time.time()
    try:
        respuesta, n_tokens = clasificar(fila[TEXT_COL])
    except Exception as exc:                      # noqa: BLE001
        respuesta, n_tokens = "", 0
        print(f"  [{posicion}] error de inferencia: {type(exc).__name__}: {exc}")

    latencias.append(time.time() - t0)
    tokens_generados += n_tokens
    respuestas_crudas.append(respuesta)
    predicciones.append(parsear_etiqueta(respuesta))

    if (posicion + 1) % 25 == 0:
        print(f"  {posicion + 1}/{len(eval_df)} "
              f"({np.mean(latencias):.2f} s por ejemplo)", flush=True)

segundos_totales = time.time() - t_inicio
predicciones = np.array(predicciones)
true_labels = eval_df["label_id"].values

n_parseadas = int((predicciones != NO_PARSEADO).sum())
tasa_parseo = n_parseadas / len(predicciones)
print(f"\nInferencia terminada en {segundos_totales:.1f} s")
print(f"Respuestas mapeadas a una categoría válida: {n_parseadas}/{len(predicciones)} "
      f"({tasa_parseo:.1%})")

"""## 13. Evaluación"""

target_names = [id2label[i] for i in range(num_labels)]
etiquetas_validas = list(range(num_labels))

report = classification_report(
    true_labels, predicciones, labels=etiquetas_validas,
    target_names=target_names, zero_division=0,
)

print("Reporte de clasificación (subconjunto de test):\n")
print(report)

# --- Registro de las métricas en MLflow ---
# Las respuestas no parseables quedan como NO_PARSEADO: no cuentan como acierto
# de ninguna clase, así que penalizan el recall de su clase real.
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    true_labels, predicciones, labels=etiquetas_validas, average="macro", zero_division=0
)
precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
    true_labels, predicciones, labels=etiquetas_validas, average="weighted", zero_division=0
)

metricas = {
    "accuracy": accuracy_score(true_labels, predicciones),
    "f1_macro": f1_macro,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_weighted": f1_w,
    "precision_weighted": precision_w,
    "recall_weighted": recall_w,
    "tasa_parseo": tasa_parseo,
    "latencia_media_s": float(np.mean(latencias)),
    "latencia_p95_s": float(np.percentile(latencias, 95)),
    "segundos_totales": segundos_totales,
    "tokens_generados": tokens_generados,
    "n_evaluados": len(eval_df),
}

print("\nMétricas registradas en MLflow:")
for nombre, valor in metricas.items():
    mlflow.log_metric(nombre, float(valor))
    print(f"  {nombre}: {valor:.4f}")

"""### 13.1 Matriz de confusión"""

cm = confusion_matrix(true_labels, predicciones, labels=etiquetas_validas)
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

"""## 14. Guardar artefactos y metadatos"""

final_dir = OUTPUT_DIR / "qwen3_14b_inferencia"
final_dir.mkdir(parents=True, exist_ok=True)

# Las respuestas crudas son el soporte que permite auditar el parseo.
predicciones_df = pd.DataFrame({
    TEXT_COL: eval_df[TEXT_COL],
    "etiqueta_real": [id2label[i] for i in true_labels],
    "respuesta_modelo": respuestas_crudas,
    "etiqueta_predicha": [id2label[i] if i != NO_PARSEADO else "SIN_PARSEO"
                          for i in predicciones],
    "latencia_s": latencias,
})
predicciones_df.to_csv(final_dir / "predicciones.csv", index=False)

with open(final_dir / "prompt_sistema.txt", "w", encoding="utf-8") as f:
    f.write(SYSTEM_PROMPT)

with open(final_dir / "test_classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

metadata = {
    "model_base": MODEL_NAME,
    "fine_tuning": False,
    "backend": BACKEND,
    "n_shot": N_SHOT,
    "temperature": TEMPERATURE,
    "balance_strategy": BALANCE_STRATEGY,
    "max_length": MAX_LENGTH,
    "muestra_por_clase": n_por_clase,
    "eval_por_clase": n_eval,
    "train_size": len(train_df),
    "val_size": len(val_df),
    "test_size": len(test_df),
    "num_labels": num_labels,
}
with open(final_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

mlflow.log_artifact(str(final_dir / "test_classification_report.txt"))
mlflow.log_artifact(str(final_dir / "predicciones.csv"))
mlflow.log_artifact(str(final_dir / "prompt_sistema.txt"))

print(f"Artefactos guardados en: {final_dir.resolve()}")

"""## 15. Cierre de la corrida en MLflow"""

mlflow.end_run(status="FINISHED")

run_id = mlflow.last_active_run().info.run_id
print(f"Corrida registrada en MLflow: {MLFLOW_TRACKING_URI}"
      f"/#/experiments/{experiment.experiment_id}/runs/{run_id}")
