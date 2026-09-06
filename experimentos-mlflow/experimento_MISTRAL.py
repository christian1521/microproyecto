# -*- coding: utf-8 -*-
"""experimento_MISTRAL.py

# Mistral Small - Clasificación de Función de Cita Científica

1. **API REST oficial** con reintentos y respaldo exponencial ante 429/5xx, en
   vez de descargar 24 000 millones de parámetros. Esto corresponde a la
   actividad A5 del proyecto —evaluación de LLMs comerciales vía API— y a A7,
   el análisis de latencia y costos, para el que se registran los tokens
   consumidos.
2. **Salida estructurada en JSON** (`response_format` de tipo `json_object`),
   que hace el parseo mucho más fiable que leer texto libre y permite pedirle
   al modelo, sin costo adicional, una confianza declarada.

La taxonomía del prompt son las definiciones operativas de `problema.md`,
sección 2.

Backends, seleccionables con `BACKEND`:

  "api"             https://api.mistral.ai — necesita `MISTRAL_API_KEY` en el
                    entorno. Es la ruta recomendada.
  "transformers"    Pesos locales desde el hub. Mistral Small son 24 B de
                    parámetros: ~48 GB en bf16 o ~14 GB en 4 bits, así que
                    exige una GPU grande.

Registro en MLflow (experimento `modelos-openweight `):

- Parámetros: `model_base`, `backend`, `n_shot`, `temperature`,
  `max_new_tokens`, `formato_respuesta`, `balance_strategy`, `max_length`,
  `train_size`, `val_size`, `test_size`, `num_labels`, `eval_por_clase`.
- Métricas: `accuracy`, `f1_macro`, `precision_macro`, `recall_macro`,
  `f1_weighted`, `precision_weighted`, `recall_weighted`, más `tasa_parseo`,
  `latencia_media_s`, `latencia_p95_s`, `tokens_prompt`, `tokens_completion`,
  `n_reintentos` y `costo_estimado_usd`.

## 1. Importar librerías
"""

# !pip install requests scikit-learn pandas matplotlib seaborn mlflow --quiet
# !pip install transformers torch accelerate bitsandbytes --quiet   # solo para BACKEND="transformers"

import re
import os
import json
import time
import random
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

# "api" (REST oficial) | "transformers" (pesos locales, requiere GPU grande)
BACKEND = "api"

# Identificador del modelo. Cambia según el backend: el alias de la API no es
# el mismo que el repositorio del hub. Conviene confirmar la versión vigente en
# https://docs.mistral.ai/getting-started/models/ antes de una corrida final,
# porque los alias "-latest" se reasignan a versiones nuevas.
MODEL_NAME_API = "mistral-small-latest"
MODEL_NAME_LOCAL = "mistralai/Mistral-Small-24B-Instruct-2501"
MODEL_NAME = MODEL_NAME_API if BACKEND == "api" else MODEL_NAME_LOCAL

OUTPUT_DIR = Path("/content/data/mistral-small-sin-finetuning")

# Credenciales: nunca escribir la clave en el archivo. En Colab:
#   import os; os.environ["MISTRAL_API_KEY"] = getpass.getpass()
API_URL = os.getenv("MISTRAL_API_URL", "https://api.mistral.ai/v1/chat/completions")
API_KEY = os.getenv("MISTRAL_API_KEY", "")

# Precios por millón de tokens. Se dejan en cero a propósito: hay que copiarlos
# de la página de precios vigente en el momento de la corrida. Con 0.0 la
# métrica de costo se registra como 0 y no como un número inventado.
PRECIO_INPUT_POR_MTOK = 0.0
PRECIO_OUTPUT_POR_MTOK = 0.0

LOAD_IN_4BIT = True

# Umbrales de limpieza por longitud del texto (en caracteres)
MIN_TEXT_LENGTH = 15
MAX_TEXT_LENGTH = 2000

# Submuestreo balanceado del dataset completo.
MUESTRA_POR_CLASE = 500

# Ejemplos por clase que se envían al modelo. Cada uno es una petición de red
# facturada: 30 por clase son 270 llamadas.
EVAL_POR_CLASE = 30

# Estrategia de prompting. N_SHOT = 0 es zero-shot.
N_SHOT = 0
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 64          # más holgado que en Qwen porque la salida es JSON
FORMATO_RESPUESTA = "json_object"

# Reintentos ante límites de tasa o errores transitorios del servidor.
MAX_REINTENTOS = 5
ESPERA_BASE_S = 2.0
PAUSA_ENTRE_PETICIONES_S = 0.0   # subir si la cuenta tiene un límite estrecho

# Longitud máxima del texto de entrada en caracteres (recorte defensivo).
MAX_LENGTH = 2000

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

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

A diferencia de `experimento_QWEN.py`, aquí se le exige al modelo una respuesta en JSON con dos
campos: la categoría y una confianza declarada entre 0 y 1. El campo de
confianza es lo que alimenta el indicador de confianza del demostrador web.
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
    f'- "{nombre}": {definicion}' for nombre, definicion in TAXONOMIA.items()
)

SYSTEM_PROMPT = (
    "You are an expert annotator of scientific citation functions. "
    "Given a sentence from a scientific paper containing a citation marker, "
    "classify the rhetorical function of that citation into exactly one of "
    "these nine categories:\n\n"
    f"{BLOQUE_TAXONOMIA}\n\n"
    "Respond with a single JSON object and nothing else, with this shape:\n"
    '{"category": "<one of the nine category names, copied verbatim>", '
    '"confidence": <a number between 0 and 1>}'
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
        mensajes.append({
            "role": "assistant",
            "content": json.dumps({"category": ejemplo_etiqueta, "confidence": 1.0}),
        })
    mensajes.append({"role": "user", "content": f"Sentence: {texto[:MAX_LENGTH]}"})
    return mensajes

"""## 10. Backend de inferencia
El cliente de la API reintenta ante 429 (límite de tasa) y ante errores 5xx,
con respaldo exponencial y una perturbación aleatoria para no sincronizar los
reintentos. Devuelve además el consumo de tokens, que es lo que alimenta el
análisis de costos de la actividad A7.
"""

contador_reintentos = 0

if BACKEND == "api":
    import requests

    if not API_KEY:
        raise RuntimeError(
            "Falta la clave de la API. Definir MISTRAL_API_KEY en el entorno "
            "antes de ejecutar; no escribirla en este archivo."
        )

    sesion = requests.Session()
    sesion.headers.update({
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    })

    def clasificar(texto):
        """Devuelve (contenido, tokens_prompt, tokens_completion)."""
        global contador_reintentos

        cuerpo = {
            "model": MODEL_NAME,
            "messages": construir_mensajes(texto),
            "temperature": TEMPERATURE,
            "max_tokens": MAX_NEW_TOKENS,
            "response_format": {"type": FORMATO_RESPUESTA},
        }

        for intento in range(MAX_REINTENTOS):
            respuesta = sesion.post(API_URL, json=cuerpo, timeout=120)

            if respuesta.status_code == 200:
                datos = respuesta.json()
                uso = datos.get("usage", {})
                return (
                    datos["choices"][0]["message"]["content"],
                    uso.get("prompt_tokens", 0),
                    uso.get("completion_tokens", 0),
                )

            if respuesta.status_code == 429 or respuesta.status_code >= 500:
                contador_reintentos += 1
                espera = ESPERA_BASE_S * (2 ** intento) + random.uniform(0, 1)
                print(f"    HTTP {respuesta.status_code}, reintento "
                      f"{intento + 1}/{MAX_REINTENTOS} en {espera:.1f} s")
                time.sleep(espera)
                continue

            # 4xx distinto de 429: clave inválida, modelo inexistente, etc.
            respuesta.raise_for_status()

        raise RuntimeError(f"Se agotaron los {MAX_REINTENTOS} reintentos")

elif BACKEND == "transformers":
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

    def clasificar(texto):
        """Devuelve (contenido, tokens_prompt, tokens_completion)."""
        prompt = tokenizer.apply_chat_template(
            construir_mensajes(texto), tokenize=False, add_generation_prompt=True,
        )
        entradas = tokenizer(prompt, return_tensors="pt").to(modelo.device)

        with torch.no_grad():
            salida = modelo.generate(
                **entradas,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=TEMPERATURE > 0,
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )

        n_prompt = entradas["input_ids"].shape[1]
        generados = salida[0][n_prompt:]
        return (tokenizer.decode(generados, skip_special_tokens=True),
                n_prompt, len(generados))

else:
    raise ValueError(f"BACKEND desconocido: {BACKEND}")

"""## 11. Parseo de la respuesta

Se intenta primero leer el JSON pedido. Si el modelo devolviera texto libre
—posible pese a `response_format`— se recurre a buscar el nombre de la
categoría dentro de la cadena. Lo que no se pueda mapear cuenta como error.
"""

NO_PARSEADO = -1

ALIAS = {}
for nombre in TAXONOMIA:
    ALIAS[nombre.lower()] = nombre
ALIAS["improvement"] = "Improvement / Modification"
ALIAS["modification"] = "Improvement / Modification"
ALIAS["improvement/modification"] = "Improvement / Modification"
ALIAS["originator"] = "Identification of the Originator"
ALIAS["identification"] = "Identification of the Originator"


def parsear_respuesta(contenido):
    """Devuelve (label_id, confianza_declarada)."""
    if not contenido:
        return NO_PARSEADO, np.nan

    # 1) Camino esperado: un objeto JSON.
    bloque = re.search(r"\{.*\}", contenido, flags=re.DOTALL)
    if bloque:
        try:
            datos = json.loads(bloque.group(0))
            categoria = str(datos.get("category", "")).strip()
            confianza = datos.get("confidence", np.nan)
            if categoria.lower() in ALIAS:
                return label2id[ALIAS[categoria.lower()]], confianza
        except json.JSONDecodeError:
            pass

    # 2) Respaldo: buscar el nombre de la categoría en el texto, de la clave
    #    más larga a la más corta para no confundir "Improvement" con
    #    "Improvement / Modification".
    normalizado = re.sub(r"\s+", " ", contenido.lower())
    for clave in sorted(ALIAS, key=len, reverse=True):
        if clave in normalizado:
            return label2id[ALIAS[clave]], np.nan

    return NO_PARSEADO, np.nan

"""## 12. Inferencia sobre el subconjunto de evaluación"""

# Se abre la corrida de MLflow aquí, que es cuando ya están definidos todos los
# parámetros que se quieren registrar. Se usa la forma imperativa
# (start_run / end_run) para no tener que indentar el resto del script.
mlflow.start_run(
    experiment_id=experiment.experiment_id,
    run_name=f"mistral-small-{'zeroshot' if N_SHOT == 0 else f'{N_SHOT}shot'}-{BACKEND}",
)

mlflow.log_param("model_base", MODEL_NAME)
mlflow.log_param("fine_tuning", False)
mlflow.log_param("backend", BACKEND)
mlflow.log_param("n_shot", N_SHOT)
mlflow.log_param("temperature", TEMPERATURE)
mlflow.log_param("max_new_tokens", MAX_NEW_TOKENS)
mlflow.log_param("formato_respuesta", FORMATO_RESPUESTA)
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

predicciones, confianzas, respuestas_crudas, latencias = [], [], [], []
tokens_prompt = tokens_completion = 0
t_inicio = time.time()

for posicion, fila in eval_df.iterrows():
    t0 = time.time()
    try:
        contenido, n_prompt, n_completion = clasificar(fila[TEXT_COL])
    except Exception as exc:                      # noqa: BLE001
        contenido, n_prompt, n_completion = "", 0, 0
        print(f"  [{posicion}] error de inferencia: {type(exc).__name__}: {exc}")

    latencias.append(time.time() - t0)
    tokens_prompt += n_prompt
    tokens_completion += n_completion
    respuestas_crudas.append(contenido)

    etiqueta_id, confianza = parsear_respuesta(contenido)
    predicciones.append(etiqueta_id)
    confianzas.append(confianza)

    if PAUSA_ENTRE_PETICIONES_S:
        time.sleep(PAUSA_ENTRE_PETICIONES_S)

    if (posicion + 1) % 25 == 0:
        print(f"  {posicion + 1}/{len(eval_df)} "
              f"({np.mean(latencias):.2f} s por ejemplo)", flush=True)

segundos_totales = time.time() - t_inicio
predicciones = np.array(predicciones)
true_labels = eval_df["label_id"].values

n_parseadas = int((predicciones != NO_PARSEADO).sum())
tasa_parseo = n_parseadas / len(predicciones)
costo_estimado = (tokens_prompt / 1e6 * PRECIO_INPUT_POR_MTOK
                  + tokens_completion / 1e6 * PRECIO_OUTPUT_POR_MTOK)

print(f"\nInferencia terminada en {segundos_totales:.1f} s")
print(f"Respuestas mapeadas a una categoría válida: {n_parseadas}/{len(predicciones)} "
      f"({tasa_parseo:.1%})")
print(f"Tokens: {tokens_prompt:,} de entrada + {tokens_completion:,} de salida "
      f"| reintentos: {contador_reintentos}")
if PRECIO_INPUT_POR_MTOK == 0.0 and PRECIO_OUTPUT_POR_MTOK == 0.0:
    print("Costo estimado: 0.0 USD — faltan los precios vigentes en "
          "PRECIO_INPUT_POR_MTOK / PRECIO_OUTPUT_POR_MTOK")

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
    "tokens_prompt": tokens_prompt,
    "tokens_completion": tokens_completion,
    "n_reintentos": contador_reintentos,
    "costo_estimado_usd": costo_estimado,
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

final_dir = OUTPUT_DIR / "mistral_small_inferencia"
final_dir.mkdir(parents=True, exist_ok=True)

# Las respuestas crudas son el soporte que permite auditar el parseo.
predicciones_df = pd.DataFrame({
    TEXT_COL: eval_df[TEXT_COL],
    "etiqueta_real": [id2label[i] for i in true_labels],
    "respuesta_modelo": respuestas_crudas,
    "etiqueta_predicha": [id2label[i] if i != NO_PARSEADO else "SIN_PARSEO"
                          for i in predicciones],
    "confianza_declarada": confianzas,
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
    "formato_respuesta": FORMATO_RESPUESTA,
    "balance_strategy": BALANCE_STRATEGY,
    "max_length": MAX_LENGTH,
    "muestra_por_clase": n_por_clase,
    "eval_por_clase": n_eval,
    "train_size": len(train_df),
    "val_size": len(val_df),
    "test_size": len(test_df),
    "num_labels": num_labels,
    "tokens_prompt": tokens_prompt,
    "tokens_completion": tokens_completion,
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
