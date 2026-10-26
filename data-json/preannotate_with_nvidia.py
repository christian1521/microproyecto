#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

LABELS = [
    "Background",
    "Gap",
    "Basis",
    "Comparison",
    "Application",
    "Improvement / Modification",
    "Evidence",
    "Identification of the Originator",
    "Further Reading",
]
LABEL_LOOKUP = {l.lower(): l for l in LABELS}
LABEL_LOOKUP.update({
    "improvement": "Improvement / Modification",
    "modification": "Improvement / Modification",
    "improvement/modification": "Improvement / Modification",
    "originator": "Identification of the Originator",
})

SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "primary": {"type": "string", "enum": LABELS},
        "secondary": {"type": "array", "items": {"type": "string", "enum": LABELS}},
        "needs_wider_context": {"type": "boolean"},
    },
    "required": ["rationale", "primary", "secondary", "needs_wider_context"],
    "additionalProperties": False,
}


def load_prompt(path):
    text = open(path, encoding="utf-8").read()
    m = re.search(
        r"SYSTEM PROMPT\n=+\n(.*?)\n=+\nUSER TEMPLATE[^\n]*\n=+\n(.*?)\n=+\nUSAGE NOTES",
        text, re.S)
    if not m:
        sys.exit("No pude separar SYSTEM PROMPT / USER TEMPLATE en el archivo de prompt.")
    return m.group(1).strip(), m.group(2).strip()


def shuffled_system_prompt(system, seed):
    start_tag = "LABELS (use these exact strings)\n\n"
    end_tag = "DECISION RULES FOR FREQUENT CONFUSIONS"
    i, j = system.find(start_tag), system.find(end_tag)
    if i < 0 or j < 0 or seed is None:
        return system
    i += len(start_tag)
    blocks = [b.strip("\n") for b in re.split(r"\n(?=\d\. )", system[i:j].strip("\n")) if b.strip()]
    random.Random(seed).shuffle(blocks)
    blocks = [re.sub(r"^\d\.", f"{n}.", b, count=1) for n, b in enumerate(blocks, 1)]
    return system[:i] + "\n\n".join(blocks) + "\n\n" + system[j:]


def fill_user(template, rec):
    return template.format(
        section=rec.get("section") or "Unknown",
        left_context=rec.get("left_context") or "(none)",
        citing_sentence=rec["citing_sentence"],
        right_context=rec.get("right_context") or "(none)",
    )


def parse_response(text):
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    if not text:
        return None, "empty content (¿modelo de razonamiento sin tokens suficientes? sube --max-tokens)"
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, "no JSON found"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    primary = LABEL_LOOKUP.get(str(obj.get("primary", "")).strip().lower())
    if primary is None:
        return None, f"invalid primary label: {obj.get('primary')!r}"
    secondary = []
    for s in obj.get("secondary") or []:
        lab = LABEL_LOOKUP.get(str(s).strip().lower())
        if lab and lab != primary and lab not in secondary:
            secondary.append(lab)
    return {
        "rationale": str(obj.get("rationale", ""))[:500],
        "primary": primary,
        "secondary": secondary,
        "needs_wider_context": bool(obj.get("needs_wider_context", False)),
    }, None


class ApiError(Exception):
    pass


class RateLimiter:

    def __init__(self, rpm):
        self.interval = 60.0 / max(rpm, 1)
        self.lock = threading.Lock()
        self.next = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next)
            self.next = t + self.interval
        if t > now:
            time.sleep(t - now)


class NvidiaClient:
    MODES = ["schema", "json_object", "off"]

    def __init__(self, args, key):
        self.base = args.base_url.rstrip("/")
        self.args = args
        self.key = key
        self.limiter = RateLimiter(args.rpm)
        self.mode = args.json_mode
        self.lock = threading.Lock()
        self.length_cut = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.n_calls = 0

    def _request(self, method, path, payload=None, timeout=120):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Authorization": f"Bearer {self.key}",
                     "Accept": "application/json",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def list_models(self):
        try:
            return sorted(m["id"] for m in self._request("GET", "/models").get("data", []))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                sys.exit(f"HTTP {e.code}: la clave NVIDIA_API_KEY no es válida o no tiene permiso. "
                         "Genera otra en https://build.nvidia.com/settings/api-keys")
            raise
        except Exception as e:
            sys.exit(f"No pude consultar {self.base}/models ({e})")

    def _format_payload(self, mode):
        if mode == "schema":
            return {"type": "json_schema",
                    "json_schema": {"name": "citation_function", "schema": SCHEMA, "strict": True}}
        if mode == "json_object":
            return {"type": "json_object"}
        return None

    def chat(self, system, user):
        a = self.args
        for attempt in range(a.http_retries + 1):
            mode = self.mode
            payload = {"model": a.model, "stream": False, "temperature": a.temperature,
                       "max_tokens": a.max_tokens,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}
            if a.seed is not None:
                payload["seed"] = a.seed
            payload.update(a.extra)  # p. ej. chat_template_kwargs para apagar el razonamiento
            rf = self._format_payload(mode)
            if rf:
                payload["response_format"] = rf
            self.limiter.wait()
            try:
                resp = self._request("POST", "/chat/completions", payload, a.timeout)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "ignore")
                if e.code == 429 or e.code >= 500:
                    ra = e.headers.get("Retry-After") if e.headers else None
                    pause = float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(5 * 2 ** attempt, 90)
                    print(f"  HTTP {e.code}; espero {pause:.0f}s", file=sys.stderr)
                    time.sleep(pause)
                    continue
                if e.code in (400, 422) and rf is not None:
                    with self.lock:  # bajar un nivel: schema -> json_object -> off
                        if self.mode == mode:
                            self.mode = self.MODES[self.MODES.index(mode) + 1]
                            print(f"AVISO: el modelo rechazó response_format={mode}; "
                                  f"paso a '{self.mode}'.\n  {body[:200]}", file=sys.stderr)
                    continue
                if e.code in (401, 403):
                    sys.exit(f"HTTP {e.code}: clave no válida o sin permiso para este modelo.\n{body[:200]}")
                if e.code == 404:
                    sys.exit(f"HTTP 404: modelo o ruta no encontrados ({a.model}). "
                             "Usa --list-models para ver los disponibles.")
                raise ApiError(f"HTTP {e.code}: {body[:300]}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                time.sleep(min(5 * (attempt + 1), 30))
                last = e
                continue
            usage = resp.get("usage") or {}
            choice = (resp.get("choices") or [{}])[0]
            with self.lock:
                self.n_calls += 1
                self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
                self.completion_tokens += usage.get("completion_tokens", 0) or 0
                if choice.get("finish_reason") == "length":
                    self.length_cut += 1
            return (choice.get("message") or {}).get("content") or ""
        raise ApiError("sin respuesta válida tras varios reintentos")


def make_client(args):
    if args.dry_run:
        rng = random.Random(0)

        class Fake:
            length_cut = n_calls = prompt_tokens = completion_tokens = 0
            mode = "dry"

            def chat(self, system, user):
                return json.dumps({"rationale": "dry run", "primary": rng.choice(LABELS),
                                   "secondary": [], "needs_wider_context": False})

            def list_models(self):
                return ["(dry-run)"]
        return Fake()
    key = os.environ.get(args.api_key_env, "").strip()
    if not key:
        sys.exit(f"Falta la variable de entorno {args.api_key_env}. Genera una clave en "
                 "https://build.nvidia.com/settings/api-keys y exporta: "
                 f'export {args.api_key_env}="nvapi-..."')
    return NvidiaClient(args, key)


def classify(client, system, user, retries):
    last_err = "unknown"
    for attempt in range(retries + 1):
        try:
            parsed, err = parse_response(client.chat(system, user))
            if parsed:
                return parsed, None
            last_err = err
        except ApiError as e:
            last_err = f"ApiError: {e}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(min(2 ** attempt, 10))
    return None, last_err


def annotate_record(client, rec, system, template, passes, retries, model):
    results, errors = [], []
    for k in range(passes):
        sp = shuffled_system_prompt(system, seed=None if k == 0 else k)
        parsed, err = classify(client, sp, fill_user(template, rec), retries)
        if parsed:
            results.append(parsed)
        else:
            errors.append(err)
    primaries = [r["primary"] for r in results]
    agree = len(results) == passes and len(set(primaries)) == 1
    out = dict(rec)
    out.update({
        "llm_model": f"nvidia:{model}",
        "llm_passes": results,
        "llm_errors": errors,
        "llm_agreement": agree,
        "llm_primary": primaries[0] if agree else None,
        "llm_needs_wider_context": any(r["needs_wider_context"] for r in results),
    })
    return out


def to_label_studio(rec, model_name, passes):
    first = (rec["llm_passes"] or [None])[0]
    task = {"data": {k: rec.get(k) for k in (
        "id", "section", "left_context", "citing_sentence", "right_context",
        "citing_paper", "cited_ref_id", "n_citations_in_sentence")}}
    task["data"]["llm_agreement"] = rec["llm_agreement"]
    if first:
        result = [{"from_name": "primary", "to_name": "citing_sentence", "type": "choices",
                   "value": {"choices": [first["primary"]]}}]
        if first["secondary"]:
            result.append({"from_name": "secondary", "to_name": "citing_sentence",
                           "type": "choices", "value": {"choices": first["secondary"]}})
        if rec["llm_needs_wider_context"]:
            result.append({"from_name": "flags", "to_name": "citing_sentence", "type": "choices",
                           "value": {"choices": ["Needs wider context"]}})
        score = sum(r["primary"] == first["primary"] for r in rec["llm_passes"]) / max(passes, 1)
        task["predictions"] = [{"model_version": model_name, "score": round(score, 3),
                                "result": result}]
    return task


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input")
    ap.add_argument("--prompt", default="preannotation_prompt.txt")
    ap.add_argument("--output", default="preannotated.jsonl")
    ap.add_argument("--ls-output", default="label_studio_tasks.json")
    ap.add_argument("--model", default="meta/llama-3.1-70b-instruct")
    ap.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    ap.add_argument("--list-models", action="store_true", help="lista los modelos disponibles y sale")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rpm", type=float, default=35.0, help="peticiones por minuto máximas (compartidas)")
    ap.add_argument("--retries", type=int, default=2, help="reintentos por respuesta inválida")
    ap.add_argument("--http-retries", type=int, default=6, help="reintentos por error HTTP/red")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None, help="semilla (no todos los modelos la respetan)")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--json-mode", choices=["schema", "json_object", "off"], default="schema")
    ap.add_argument("--extra-body", default="",
                    help='JSON con campos extra para la petición, p. ej. '
                         '\'{"chat_template_kwargs": {"enable_thinking": false}}\'')
    ap.add_argument("--extra-body-file", default="",
                    help="igual que --extra-body pero desde un archivo .json "
                         "(evita problemas de comillas en PowerShell)")
    ap.add_argument("--limit", type=int, default=0, help="procesar solo N ejemplos (0 = todos)")
    ap.add_argument("--sample-seed", type=int, default=13)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.extra = {}
    try:
        if args.extra_body_file:
            args.extra.update(json.load(open(args.extra_body_file, encoding="utf-8")))
        if args.extra_body:
            args.extra.update(json.loads(args.extra_body))
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"--extra-body / --extra-body-file no es un JSON válido: {e}")

    client = make_client(args)
    if args.list_models:
        for m in client.list_models():
            print(m)
        return
    if not args.input:
        ap.error("--input es obligatorio (salvo con --list-models)")

    if not args.dry_run:
        models = client.list_models()
        if args.model not in models:
            sug = [m for m in models if "instruct" in m or "chat" in m][:15]
            sys.exit(f"El modelo '{args.model}' no aparece en el catálogo.\n"
                     f"Algunos disponibles: {', '.join(sug) or ', '.join(models[:15])}\n"
                     "Lista completa: --list-models")

    system, template = load_prompt(args.prompt)
    records = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.limit:
        random.Random(args.sample_seed).shuffle(records)
        records = records[:args.limit]

    done = {}
    try:
        for l in open(args.output, encoding="utf-8"):
            r = json.loads(l)
            done[r["id"]] = r
    except FileNotFoundError:
        pass
    todo = [r for r in records if r["id"] not in done]
    calls = len(todo) * args.passes
    print(f"{len(records)} ejemplos | {len(done)} ya procesados | {len(todo)} pendientes")
    if not args.dry_run:
        print(f"Peticiones previstas: {calls} | a {args.rpm:.0f}/min tardará al menos "
              f"{calls / args.rpm:.0f} min")

    t0 = time.time()
    with open(args.output, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(annotate_record, client, r, system, template,
                          args.passes, args.retries, args.model): r for r in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            done[rec["id"]] = rec
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  ({(time.time() - t0) / n:.1f}s por ejemplo)")

    if client.length_cut:
        print(f"\nAVISO: {client.length_cut} respuestas se cortaron por --max-tokens "
              f"({args.max_tokens}); sube el valor si hay errores de JSON.")
    if client.n_calls:
        print(f"Tokens: {client.prompt_tokens} de entrada + {client.completion_tokens} de salida "
              f"en {client.n_calls} llamadas | modo de formato final: {client.mode}")

    final = [done[r["id"]] for r in records if r["id"] in done]
    n_dis = sum(not r["llm_agreement"] for r in final)
    n_ctx = sum(r["llm_needs_wider_context"] for r in final)
    n_err = sum(bool(r["llm_errors"]) for r in final)
    print(f"desacuerdos/incompletos: {n_dis} | necesitan más contexto: {n_ctx} | con errores: {n_err}")

    final.sort(key=lambda r: (r["llm_agreement"] and not r["llm_needs_wider_context"]))
    tasks = [to_label_studio(r, f"nvidia:{args.model}", args.passes) for r in final]
    with open(args.ls_output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=1)
    print(f"Label Studio: {args.ls_output} ({len(tasks)} tareas)")

    dist = {}
    for r in final:
        if r["llm_passes"]:
            lab = r["llm_passes"][0]["primary"]
            dist[lab] = dist.get(lab, 0) + 1
    for lab, c in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {lab:35s} {c}")


if __name__ == "__main__":
    main()
