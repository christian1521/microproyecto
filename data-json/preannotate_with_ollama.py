#!/usr/bin/env python3
import argparse
import json
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
    """Baraja y renumera el bloque de definiciones de etiquetas."""
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
    """Devuelve (dict_validado, None) o (None, mensaje_de_error)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # algunos modelos
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


class OllamaError(Exception):
    pass


class OllamaClient:
    def __init__(self, args):
        self.host = args.host.rstrip("/")
        self.args = args
        self.fmt = args.format
        self.lock = threading.Lock()
        self.max_prompt_tokens = 0
        self.truncated = 0

    def _post(self, path, payload, timeout):
        req = urllib.request.Request(
            self.host + path, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def check_server(self):
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=10) as r:
                tags = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            sys.exit(f"No pude conectar con Ollama en {self.host} ({e}).\n"
                     "¿Está arrancado? (ollama serve, o abre la app de Ollama).")
        names = [m.get("name", "") for m in tags.get("models", [])]
        want = self.args.model
        if not any(n == want or n == want + ":latest" or n.split(":")[0] == want for n in names):
            sys.exit(f"El modelo '{want}' no está descargado.\n"
                     f"Ejecuta: ollama pull {want}\n"
                     f"Modelos disponibles: {', '.join(names) or '(ninguno)'}")

    def chat(self, system, user):
        a = self.args
        payload = {
            "model": a.model,
            "stream": False,
            "keep_alive": a.keep_alive,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"temperature": a.temperature, "seed": a.seed,
                        "num_ctx": a.num_ctx, "num_predict": a.num_predict},
        }
        if a.no_think:
            payload["think"] = False
        if self.fmt == "schema":
            payload["format"] = SCHEMA
        elif self.fmt == "json":
            payload["format"] = "json"
        try:
            resp = self._post("/api/chat", payload, a.timeout)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            if e.code == 400 and self.fmt == "schema":
                print("AVISO: este Ollama rechazó el esquema; paso a format=json.\n  "
                      + body[:200], file=sys.stderr)
                self.fmt = "json"
                return self.chat(system, user)
            raise OllamaError(f"HTTP {e.code}: {body[:300]}")
        n = resp.get("prompt_eval_count") or 0
        with self.lock:
            self.max_prompt_tokens = max(self.max_prompt_tokens, n)
            if n and n >= a.num_ctx - 8:
                self.truncated += 1
        return (resp.get("message") or {}).get("content", "")


def make_caller(args):
    if args.dry_run:
        rng = random.Random(0)

        class Fake:
            max_prompt_tokens, truncated = 0, 0

            def check_server(self):
                pass

            def chat(self, system, user):
                return json.dumps({"rationale": "dry run", "primary": rng.choice(LABELS),
                                   "secondary": [], "needs_wider_context": False})
        return Fake()
    client = OllamaClient(args)
    client.check_server()
    return client


def classify(client, system, user, retries):
    last_err = "unknown"
    for attempt in range(retries + 1):
        try:
            parsed, err = parse_response(client.chat(system, user))
            if parsed:
                return parsed, None
            last_err = err
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
        "llm_model": f"ollama:{model}",
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
    ap.add_argument("--input", required=True)
    ap.add_argument("--prompt", default="preannotation_prompt.txt")
    ap.add_argument("--output", default="preannotated.jsonl")
    ap.add_argument("--ls-output", default="label_studio_tasks.json")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1,
                    help="peticiones simultáneas; con una sola GPU suele convenir 1-2")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1, help="semilla de generación de Ollama")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=300)
    ap.add_argument("--keep-alive", default="30m")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--format", choices=["schema", "json", "none"], default="schema")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="procesar solo N ejemplos (0 = todos)")
    ap.add_argument("--sample-seed", type=int, default=13)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

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
    print(f"{len(records)} ejemplos | {len(done)} ya procesados | {len(todo)} pendientes")

    client = make_caller(args)
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
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  ({el / n:.1f}s por ejemplo)")

    if client.truncated:
        print(f"\n¡ATENCIÓN! {client.truncated} llamadas llenaron la ventana de contexto "
              f"(num_ctx={args.num_ctx}); el prompt pudo truncarse. Sube --num-ctx "
              f"(máx. de tokens de prompt vistos: {client.max_prompt_tokens}).")
    elif client.max_prompt_tokens:
        print(f"\nTokens de prompt máx.: {client.max_prompt_tokens} (num_ctx={args.num_ctx}) OK")

    final = [done[r["id"]] for r in records if r["id"] in done]
    n_dis = sum(not r["llm_agreement"] for r in final)
    n_ctx = sum(r["llm_needs_wider_context"] for r in final)
    n_err = sum(bool(r["llm_errors"]) for r in final)
    print(f"desacuerdos/incompletos: {n_dis} | necesitan más contexto: {n_ctx} | con errores: {n_err}")

    final.sort(key=lambda r: (r["llm_agreement"] and not r["llm_needs_wider_context"]))
    tasks = [to_label_studio(r, f"ollama:{args.model}", args.passes) for r in final]
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
