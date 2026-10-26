#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

LABELS = [
    "Application",
    "Improvement / Modification",
    "Further Reading",
    "Identification of the Originator",
    "Comparison",
    "Gap",
    "Evidence",
    "Basis",
    "Background",
]
BAR_WIDTH = 30


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"AVISO: línea {n} de {path} no es JSON válido; se omite.", file=sys.stderr)
    return rows


def pick_labels(rec, pass_sel):
    passes = rec.get("llm_passes") or []
    if pass_sel == "all":
        return [p["primary"] for p in passes if p.get("primary")]
    k = int(pass_sel) - 1
    if 0 <= k < len(passes) and passes[k].get("primary"):
        return [passes[k]["primary"]]
    return []


def ordered(counter):
    known = [l for l in LABELS if l in counter]
    extra = sorted(l for l in counter if l not in LABELS)
    return known + extra


def pct(n, total):
    return 100.0 * n / total if total else 0.0


def bar(p):
    return "#" * int(round(p / 100 * BAR_WIDTH))


def norm_section(s):
    s = re.sub(r"^(?:\d+(?:\.\d+)*\.?|[IVXivx]+\.|[IVX]+(?=\s))\s*", "", str(s or "").strip()).strip().lower()
    return s or "(sin seccion)"


def summarize(rows, pass_sel):
    prim = Counter()
    n_labeled = 0
    for r in rows:
        labs = pick_labels(r, pass_sel)
        if labs:
            n_labeled += 1
            prim.update(labs)
    return prim, n_labeled


def print_table(title, counter, base, extra_cols=None):
    print(f"\n== {title} ==")
    print(f"{'Función':36s} {'n':>6s} {'%':>7s}  barra")
    for l in ordered(counter):
        p = pct(counter[l], base)
        print(f"{l:36s} {counter[l]:6d} {p:6.1f}%  {bar(p)}")
    print(f"{'TOTAL':36s} {sum(counter.values()):6d}")


def report_single(path, rows, pass_sel, by_section, csv_rows):
    n = len(rows)
    models = Counter(r.get("llm_model") for r in rows if r.get("llm_model"))
    prim, n_labeled = summarize(rows, pass_sel)
    n_err = n - n_labeled
    print(f"\nArchivo: {path}")
    print(f"Ejemplos: {n} | con etiqueta: {n_labeled} | sin etiqueta (errores): {n_err}"
          + (f" | modelo: {', '.join(models)}" if models else ""))
    if not n_labeled:
        print("No hay etiquetas que resumir.")
        return

    label = "todas las pasadas juntas" if pass_sel == "all" else f"pasada {pass_sel}"
    base = sum(prim.values())
    print_table(f"Función principal asignada ({label}); base = {base}", prim, base)
    for l in ordered(prim):
        csv_rows.append([os.path.basename(path), "principal", l, prim[l], round(pct(prim[l], base), 2)])

    n_passes = max((len(r.get("llm_passes") or []) for r in rows), default=0)
    if n_passes >= 2:
        agree = [r for r in rows if r.get("llm_agreement") and r.get("llm_primary")]
        cons = Counter(r["llm_primary"] for r in agree)
        print(f"\n== Consenso entre pasadas ({n_passes} pasadas) ==")
        print(f"Con la misma etiqueta en todas las pasadas: {len(agree)} de {n_labeled} "
              f"({pct(len(agree), n_labeled):.1f}%) | sin consenso: {n_labeled - len(agree)} "
              f"({pct(n_labeled - len(agree), n_labeled):.1f}%)")
        print_table("Reparto de los ejemplos CON consenso", cons, len(agree))
        for l in ordered(cons):
            csv_rows.append([os.path.basename(path), "consenso", l, cons[l], round(pct(cons[l], len(agree)), 2)])

        tot, ok = Counter(), Counter()
        for r in rows:
            first = pick_labels(r, "1")
            if first:
                tot[first[0]] += 1
                ok[first[0]] += bool(r.get("llm_agreement"))
        print("\n== Acuerdo entre pasadas por función (etiqueta de la pasada 1) ==")
        print(f"{'Función':36s} {'n':>6s} {'acuerdo':>9s}")
        for l in ordered(tot):
            print(f"{l:36s} {tot[l]:6d} {pct(ok[l], tot[l]):8.1f}%")
    else:
        print("\n(Solo hay una pasada por ejemplo: no se calcula consenso.)")

    sec = Counter()
    with_sec = 0
    for r in rows:
        passes = r.get("llm_passes") or []
        if passes and passes[0].get("secondary"):
            with_sec += 1
            sec.update(passes[0]["secondary"])
    print("\n== Funciones secundarias (pasada 1) ==")
    print(f"Ejemplos con alguna secundaria: {with_sec} de {n_labeled} ({pct(with_sec, n_labeled):.1f}%)")
    for l in ordered(sec):
        print(f"  {l:36s} {sec[l]:5d}")
        csv_rows.append([os.path.basename(path), "secundaria", l, sec[l], round(pct(sec[l], n_labeled), 2)])
    n_ctx = sum(bool(r.get("llm_needs_wider_context")) for r in rows)
    print(f"\nMarcados 'necesita más contexto': {n_ctx} ({pct(n_ctx, n):.1f}% del total)")

    if by_section:
        by = defaultdict(Counter)
        for r in rows:
            for lab in pick_labels(r, pass_sel):
                by[norm_section(r.get("section"))][lab] += 1
        print(f"\n== Funciones más frecuentes por sección ({label}) ==")
        top = sorted(by.items(), key=lambda kv: -sum(kv[1].values()))[:10]
        for sec_name, c in top:
            t = sum(c.values())
            parts = ", ".join(f"{l} {pct(v, t):.0f}%" for l, v in c.most_common(4))
            print(f"  {sec_name[:28]:28s} (n={t}): {parts}")
            for l, v in c.items():
                csv_rows.append([os.path.basename(path), f"seccion:{sec_name}", l, v, round(pct(v, t), 2)])


def report_multi(paths, all_rows, pass_sel, csv_rows):
    summaries = []
    for p, rows in zip(paths, all_rows):
        prim, n_labeled = summarize(rows, pass_sel)
        agree = sum(bool(r.get("llm_agreement")) for r in rows)
        summaries.append((os.path.basename(p), prim, sum(prim.values()), len(rows), n_labeled, agree))
    labels = []
    for _, prim, *_ in summaries:
        labels += [l for l in prim if l not in labels]
    labels = [l for l in LABELS if l in labels] + sorted(l for l in labels if l not in LABELS)
    names = [s[0][:16] for s in summaries]
    label = "todas las pasadas" if pass_sel == "all" else f"pasada {pass_sel}"
    print(f"\n== Comparación de la función principal ({label}; % sobre ejemplos con etiqueta) ==")
    print(f"{'Función':34s}" + "".join(f"{n:>18s}" for n in names))
    for l in labels:
        cells = []
        for fname, prim, base, *_rest in summaries:
            cells.append(f"{prim.get(l, 0):5d} ({pct(prim.get(l, 0), base):5.1f}%)")
            csv_rows.append([fname, "principal", l, prim.get(l, 0), round(pct(prim.get(l, 0), base), 2)])
        print(f"{l:34s}" + "".join(f"{c:>18s}" for c in cells))
    print(f"{'Ejemplos totales':34s}" + "".join(f"{s[3]:>18d}" for s in summaries))
    print(f"{'Con etiqueta':34s}" + "".join(f"{s[4]:>18d}" for s in summaries))
    print(f"{'Pasadas en consenso (%)':34s}" + "".join(
        f"{pct(s[5], s[4]):>17.1f}%" for s in summaries))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="uno o varios archivos preannotated.jsonl")
    ap.add_argument("--pass", dest="pass_sel", default="1",
                    help="pasada a resumir: 1, 2, ... o 'all' (por defecto 1)")
    ap.add_argument("--by-section", action="store_true", help="desglose por sección (un solo archivo)")
    ap.add_argument("--csv", help="guarda las tablas en un CSV")
    a = ap.parse_args()
    if a.pass_sel != "all" and not a.pass_sel.isdigit():
        ap.error("--pass debe ser un número (1, 2, ...) o 'all'")

    all_rows = [load(p) for p in a.inputs]
    csv_rows = []
    if len(a.inputs) == 1:
        report_single(a.inputs[0], all_rows[0], a.pass_sel, a.by_section, csv_rows)
    else:
        report_multi(a.inputs, all_rows, a.pass_sel, csv_rows)
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["archivo", "tabla", "funcion", "n", "porcentaje"])
            w.writerows(csv_rows)
        print(f"\nCSV guardado en {a.csv}")


if __name__ == "__main__":
    main()
