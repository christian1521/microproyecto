#!/usr/bin/env python3
import argparse
import json
import random

DATA_KEYS = ["id", "section", "left_context", "citing_sentence", "right_context",
             "citing_paper", "cited_ref_id", "n_citations_in_sentence"]


def add_link_fields(data):
    pid = str(data.get("citing_paper") or "").strip()
    if not pid:
        data.setdefault("links_html", "(sin identificador de artículo)")
        return False
    abs_url = f"https://arxiv.org/abs/{pid}"
    pdf_url = f"https://arxiv.org/pdf/{pid}"
    data["abs_url"], data["pdf_url"] = abs_url, pdf_url
    data["links_html"] = (f'arXiv: <a href="{abs_url}" target="_blank" style="text-decoration: underline;">Resumen</a>'
                          f' <br/>arXiv: <a href="{pdf_url}" target="_blank" style="text-decoration: underline;">Documento PDF</a>')
    return True


def cmd_from_contexts(a):
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
    if a.sample and a.sample < len(rows):
        random.Random(a.seed).shuffle(rows)
        rows = rows[:a.sample]
    tasks = [{"data": {k: r.get(k) for k in DATA_KEYS}} for r in rows]
    if not a.no_links:
        n_ok = sum(add_link_fields(t["data"]) for t in tasks)
        print(f"enlaces al artículo original: {n_ok}/{len(tasks)} tareas")
    json.dump(tasks, open(a.output, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(tasks)} tareas -> {a.output}")


def cmd_split_blind(a):
    tasks = json.load(open(a.input, encoding="utf-8"))
    if not a.no_links:
        for t in tasks:
            add_link_fields(t["data"])
    idx = list(range(len(tasks)))
    random.Random(a.seed).shuffle(idx)
    blind_idx = set(idx[:int(round(len(tasks) * a.frac))])
    with_pred, blind, ids = [], [], []
    for i, t in enumerate(tasks):
        if i in blind_idx:
            t = {k: v for k, v in t.items() if k != "predictions"}
            blind.append(t)
            ids.append(t["data"]["id"])
        else:
            with_pred.append(t)
    json.dump(with_pred, open(a.out_with, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(blind, open(a.out_blind, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(a.ids, "w").write("\n".join(ids) + "\n")
    print(f"con predicción: {len(with_pred)} -> {a.out_with} | "
          f"ciegas: {len(blind)} -> {a.out_blind} | ids -> {a.ids}")


def cmd_merge(a):
    merged, order = {}, []
    for spec in a.exports:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"Formato esperado nombre=archivo.json, recibí: {spec}")
        for t in json.load(open(path, encoding="utf-8")):
            tid = t["data"].get("id")
            if tid is None:
                continue
            if tid not in merged:
                merged[tid] = {k: v for k, v in t.items() if k != "annotations"}
                merged[tid]["annotations"] = []
                order.append(tid)
            for ann in t.get("annotations", []):
                if ann.get("was_cancelled"):
                    continue
                ann = dict(ann)
                ann["completed_by"] = name
                merged[tid]["annotations"].append(ann)
    out = [merged[t] for t in order]
    json.dump(out, open(a.output, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    multi = sum(len(t["annotations"]) >= 2 for t in out)
    print(f"{len(out)} tareas | con >=2 anotaciones: {multi} -> {a.output}")


def cmd_add_links(a):
    tasks = json.load(open(a.input, encoding="utf-8"))
    n_ok = sum(add_link_fields(t["data"]) for t in tasks)
    json.dump(tasks, open(a.output, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{n_ok} tareas con enlaces, {len(tasks) - n_ok} sin identificador -> {a.output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("from-contexts")
    p.add_argument("input")
    p.add_argument("-o", "--output", default="tasks.json")
    p.add_argument("--sample", type=int, default=0, help="N ejemplos al azar (0 = todos)")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--no-links", action="store_true",
                   help="no añadir los enlaces al artículo original")
    p.set_defaults(fn=cmd_from_contexts)

    p = sub.add_parser("split-blind")
    p.add_argument("input")
    p.add_argument("--frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out-with", default="tasks_with_pred.json")
    p.add_argument("--out-blind", default="tasks_blind.json")
    p.add_argument("--ids", default="blind_ids.txt")
    p.add_argument("--no-links", action="store_true",
                   help="no añadir los enlaces al artículo original")
    p.set_defaults(fn=cmd_split_blind)

    p = sub.add_parser("merge")
    p.add_argument("exports", nargs="+", help="nombre=archivo.json")
    p.add_argument("-o", "--output", default="merged_export.json")
    p.set_defaults(fn=cmd_merge)

    p = sub.add_parser("add-links")
    p.add_argument("input")
    p.add_argument("-o", "--output", default="tasks_links.json")
    p.set_defaults(fn=cmd_add_links)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
