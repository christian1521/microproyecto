#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import os
from collections import Counter, defaultdict

LABELS = [
    "Background", "Gap", "Basis", "Comparison", "Application",
    "Improvement / Modification", "Evidence",
    "Identification of the Originator", "Further Reading",
]
NOISE_FLAG = "Not a real citation (noise / parsing error)"


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[l] * cb[l] for l in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def krippendorff_alpha_nominal(units):
    o = defaultdict(float)
    for vals in units:
        m = len(vals)
        if m < 2:
            continue
        for i, j in itertools.permutations(range(m), 2):
            o[(vals[i], vals[j])] += 1.0 / (m - 1)
    n_c = defaultdict(float)
    for (c, _), v in o.items():
        n_c[c] += v
    n = sum(n_c.values())
    if n <= 1:
        return float("nan")
    d_o = sum(v for (c, k), v in o.items() if c != k)
    d_e = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k)
    return 1.0 if d_e == 0 else 1 - (n - 1) * d_o / d_e


def prf(gold, pred, labels):
    out = {}
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        out[l] = {"precision": pr, "recall": rc, "f1": f1, "support": tp + fn}
    return out


def confusion(gold, pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        m[idx[g]][idx[p]] += 1
    return m


def read_choices(result, name):
    for r in result:
        if r.get("from_name") == name and r.get("type") == "choices":
            return r["value"].get("choices", [])
    return []


def load_human(path):
    tasks = json.load(open(path, encoding="utf-8"))
    human = {}
    for t in tasks:
        tid = t["data"].get("id")
        anns = []
        for a in t.get("annotations", []):
            if a.get("was_cancelled"):
                continue
            res = a.get("result", [])
            prim = read_choices(res, "primary")
            if not prim:
                continue
            anns.append({
                "annotator": str(a.get("completed_by", "unknown")),
                "primary": prim[0],
                "secondary": read_choices(res, "secondary"),
                "flags": read_choices(res, "flags"),
            })
        if tid and anns:
            human[tid] = anns
    return human


def build_gold(human, adjudicator):
    gold, ties, noise = {}, 0, 0
    for tid, anns in human.items():
        if any(NOISE_FLAG in a["flags"] for a in anns):
            noise += 1
            continue
        adj = [a for a in anns if adjudicator and a["annotator"] == adjudicator]
        if adj:
            gold[tid] = adj[0]["primary"]
            continue
        c = Counter(a["primary"] for a in anns).most_common()
        if len(c) > 1 and c[0][1] == c[1][1]:
            ties += 1
            continue
        gold[tid] = c[0][0]
    return gold, ties, noise


def human_agreement(human):
    by_ann = defaultdict(dict)
    for tid, anns in human.items():
        for a in anns:
            by_ann[a["annotator"]][tid] = a["primary"]
    pairs = {}
    for a, b in itertools.combinations(sorted(by_ann), 2):
        common = sorted(set(by_ann[a]) & set(by_ann[b]))
        if len(common) >= 5:
            pairs[f"{a} vs {b}"] = {
                "n": len(common),
                "agreement": sum(by_ann[a][t] == by_ann[b][t] for t in common) / len(common),
                "kappa": cohen_kappa([by_ann[a][t] for t in common],
                                     [by_ann[b][t] for t in common]),
            }
    units = [[a["primary"] for a in anns] for anns in human.values() if len(anns) >= 2]
    return pairs, krippendorff_alpha_nominal(units), len(units)


def evaluate(ids, gold, llm, human, label):
    ids = [t for t in ids if t in gold and t in llm and llm[t]["passes"]]
    if not ids:
        print(f"\n[{label}] sin ejemplos en común.")
        return None
    g = [gold[t] for t in ids]
    p = [llm[t]["passes"][0]["primary"] for t in ids]
    acc = sum(x == y for x, y in zip(g, p)) / len(ids)
    kappa = cohen_kappa(g, p)
    labels = [l for l in LABELS if l in set(g) | set(p)]
    per = prf(g, p, labels)
    macro = sum(per[l]["f1"] for l in labels) / len(labels)

    lenient = 0
    for t in ids:
        sec_h = {s for a in human[t] for s in a["secondary"]}
        sec_l = set(llm[t]["passes"][0]["secondary"])
        lenient += (llm[t]["passes"][0]["primary"] == gold[t]
                    or llm[t]["passes"][0]["primary"] in sec_h
                    or gold[t] in sec_l)

    print(f"\n===== LLM vs humanos [{label}]  n={len(ids)} =====")
    print(f"accuracy {acc:.3f} | kappa {kappa:.3f} | macro-F1 {macro:.3f} | "
          f"accuracy permisiva (acepta secundarias) {lenient / len(ids):.3f}")
    print(f"{'clase':35s} {'P':>6s} {'R':>6s} {'F1':>6s} {'n':>5s}")
    for l in labels:
        s = per[l]
        print(f"{l:35s} {s['precision']:6.2f} {s['recall']:6.2f} {s['f1']:6.2f} {s['support']:5d}")

    ag = [t for t in ids if llm[t]["agreement"]]
    dis = [t for t in ids if not llm[t]["agreement"]]
    acc_of = lambda S: (sum(llm[t]["passes"][0]["primary"] == gold[t] for t in S) / len(S)
                        if S else float("nan"))
    print(f"\nPasadas coinciden: n={len(ag)}, accuracy {acc_of(ag):.3f} | "
          f"pasadas discrepan: n={len(dis)}, accuracy {acc_of(dis):.3f}")

    cm = confusion(g, p, labels)
    print("\nMatriz de confusión (filas = humano, columnas = LLM):")
    short = [l[:6] for l in labels]
    print(" " * 20 + " ".join(f"{s:>7s}" for s in short))
    for l, row in zip(labels, cm):
        print(f"{l[:19]:20s}" + " ".join(f"{v:7d}" for v in row))
    return {"n": len(ids), "accuracy": acc, "kappa": kappa, "macro_f1": macro,
            "lenient_accuracy": lenient / len(ids), "per_class": per,
            "labels": labels, "confusion": cm, "ids": ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="JSON exportado de Label Studio")
    ap.add_argument("--preannotated", required=True, help="preannotated.jsonl")
    ap.add_argument("--blind-ids", help="archivo con ids (uno por línea) anotados sin ver el LLM")
    ap.add_argument("--adjudicator", help="id (completed_by) del adjudicador, si existe")
    ap.add_argument("--out-dir", default="eval_out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    human = load_human(args.export)
    llm = {}
    for line in open(args.preannotated, encoding="utf-8"):
        r = json.loads(line)
        llm[r["id"]] = {"passes": r.get("llm_passes", []),
                        "agreement": r.get("llm_agreement", False),
                        "sentence": r.get("citing_sentence"), "section": r.get("section")}
    gold, ties, noise = build_gold(human, args.adjudicator)
    print(f"Ejemplos con anotación humana: {len(human)} | con referencia: {len(gold)} "
          f"| empates excluidos: {ties} | ruido excluido: {noise}")

    pairs, alpha, n_units = human_agreement(human)
    print("\n===== Acuerdo entre humanos =====")
    if pairs:
        for k, v in pairs.items():
            print(f"{k}: n={v['n']} acuerdo={v['agreement']:.3f} kappa={v['kappa']:.3f}")
    print(f"Alpha de Krippendorff (nominal): {alpha:.3f}  (ejemplos con >=2 anotadores: {n_units})")

    report = {"human_pairs": pairs, "krippendorff_alpha": alpha}
    rep_all = evaluate(list(gold), gold, llm, human, "todos los ejemplos")
    report["llm_all"] = {k: v for k, v in (rep_all or {}).items() if k != "ids"}

    if args.blind_ids:
        blind = {l.strip() for l in open(args.blind_ids, encoding="utf-8") if l.strip()}
        rep_b = evaluate([t for t in gold if t in blind], gold, llm, human,
                         "solo ejemplos anotados SIN ver al LLM")
        report["llm_blind"] = {k: v for k, v in (rep_b or {}).items() if k != "ids"}
        if rep_all and rep_b:
            print(f"\nDiferencia de accuracy (todos - ciegos): "
                  f"{rep_all['accuracy'] - rep_b['accuracy']:+.3f}  "
                  f"(si es grande y positiva, sospecha de anclaje)")

    path = os.path.join(args.out_dir, "disagreements.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "section", "citing_sentence", "gold", "llm_primary",
                    "llm_secondary", "human_labels", "llm_pass_agreement"])
        for t, gl in gold.items():
            if t in llm and llm[t]["passes"] and llm[t]["passes"][0]["primary"] != gl:
                p0 = llm[t]["passes"][0]
                w.writerow([t, llm[t]["section"], llm[t]["sentence"], gl, p0["primary"],
                            "; ".join(p0["secondary"]),
                            "; ".join(a["primary"] for a in human[t]),
                            llm[t]["agreement"]])
    with open(os.path.join(args.out_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\nGuardado: {path} y eval_report.json en {args.out_dir}/")


if __name__ == "__main__":
    main()
