#!/usr/bin/env python3
import argparse
import glob
import json
import random
import re
import sys

try:
    import pysbd
except ImportError:
    pysbd = None

CITE_RE = re.compile(r"\{\{cite:([0-9a-f]+)\}\}")
FORMULA_RE = re.compile(r"\{\{formula:[0-9a-f\-]+\}\}")

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z{\[(])")
PROT = "\ue010"

EG_IE_RE = re.compile(r"\b(?:e\.g|i\.e)\.")
ABBREV_RE = re.compile(
    r"\b(Eqs?|Figs?|Secs?|Tabs?|Refs?|Alg|Thm|Lem|Prop|Cor|Def|Ch|Vol|Dr|Prof|"
    r"eq|fig|cf|vs|pp|resp|approx|viz)\.(?=\s)")
ET_AL_RE = re.compile(r"\bet al\.(?=\s*(?:[a-z0-9,;:)\[(]|\{\{))")


def protect_abbreviations(text):
    text = EG_IE_RE.sub(lambda m: m.group(0).replace(".", PROT), text)
    text = ABBREV_RE.sub(lambda m: m.group(1) + PROT, text)
    return ET_AL_RE.sub("et al" + PROT, text)


class SentenceSplitter:
    def __init__(self, kind="auto"):
        if kind == "auto":
            kind = "pysbd" if pysbd is not None else "regex"
            if kind == "regex":
                print("AVISO: pysbd no está instalado; uso la regex simple. "
                      "Instálalo con: pip install pysbd", file=sys.stderr)
        if kind == "pysbd" and pysbd is None:
            sys.exit("Falta pysbd. Instálalo con: pip install pysbd")
        self.kind = kind
        self.seg = pysbd.Segmenter(language="en", clean=False) if kind == "pysbd" else None

    def __call__(self, text):
        prot = protect_abbreviations(text)
        sents = None
        if self.seg is not None:
            try:
                sents = self.seg.segment(prot)
            except Exception:
                sents = None
        if sents is None:
            sents = SENT_SPLIT_RE.split(prot)
        sents = [s.replace(PROT, ".").strip() for s in sents]
        return [s for s in sents if s]


def get_categories(paper):
    meta = paper.get("metadata") or {}
    cats = meta.get("categories") or paper.get("categories") or ""
    if isinstance(cats, list):
        cats = " ".join(cats)
    return cats.split()


def is_cs(paper):
    return any(c.startswith("cs.") for c in get_categories(paper))


def paper_year(paper):
    meta = paper.get("metadata") or {}
    m = re.match(r"(\d{4})", str(meta.get("published") or meta.get("year") or ""))
    return int(m.group(1)) if m else None


def normalize(text, target_ref):
    def repl(m):
        return "#CITATION_TAG" if m.group(1) == target_ref else "#OTHER_CITATION"
    text = CITE_RE.sub(repl, text)
    return FORMULA_RE.sub("[MATH]", text)


def process_paper(paper, window, max_per_paper, rng, split, min_chars, max_chars):
    pid = paper.get("paper_id")
    bib = paper.get("bib_entries") or {}
    records = []
    for p_idx, para in enumerate(paper.get("body_text") or []):
        text = para.get("text") or ""
        if not text or not CITE_RE.search(text):
            continue
        sents = split(text)
        for s_idx, sent in enumerate(sents):
            refs = CITE_RE.findall(sent)
            if not refs:
                continue
            citing = FORMULA_RE.sub("[MATH]", CITE_RE.sub("", sent))
            if not (min_chars <= len(citing) <= max_chars):
                continue 
            for ref in dict.fromkeys(refs):  # únicos, conserva orden
                lo, hi = max(0, s_idx - window), min(len(sents), s_idx + window + 1)
                bib_entry = bib.get(ref) or {}
                records.append({
                    "id": f"{pid}_{p_idx}_{s_idx}_{ref[:8]}",
                    "citing_paper": pid,
                    "cited_ref_id": ref,
                    "cited_ids": bib_entry.get("ids"),
                    "cited_raw": bib_entry.get("bib_entry_raw"),
                    "section": para.get("section"),
                    "sec_number": para.get("sec_number"),
                    "left_context": " ".join(normalize(x, ref) for x in sents[lo:s_idx]),
                    "citing_sentence": normalize(sent, ref),
                    "right_context": " ".join(normalize(x, ref) for x in sents[s_idx + 1:hi]),
                    "n_citations_in_sentence": len(set(refs)),
                    "label": None,
                })
    if len(records) > max_per_paper: 
        records = rng.sample(records, max_per_paper)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="glob de archivos .jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--max-per-paper", type=int, default=40)
    ap.add_argument("--min-year", type=int, default=0,
                    help="descarta artículos anteriores a este año (si el dato existe)")
    ap.add_argument("--splitter", choices=["auto", "pysbd", "regex"], default="auto")
    ap.add_argument("--min-sentence-chars", type=int, default=20,
                    help="descarta oraciones citantes más cortas (sin marcadores)")
    ap.add_argument("--max-sentence-chars", type=int, default=1200,
                    help="descarta oraciones citantes más largas (suelen ser errores de segmentación)")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    split = SentenceSplitter(args.splitter)
    print(f"Segmentador de oraciones: {split.kind}")
    n_papers = n_records = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for path in sorted(glob.glob(args.input)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    paper = json.loads(line)
                    if not is_cs(paper):
                        continue
                    year = paper_year(paper)
                    if args.min_year and year is not None and year < args.min_year:
                        continue
                    n_papers += 1
                    for rec in process_paper(paper, args.window, args.max_per_paper, rng, split,
                                             args.min_sentence_chars, args.max_sentence_chars):
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_records += 1
    print(f"papers cs.*: {n_papers} | contextos: {n_records}")


if __name__ == "__main__":
    main()
