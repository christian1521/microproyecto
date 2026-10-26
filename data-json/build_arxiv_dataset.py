#!/usr/bin/env python3
import argparse
import calendar
import gzip
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ATOM = "{http://www.w3.org/2005/Atom}"
ARX = "{http://arxiv.org/schemas/atom}"
OPEN, CLOSE = "\ue000", "\ue001"

class FetchError(Exception):
    pass


class TooLarge(Exception):
    pass


class Client:

    def __init__(self, contact, delay=3.0):
        self.ua = f"arxiv-cs-citation-dataset/0.1 (academic research; mailto:{contact})"
        self.delay = max(3.0, float(delay))
        self.last = 0.0

    def get(self, url, max_bytes=None, retries=4):
        for attempt in range(retries + 1):
            wait = self.last + self.delay - time.time()
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": self.ua})
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = r.read(max_bytes + 1) if max_bytes else r.read()
                self.last = time.time()
                if max_bytes and len(data) > max_bytes:
                    raise TooLarge(f">{max_bytes} bytes")
                return data
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504):
                    ra = e.headers.get("Retry-After") if e.headers else None
                    pause = float(ra) if ra and ra.isdigit() else 15 * (2 ** attempt)
                    print(f"  HTTP {e.code}; espero {min(pause, 300):.0f}s", file=sys.stderr)
                    time.sleep(min(pause, 300))
                    continue
                raise FetchError(f"HTTP {e.code}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                time.sleep(5 * (attempt + 1))
                last = e
        raise FetchError(f"sin respuesta tras {retries + 1} intentos")


def parse_atom(data):
    root = ET.fromstring(data)
    out = []
    for e in root.findall(ATOM + "entry"):
        url = (e.findtext(ATOM + "id") or "").strip()
        m = re.search(r"/abs/(.+?)(?:v\d+)?$", url)
        if not m:
            continue
        cats = [c.get("term") for c in e.findall(ATOM + "category") if c.get("term")]
        prim = e.find(ARX + "primary_category")
        out.append({
            "id": m.group(1),
            "title": re.sub(r"\s+", " ", e.findtext(ATOM + "title") or "").strip(),
            "abstract": re.sub(r"\s+", " ", e.findtext(ATOM + "summary") or "").strip(),
            "published": (e.findtext(ATOM + "published") or "")[:10],
            "categories": cats,
            "primary_category": prim.get("term") if prim is not None else (cats[0] if cats else ""),
            "authors": [a.findtext(ATOM + "name") for a in e.findall(ATOM + "author")],
        })
    return out


def month_windows(year):
    for m in range(1, 13):
        last = calendar.monthrange(year, m)[1]
        yield f"{year}{m:02d}010000", f"{year}{m:02d}{last:02d}2359"


def list_cell(client, cat, year, per_cell, rng):
    per_month = max(1, math.ceil(per_cell / 12))
    fetch_n = min(100, per_month * 5)
    pool = []
    for a, b in month_windows(year):
        q = f"cat:{cat} AND submittedDate:[{a} TO {b}]"
        params = {"search_query": q, "start": 0, "max_results": fetch_n,
                  "sortBy": "submittedDate", "sortOrder": "descending"}
        data = client.get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params))
        entries = parse_atom(data)
        rng.shuffle(entries)
        pool.extend(entries[:per_month])
    return pool


def entries_from_ids(client, ids):
    out = []
    for i in range(0, len(ids), 50):
        batch = ids[i:i + 50]
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(batch), "max_results": len(batch)})
        out.extend(parse_atom(client.get(url)))
    return out


def _decode(b):
    for enc in ("utf-8", "latin-1"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "ignore")


def read_archive(data, max_unpacked=60 * 1024 * 1024):
    files, total = {}, 0
    bio = io.BytesIO(data)
    if tarfile.is_tarfile(bio):
        bio.seek(0)
        with tarfile.open(fileobj=bio, mode="r:*") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                name = m.name.lstrip("./")
                if not name.lower().endswith((".tex", ".bbl", ".bib")):
                    continue
                total += m.size
                if total > max_unpacked or m.size > 20 * 1024 * 1024:
                    return None
                f = tar.extractfile(m)
                if f:
                    files[name] = _decode(f.read())
        return files or None
    try:
        raw = gzip.decompress(data)
    except OSError:
        return None
    text = _decode(raw)
    if "\\begin{document}" in text or "\\documentclass" in text:
        return {"main.tex": text}
    return None


def read_braced(s, i):
    """s[i] == '{'. Devuelve (contenido, índice tras la llave de cierre)."""
    depth, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def remove_command_with_arg(s, names):
    pat = re.compile(r"\\(?:%s)\*?\s*(?:\[[^\]]*\]\s*)*\{" % "|".join(names))
    pos = 0
    while True:
        m = pat.search(s, pos)
        if not m:
            return s
        _, end = read_braced(s, m.end() - 1)
        s = s[:m.start()] + s[end:]
        pos = m.start()


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*$", "", s, flags=re.M)


def find_file(name, files):
    name = name.strip().lstrip("./")
    for cand in (name, name + ".tex"):
        if cand in files:
            return cand
    base = os.path.basename(name)
    for cand in (base, base + ".tex"):
        for k in files:
            if k == cand or k.endswith("/" + cand):
                return k
    return None


def inline_inputs(text, files, depth=0):
    if depth > 5:
        return text

    def repl(m):
        k = find_file(m.group(1), files)
        if not k:
            return ""
        return inline_inputs(strip_comments(files[k]), files, depth + 1)

    return re.sub(r"\\(?:input|include|subfile)\s*\{([^}]+)\}", repl, text)


def pick_main(files):
    cands = [k for k, v in files.items()
             if k.lower().endswith(".tex") and "\\begin{document}" in v]
    if not cands:
        return None
    return max(cands, key=lambda k: len(files[k]))


def mk(kind, payload):
    return f"{OPEN}{kind}:{payload}{CLOSE}"


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


DROP_ENVS = ["figure", "figure\\*", "table", "table\\*", "tabular", "tabular\\*",
             "tabularx", "longtable", "algorithm", "algorithm\\*", "algorithmic",
             "lstlisting", "verbatim", "tikzpicture", "wrapfigure", "sidewaystable",
             "sidewaysfigure", "minted", "comment", "abstract", "IEEEkeywords",
             "filecontents", "filecontents\\*", "subfigure", "minipage"]
MATH_ENVS = ["equation", "equation\\*", "align", "align\\*", "eqnarray", "eqnarray\\*",
             "gather", "gather\\*", "multline", "multline\\*", "displaymath", "math",
             "flalign", "flalign\\*"]
META_CMDS = ["title", "author", "date", "affiliation", "address", "email", "thanks",
             "keywords", "institute", "icmlauthor", "icmlaffiliation", "icmlcorrespondingauthor",
             "footnote", "footnotetext", "footnotemark", "maketitle", "caption",
             "captionof", "tableofcontents"]

CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|citealp|citealt|citeauthor|citeyearpar|parencite|textcite|"
    r"autocite|footcite|Citep|Citet|Cite|citenum)\*?\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}")
SEC_RE = re.compile(
    r"\\(chapter|section|subsection|subsubsection|paragraph|subparagraph)(\*?)\s*"
    r"(?:\[[^\]]*\]\s*)?\{")


def replace_math(s):
    s = s.replace("\\$", "\ue002")
    for env in MATH_ENVS:
        s = re.sub(r"\\begin\{(%s)\}.*?\\end\{\1\}" % env,
                   lambda m: mk("formula", md5(m.group(0))), s, flags=re.S)
    s = re.sub(r"\$\$.*?\$\$", lambda m: mk("formula", md5(m.group(0))), s, flags=re.S)
    s = re.sub(r"\\\[.*?\\\]", lambda m: mk("formula", md5(m.group(0))), s, flags=re.S)
    s = re.sub(r"\\\(.*?\\\)", lambda m: mk("formula", md5(m.group(0))), s, flags=re.S)
    s = re.sub(r"\$([^$]{1,500}?)\$", lambda m: mk("formula", md5(m.group(0))), s)
    return s.replace("\ue002", "$")


def replace_cites(s, pid, used_keys):
    def repl(m):
        keys = [k.strip() for k in m.group(1).split(",") if k.strip()]
        for k in keys:
            used_keys.add(k)
        return ", ".join(mk("cite", md5(f"{pid}|{k}")) for k in keys)

    s = re.sub(r"\\citeyear\*?\s*(?:\[[^\]]*\]\s*)?\{[^}]*\}", "", s)
    return CITE_RE.sub(repl, s)


ACCENT_RE = re.compile(r"""\\["'`^~=.]\s*\{?([A-Za-z])\}?""")


def clean_inline(s):
    s = re.sub(r"\\\\(?:\[[^\]]*\])?", " ", s)
    s = re.sub(r"\\(?:eqref|autoref|cref|Cref|ref|pageref|nameref|vref)\*?\{[^}]*\}", "[REF]", s)
    s = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\url\{[^}]*\}", "[URL]", s)
    s = re.sub(r"\\textcolor\{[^}]*\}\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:label|index|vspace\*?|hspace\*?|includegraphics|bibliographystyle|"
               r"bibliography|pagestyle|thispagestyle|setlength|setcounter|usepackage|"
               r"newblock|bibitem)\s*(?:\[[^\]]*\])?(?:\{[^{}]*\})?", "", s)
    s = re.sub(r"\\begin\{[^}]*\}(?:\[[^\]]*\])?|\\end\{[^}]*\}", " ", s)
    for _ in range(5):  # \cmd{arg} -> arg (de dentro hacia fuera)
        new = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", s)
        if new == s:
            break
        s = new
    s = re.sub(r"\{\\(?:em|it|bf|sc|sl|tt|rm|sf)\s+([^{}]*)\}", r"\1", s)
    s = ACCENT_RE.sub(r"\1", s)
    for a, b in (("\\&", "&"), ("\\%", "%"), ("\\_", "_"), ("\\#", "#"),
                 ("\\{", "{"), ("\\}", "}"), ("\\ldots", "..."), ("\\dots", "...")):
        s = s.replace(a, b)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("~", " ").replace("``", '"').replace("''", '"')
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"---|--", "-", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_bbl(text):
    out = {}
    pat = re.compile(
        r"\\bibitem\s*(?:\[[^\]]{0,600}\])?\s*\{([^}]*)\}(.*?)"
        r"(?=\\bibitem|\\end\{thebibliography\}|\Z)", re.S)
    for m in pat.finditer(text):
        raw = clean_inline(strip_comments(m.group(2)))
        out[m.group(1).strip()] = raw
    return out


def parse_bib(text):
    out = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)(?=\n\s*@|\Z)", text, re.S):
        if m.group(1).lower() in ("comment", "string", "preamble"):
            continue
        fields = {}
        for f in re.finditer(
                r'(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|"[^"]*"|\w+)', m.group(3)):
            val = f.group(2).strip()
            if val[:1] in "{\"":
                val = val[1:-1]
            fields[f.group(1).lower()] = clean_inline(val)
        parts = [fields.get(k, "") for k in ("author", "title", "journal", "booktitle", "year")]
        extra = " ".join(f"{k}:{fields[k]}" for k in ("eprint", "doi") if k in fields)
        out[m.group(2)] = (". ".join(p for p in parts if p) + " " + extra).strip()
    return out


def raw_ids(raw):
    ids = {}
    m = re.search(r"(?:arXiv:|arxiv\.org/abs/|eprint:)\s*(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})",
                  raw, re.I)
    if m:
        ids["arxiv_id"] = m.group(1)
    m = re.search(r"(10\.\d{4,9}/[^\s\"<>,;]+)", raw)
    if m:
        ids["doi"] = m.group(1).rstrip(".)")
    return ids


def split_sections(body):
    """Devuelve [(section, subsection, sec_number, texto_crudo)]."""
    segs, buf = [], []
    top, sub, num = "", "", ""
    counters = [0, 0, 0]
    pos = 0

    def flush():
        txt = "".join(buf)
        if txt.strip():
            segs.append((top, sub, num, txt))
        buf.clear()

    while True:
        m = SEC_RE.search(body, pos)
        if not m:
            buf.append(body[pos:])
            break
        buf.append(body[pos:m.start()])
        title, end = read_braced(body, m.end() - 1)
        kind, star = m.group(1), m.group(2)
        pos = end
        if kind in ("paragraph", "subparagraph"):
            buf.append("\n\n")
            continue
        flush()
        level = {"chapter": 1, "section": 1, "subsection": 2, "subsubsection": 3}[kind]
        t = clean_inline(title)
        if not star:
            counters[level - 1] += 1
            for i in range(level, 3):
                counters[i] = 0
            num = ".".join(str(c) for c in counters[:level])
        else:
            num = ""
        if level == 1:
            top, sub = t, ""
        else:
            sub = t
    flush()
    return segs


def looks_english(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 50:
        return True
    common = {"the", "of", "and", "in", "to", "is", "we", "a", "for", "that", "with"}
    return sum(w in common for w in words) / len(words) > 0.12


def finalize_text(t):
    return t.replace(OPEN, "{{").replace(CLOSE, "}}")


def process_paper(meta, files, english_filter=True):
    """Devuelve (registro, None) o (None, motivo)."""
    pid = meta["id"]
    main_key = pick_main(files)
    if not main_key:
        return None, "no_main_tex"
    raw_main = strip_comments(files[main_key])
    text = inline_inputs(raw_main, files)

    bib = {}
    for k, v in files.items():
        if k.lower().endswith(".bbl"):
            bib.update(parse_bbl(strip_comments(v)))
    for m in re.finditer(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", text, re.S):
        for k, v in parse_bbl(m.group(0)).items():
            bib.setdefault(k, v)
    for k, v in files.items():
        if k.lower().endswith(".bib"):
            for kk, vv in parse_bib(v).items():
                bib.setdefault(kk, vv)

    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", text, re.S)
    body = m.group(1) if m else text
    body = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", body, flags=re.S)
    body = re.sub(r"\\(?:bibliographystyle|bibliography|printbibliography)\*?\s*(?:\[[^\]]*\])?(?:\{[^}]*\})?", "", body)
    body = body.replace("\\appendix", "")
    for env in DROP_ENVS:
        body = re.sub(r"\\begin\{(%s)\}.*?\\end\{\1\}" % env, "\n\n", body, flags=re.S)
    body = remove_command_with_arg(body, META_CMDS)
    body = replace_math(body)
    used = set()
    body = replace_cites(body, pid, used)
    if not used:
        return None, "no_citations"

    body_text, n_chars = [], 0
    for section, subsection, sec_number, raw in split_sections(body):
        raw = re.sub(r"\\item(?:\[[^\]]*\])?", "\n\n", raw)
        raw = raw.replace("\\par", "\n\n")
        for para in re.split(r"\n\s*\n", raw):
            t = clean_inline(para)
            plain = re.sub(OPEN + "[^" + CLOSE + "]*" + CLOSE, "", t)
            if len(plain) < 40 or sum(c.isalpha() for c in plain) / max(len(plain), 1) < 0.5:
                continue
            t = finalize_text(t)
            spans = [{"start": s.start(), "end": s.end(), "text": s.group(0),
                      "ref_id": s.group(1)}
                     for s in re.finditer(r"\{\{cite:([0-9a-f]+)\}\}", t)]
            body_text.append({"text": t, "cite_spans": spans, "section": section,
                              "subsection": subsection, "sec_number": sec_number})
            n_chars += len(t)
    if len(body_text) < 5:
        return None, "too_few_paragraphs"
    if english_filter and not looks_english(" ".join(p["text"] for p in body_text[:30])):
        return None, "not_english"

    bib_entries = {}
    for key in sorted(used):
        h = md5(f"{pid}|{key}")
        raw = bib.get(key, "")
        bib_entries[h] = {"ref_id": h, "bib_key": key, "bib_entry_raw": raw,
                          "ids": raw_ids(raw)}
    return {
        "paper_id": pid,
        "metadata": {"title": meta["title"], "authors": meta["authors"],
                     "categories": " ".join(meta["categories"]),
                     "primary_category": meta["primary_category"],
                     "published": meta["published"]},
        "abstract": meta["abstract"],
        "body_text": body_text,
        "bib_entries": bib_entries,
    }, None


def parse_years(spec):
    years = []
    for part in spec:
        if "-" in part:
            a, b = part.split("-")
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def safe(pid):
    return pid.replace("/", "_")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contact", required=True, help="tu correo (va en el User-Agent)")
    ap.add_argument("--categories", nargs="+", default=["cs.CL", "cs.LG", "cs.CV", "cs.SE", "cs.IR"])
    ap.add_argument("--years", nargs="+", default=["2021-2023"], help="p. ej. 2021-2023 o 2020 2022")
    ap.add_argument("--per-cell", type=int, default=20, help="artículos por (categoría, año)")
    ap.add_argument("--ids-file", help="en vez de muestrear: archivo con ids de arXiv, uno por línea")
    ap.add_argument("--output", default="data/arxiv_cs.jsonl")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--delete-sources", action="store_true", help="borra el fuente tras procesarlo")
    ap.add_argument("--max-source-mb", type=float, default=25.0)
    ap.add_argument("--delay", type=float, default=3.0, help="segundos entre peticiones (mínimo 3)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-english-filter", action="store_true")
    ap.add_argument("--list-only", action="store_true", help="solo lista candidatos (no descarga fuentes)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    client = Client(args.contact, args.delay)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    src_dir = os.path.join(args.cache_dir, "src")
    os.makedirs(src_dir, exist_ok=True)

    done_path, fail_path = args.output + ".done", args.output + ".failed.tsv"
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()

    if args.ids_file:
        ids = [l.strip() for l in open(args.ids_file) if l.strip()]
        candidates = entries_from_ids(client, ids)
    else:
        candidates, seen = [], set()
        for cat in args.categories:
            for year in parse_years(args.years):
                print(f"Listando {cat} {year} ...")
                pool = [e for e in list_cell(client, cat, year, args.per_cell, rng)
                        if e["id"] not in seen]
                rng.shuffle(pool)
                for e in pool[:args.per_cell]:
                    seen.add(e["id"])
                    candidates.append(e)
    todo = [e for e in candidates if e["id"] not in done]
    print(f"{len(candidates)} candidatos | {len(done)} ya procesados | {len(todo)} pendientes "
          f"| tiempo mínimo estimado de descarga: {len(todo) * args.delay / 60:.0f} min")
    if args.list_only:
        for e in todo:
            print(e["id"], e["primary_category"], e["published"], e["title"][:70])
        return

    stats = {}
    with open(args.output, "a", encoding="utf-8") as out, \
            open(done_path, "a") as dn, open(fail_path, "a") as fl:
        for n, meta in enumerate(todo, 1):
            pid = meta["id"]
            path = os.path.join(src_dir, safe(pid) + ".bin")
            reason = None
            try:
                if os.path.exists(path):
                    data = open(path, "rb").read()
                else:
                    data = client.get(f"https://arxiv.org/e-print/{pid}",
                                      max_bytes=int(args.max_source_mb * 1024 * 1024))
                    with open(path, "wb") as f:
                        f.write(data)
                files = read_archive(data)
                if not files:
                    reason = "no_latex_source"
                else:
                    rec, reason = process_paper(meta, files, not args.no_english_filter)
                    if rec:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out.flush()
                        reason = "ok"
            except TooLarge:
                reason = "source_too_large"
            except FetchError as e:
                reason = f"fetch_error:{e}"
            except Exception as e:  # un artículo roto no debe detener el proceso
                reason = f"parse_error:{type(e).__name__}"
            stats[reason] = stats.get(reason, 0) + 1
            if reason != "ok":
                fl.write(f"{pid}\t{reason}\n")
                fl.flush()
            if not reason.startswith("fetch_error"):
                dn.write(pid + "\n")
                dn.flush()
            if args.delete_sources and os.path.exists(path):
                os.remove(path)
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  {stats}")
    print("Resumen:", stats)
    print(f"Salida: {args.output}")


if __name__ == "__main__":
    main()
