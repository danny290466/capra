#!/usr/bin/env python3
"""Consistency gate for the CAPRA manuscript of record (modules 1–7b of the `kiem-tra-nhat-quan` skill).

Run it after every write to the manuscript, and treat a non-zero exit as a build failure.

    python3 tools/audit_consistency.py                       # gate the live Drive file
    python3 tools/audit_consistency.py --docx other.docx      # gate another build
    python3 tools/audit_consistency.py --json out.json        # machine-readable findings
    python3 tools/audit_consistency.py --assertions FILE      # extra standing assertions

Two things make this different from a grep harness:

1. **Equations are read.** After the round-10 pass every formula is an `m:oMath` object with no `w:t`,
   so a python-docx text view shows empty notation. Every view here is built from `w:t` AND `m:t` in
   true element order (paragraphs and tables interleaved).
2. **The ledger is checked.** The last `after md5` in `MANUSCRIPT_CHANGELOG.md` must equal the live
   file's md5 — the one check that catches a write made outside the tooling, which is how this gate
   was born (a 12:16 external save left the file unattributable).

Findings carry a severity: `mismatch`, `broken`, `missing-def`, `variant`, `provenance` fail the gate;
`note` never does. Every accepted variant lives in ALLOWED below with its one-line reason, so a new
inconsistency cannot hide behind a growing list of exceptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from collections import Counter

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCX = os.environ.get("CAPRA_DOCX")   # the manuscript .docx is not part of this release
CHANGELOG = os.path.join(REPO, "MANUSCRIPT_CHANGELOG.md")

FAILING = {"mismatch", "broken", "missing-def", "variant", "provenance"}

# ── reviewed and accepted: token -> why it is not a finding ────────────────────────────────────────
ALLOWED_ABBREV = {
    "CAPRA": "defined in the abstract: capacity-aware parity risk allocation",
    "CONRA": "defined in §6.3 and named in §4.2",
    "RQ": "defined in §1 as 'three research questions (RQ1–RQ3)'",
    "PF": "defined in the Table 8 caption (parity-first endpoint, CAPRA-parity)",
    "CF": "defined in the Table 8 caption (composition-first, CAPRA-composition)",
    "GLOBAL": "defined in the Table 12 note (the pooled threshold)",
    "EQUAL-COV": "defined in the Table 12 note",
    "ABS-RISK": "defined in the Table 12 note (group-conditional risk control)",
    "ABS-RISK-FINITE": "defined in the Table 12 note (with a finite-sample slack)",
    "BONF-CRC": "defined in the Table 12 note (Bonferroni-style split of the risk target)",
    "MM-CAP": "defined in the Table 12 note (capacity-aware worst-group allocation)",
    "RAND": "defined in the Table 12 note (the null policy)",
    "ORACLE": "defined in the Table 12 note (test-optimal bound, not deployable)",
    "NAT": "defined in the Table 6 note (the natural items of each language)",
    "MAT": "defined in the Table 6 note (both languages forced to the same base rate)",
    "PAIR": "defined in the Table 6 note (English test items translated into Vietnamese)",
    "ECE": "defined in the Table 4 caption (calibration error of the abstention confidence)",
    "AUC": "defined in §5.4 (area under the ROC curve)",
    "SD": "defined in the Table 11 caption (standard deviation)",
    "RTL": "defined in §5.1 (Arabic, right-to-left)",
    "CJK": "defined in §5.1 (Chinese–Japanese–Korean)",
    "PP": "defined by every caption that uses it ('in percentage points')",
    "KB": "notation K_B, defined in §3.1",
    "KQ": "notation K_Q, defined in §3.1",
    "EN": "the English side of the EN/VI pair, spelled out in §5.1 and Table 1",
    "VI": "the Vietnamese side of the EN/VI pair, spelled out in §5.1 and Table 1",
    "EN/VI": "the English–Vietnamese pair, spelled out in §5.1",
    "LLM": "standard, and the paper writes 'large-language-model moderators' where it matters",
    "API": "standard",
    "JSON": "standard",
    "DOI": "standard",
    "CREDIT": "the CRediT taxonomy, named as such in its own heading",
    "SEED": "'the SEED-level test', contrasted with the item level in the same sentence",
    "ITEMS": "capitalised for emphasis in the table notes ('reviewed ITEMS'), not an abbreviation",
    "IS": "part of the Moderator prompt literal in Appendix D ('the comment IS hateful or toxic'), not an abbreviation",
    "BUDGET": "capitalised for emphasis in the table notes ('the review BUDGET'), not an abbreviation",
    "PAN": "the CLEF PAN lab, a proper name",
    "RQ1": "defined in §1 as 'three research questions (RQ1–RQ3)'",
    "RQ2": "defined in §1 as 'three research questions (RQ1–RQ3)'",
    "RQ3": "defined in §1 as 'three research questions (RQ1–RQ3)'",
    "XLM-R": "the model's short name; Table 2 prints the checkpoint id xlm-roberta-base",
    "AI": "standard ('AI-assisted technologies', 'human–AI moderation')",
    "L-1": "notation L − 1; the minus sign is an equation glyph, so the text view glues the two",
    "L1": "notation for the L1 norm, defined in Appendix D ('we report the L1 norm')",
    "KQ-K": "notation |K_Q − K*|; the equation glyphs glue the text view into one token",
}

ALLOWED_VARIANT = {
    "monotone/monotonic": "'monotone' is the adjective, 'monotonically' the adverb — different roles",
    "per-item/per item": "attributive vs adverbial: 'per-item review cost' vs 'takes longer per item'",
    "per-group/per group": "attributive vs adverbial",
    "per-language/per language": "attributive vs adverbial",
    "re-saved/resaved": "only 're-saved' occurs",
    "accented-british": "the manuscript is British English throughout (analysed, behaviour, modelling)",
}

TERM_FAMILIES = [
    ("hate speech", [r"hate speech", r"hate-speech", r"Hate Speech"]),
    ("pre-trained", [r"pre-trained", r"pretrained"]),
    ("fine-tuning", [r"fine-tuning", r"finetuning", r"fine tuning"]),
    ("cross-lingual", [r"cross-lingual", r"cross lingual"]),
    ("low-resource", [r"low-resource", r"low resource"]),
    ("code-mixing", [r"code-mixing", r"code mixing"]),
    ("multi-task", [r"multi-task", r"multitask"]),
    ("multimodal", [r"multimodal", r"multi-modal"]),
    ("zero-shot", [r"zero-shot", r"zero shot"]),
    ("few-shot", [r"few-shot", r"few shot"]),
    ("social media", [r"social media", r"social-media"]),
    ("dataset", [r"dataset", r"data set", r"data-set"]),
    ("labelling", [r"labelling", r"labeling"]),
    ("knowledge graph", [r"knowledge graph", r"knowledge-graph"]),
    ("group-conditional", [r"group-conditional", r"group conditional"]),
    ("held-out", [r"held-out", r"held out"]),
    ("machine-translated", [r"machine-translated", r"machine translated"]),
    ("two-sided", [r"two-sided", r"two sided"]),
    ("pre-specified", [r"pre-specified", r"prespecified"]),
    ("per-item", [r"per-item", r"per item"]),
    ("per-group", [r"per-group", r"per group"]),
    ("per-language", [r"per-language", r"per language"]),
    ("worst-group", [r"worst-group", r"worst group"]),
    ("cost-blind", [r"cost-blind", r"cost_blind"]),
    ("cost-aware", [r"cost-aware", r"cost_aware"]),
    ("cross-entropy", [r"cross-entropy", r"cross entropy"]),
]

# Each entry is (family, British patterns, American patterns). Only a family with a hit on BOTH sides is a
# finding: a manuscript that writes 'analysed' throughout is British, not inconsistent.
UK_US = [
    ("-ise/-ize", [r"\banalys", r"\bnormalis", r"\bminimis", r"\bmaximis", r"\boperationalis",
                   r"\boptimis", r"\bstandardis", r"\bgeneralis", r"\bcharacteris", r"\bvisualis",
                   r"\bprioritis"],
     [r"\banalyz", r"\bnormaliz", r"\bminimiz", r"\bmaximiz", r"\boperationaliz", r"\boptimiz",
      r"\bstandardiz", r"\bgeneraliz", r"\bcharacteriz", r"\bvisualiz", r"\bprioritiz"]),
    ("-our/-or", [r"\bbehaviour", r"\bcolour", r"\bfavour"],
     [r"\bbehavior", r"\bcolor\b", r"\bfavor"]),
    ("-ll-", [r"\bmodelled", r"\bmodelling", r"\blabelled", r"\blabelling", r"\bcancelled",
              r"\btravelled", r"\bsignalling"],
     [r"\bmodeled", r"\bmodeling", r"\blabeled", r"\blabeling", r"\bcanceled", r"\btraveled",
      r"\bsignaling"]),
    ("-re/-er", [r"\bcentre\b", r"\bmetre\b", r"\btheatre\b"],
     [r"\bcenter\b", r"\bmeter\b", r"\btheater\b"]),
    ("misc", [r"\bprogramme\b", r"\bcatalogue\b", r"\bdefence\b", r"\blicence\b"],
     [r"\bprogram\b", r"\bcatalog\b", r"\bdefense\b", r"\blicense\b"]),
]

CAP_RE = re.compile(r"^(Figure|Table)\s+([A-E])?\.?(\d+)\.\s")
CITE_RE = re.compile(r"\b(Figures?|Tables?|Fig\.)\s*([A-E])?\.?(\d+)"
                     r"(?:\s*(?:,|and|&|through|to|\u2013|-)\s*([A-E])?\.?(\d+))?")


# ── views ─────────────────────────────────────────────────────────────────────────────────────────
def para_text(p_el):
    return "".join(c.text or "" for c in p_el.iter() if c.tag in (qn("w:t"), qn("m:t")))


def load(path):
    doc = Document(path)
    blocks, pi, ti = [], 0, 0
    for ch in doc.element.body.iterchildren():
        if ch.tag == qn("w:p"):
            blocks.append({"kind": "p", "i": pi, "text": para_text(ch),
                           "style": Paragraph(ch, doc).style.name or ""})
            pi += 1
        elif ch.tag == qn("w:tbl"):
            t = Table(ch, doc)
            rows = [["\n".join(para_text(p) for p in c._tc.findall(qn("w:p"))).strip()
                     for c in r.cells] for r in t.rows]
            blocks.append({"kind": "tbl", "i": ti, "rows": rows})
            ti += 1
    return doc, blocks


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def census(text, pats):
    """Boundary regexes and case-sensitive counts: 'AI adoption' and 'ai adoption' are one string."""
    out = []
    for p in pats:
        n = len(re.findall(p, text))
        if n:
            out.append((p, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", default=DOCX)
    ap.add_argument("--json", default=None)
    ap.add_argument("--assertions", default=None)
    args = ap.parse_args()
    if not args.docx:
        sys.exit("FAIL: no manuscript given — pass --docx <path.docx> or set CAPRA_DOCX "
                 "(the manuscript .docx is not part of this release)")

    doc, blocks = load(args.docx)
    paras = [b for b in blocks if b["kind"] == "p"]
    tbls = [b for b in blocks if b["kind"] == "tbl"]
    heads = {b["text"].strip() for b in paras if b["style"].startswith("Heading")}
    ref_i = next((b["i"] for b in paras if b["text"].strip() == "References"), None)
    prose = [b for b in paras if ref_i is None or b["i"] < ref_i]
    refs = [b["text"] for b in paras if ref_i is not None and b["i"] > ref_i and b["text"].strip()]
    PROSE = "\n".join(b["text"] for b in prose)
    CELLS = "\n".join(" | ".join(c for r in t["rows"] for c in r) for t in tbls)
    ALL = PROSE + "\n" + CELLS
    ASSERT_VIEW = ALL + "\n" + "\n".join(refs)   # a reference entry is text too

    F = []                                    # findings

    def add(sev, module, what, where=""):
        F.append({"severity": sev, "module": module, "finding": what, "where": where})

    # ── M1 abbreviations ─────────────────────────────────────────────────────────────────────────
    caps = Counter(re.findall(r"(?<![A-Za-z0-9\-])[A-Z][A-Z0-9\-]{1,7}(?![A-Za-z0-9\-])", PROSE))
    for tok, n in sorted(caps.items()):
        if n >= 2 and tok not in ALLOWED_ABBREV:
            add("missing-def", "M1", f"'{tok}' used {n}x and never expanded", "")

    # ── M2 terms and spelling ────────────────────────────────────────────────────────────────────
    for name, pats in TERM_FAMILIES:
        hits = census(PROSE, pats)
        if len(hits) > 1 and f"{name}/{hits[1][0]}" not in ALLOWED_VARIANT:
            add("variant", "M2", f"{name}: " + ", ".join(f"{p!r} x{n}" for p, n in hits), "")
    for name, brit, amer in UK_US:
        b, a = census(PROSE, brit), census(PROSE, amer)
        if b and a:
            add("variant", "M2", f"UK/US {name}: " + ", ".join(f"{p!r} x{n}" for p, n in b + a), "")

    # ── M3 notation ──────────────────────────────────────────────────────────────────────────────
    tight = sorted({f"P{b['i']:04d}" for b in prose if re.search(r"\d%", b["text"])} |
                   {f"T{t['i']}" for t in tbls for r in t["rows"] for c in r if re.search(r"\d%", c)})
    if tight:
        add("mismatch", "M3", "percent sign without a space (the paper writes '5 %')", ", ".join(tight))
    for bad, good in ((r"0-40 %", "0\u201340 %"), (r"\b\d{1,3}[\u00a0\u2009 ]\d{3}\b", "no separator")):
        for b in prose:
            if re.search(bad, b["text"]):
                add("mismatch", "M3", f"{bad!r} in prose (expected {good!r})", f"P{b['i']:04d}")
    labels = {lbl for lbl in ("CAPRA-parity", "CAPRA-composition", "PF", "CF", "Δ CAPRA")
              if lbl in ALL}
    if re.search(r"Δ CAPRA(?![\w-])", ALL) or "| CAPRA |" in CELLS:
        add("mismatch", "M3", f"endpoint labels not unified: {sorted(labels)}", "")
    eqs = [(b["i"], m.group(1)) for b in paras
           for m in [re.search(r"\((\d{1,2})\)\s*$", b["text"])] if m]
    nums = [int(n) for _, n in eqs]
    if nums != sorted(nums) or (nums and nums != list(range(1, len(nums) + 1))):
        add("mismatch", "M3", f"display-equation numbers are not 1..N in order: {nums}", "")

    # ── M4 model / dataset names ─────────────────────────────────────────────────────────────────
    for name, pats in (("PhoBERT", [r"\bPhoBERT\b", r"\bphobert\b", r"\bPhobert\b"]),
                       ("XLM-R", [r"\bXLM-R\b", r"\bXLMR\b", r"\bxlm-roberta\b"]),
                       ("RoBERTa", [r"\bRoBERTa\b", r"\bRoberta\b", r"\broberta\b"]),
                       ("CAPRA", [r"\bCAPRA\b", r"\bCapra\b"]),
                       ("ViHSD", [r"\bViHSD\b", r"\bVIHSD\b"])):
        hits = [(p, len(re.findall(p, ALL))) for p in pats]
        hits = [h for h in hits if h[1]]
        prose_hits = [h for h in hits if len(re.findall(h[0], PROSE))]
        # a lowercase form is legal only inside a table cell (a HuggingFace id or a dataset key)
        if len(prose_hits) > 1:
            add("variant", "M4", f"{name}: " + ", ".join(f"{p!r} x{n}" for p, n in prose_hits), "")

    # ── M5 cross-references and the reference list ───────────────────────────────────────────────
    for m in re.finditer(r"\bSection\s+(\d+(?:\.\d+)?)", ALL):
        n = m.group(1)
        if not any(h == n or h.startswith(n + ".") or h.startswith(n + " ") for h in heads):
            add("broken", "M5", f"Section {n} is cited but has no heading", "")
    for m in re.finditer(r"\bAppendix\s+([A-E])", ALL):
        if not any(h.startswith(f"Appendix {m.group(1)}") for h in heads):
            add("broken", "M5", f"Appendix {m.group(1)} is cited but has no heading", "")
    for m in re.finditer(r"\bEquation\s+(\d+)", ALL):
        if not re.search(rf"(?<![0-9]){m.group(1)}\)", ALL):
            add("broken", "M5", f"Equation {m.group(1)} is cited but not numbered", "")
    if refs:
        allrefs = " ".join(refs).lower()
        numbered = [re.match(r"^\[(\d+)\]", t) for t in refs]
        if all(numbered):
            # Numbered, citation-order style (the author's Zotero refresh): the invariances are
            # 1..N in list order, every entry cited, and no in-text number without an entry. The
            # alphabetical and surname checks below describe an author-date list and are skipped.
            nums = [int(m.group(1)) for m in numbered]
            if nums != list(range(1, len(nums) + 1)):
                bad = [n for i, n in enumerate(nums) if n != i + 1]
                add("mismatch", "M5", f"numbered reference list is not 1..{len(refs)} in list order "
                                      f"(offending numbers: {bad[:5]})", "")
            cited = {int(x) for x in re.findall(r"\[(\d+)\]", ALL)}
            for n in nums:
                if n not in cited:
                    add("broken", "M5", f"reference [{n}] is never cited in the text", "")
            dangling = sorted(c for c in cited if c > len(nums))
            if dangling:
                add("broken", "M5", f"in-text citations {dangling} have no reference entry", "")
        else:
            keys = [re.match(r"([^,(]+)", t).group(1).strip().lower() for t in refs]
            if keys != sorted(keys):
                bad = [k for k, s in zip(keys, sorted(keys)) if k != s]
                add("mismatch", "M5", f"reference list is not alphabetical (first divergence: {bad[:2]})", "")
            for t in refs:
                m = re.match(r"([^,(]+)", t)
                surname = m.group(1).strip().split()[0].rstrip(",")
                if len(surname) > 3 and surname.lower() not in PROSE.lower():
                    add("broken", "M5", f"reference '{surname}' is never cited in the text", "")
        for m in re.finditer(r"\(([^()]*?\d{4})\)", PROSE):
            for part in m.group(1).split(";"):
                mm = re.match(r"\s*([A-Z][^,]*?),?\s*(?:et al\.)?\s*(\d{4})", part.strip())
                if not mm:
                    continue
                # "Gibbs & Candès" cites one entry under either surname, so test both sides
                for who in re.split(r"\s*&\s*", mm.group(1)):
                    # "Ross et al." must yield "Ross", not the literal "al." (the trailing token) --
                    # otherwise the check silently passes for every et-al citation whose reference is missing
                    who = re.sub(r"\s*et\s+al\.?$", "", who.strip())
                    surname = who.strip().split()[-1]
                    if len(surname) > 3 and surname.lower() not in allrefs:
                        add("broken", "M5", f"in-text citation '{surname}' has no reference entry", "")

    # ── M5b float order ──────────────────────────────────────────────────────────────────────────
    captions, mentions = [], {}
    for pos, b in enumerate(paras):
        t = b["text"].strip()
        if not t:
            continue
        cap = CAP_RE.match(t)
        if cap:
            captions.append((pos, cap.group(1), cap.group(2) or "", int(cap.group(3))))
            continue
        for cm in CITE_RE.finditer(t):
            k = "Figure" if cm.group(1).startswith("Fig") else "Table"
            mentions.setdefault((k, cm.group(2) or "", int(cm.group(3))), pos)
            if cm.group(5):
                mentions.setdefault((k, cm.group(4) or cm.group(2) or "", int(cm.group(5))), pos)
    for scope, want in (("main text", ""), ("appendix", None)):
        caps = [c for c in captions if (c[2] == want if want is not None else c[2] != "")]
        if not caps:
            continue
        for kind in ("Figure", "Table"):
            seq = [(c[2], c[3]) for c in caps if c[1] == kind]
            if seq != sorted(seq):
                add("mismatch", "M5b", f"{scope}: {kind.lower()} captions do not ascend", f"{seq}")
            cited = sorted([(c[1], c[2], c[3]) for c in caps if c[1] == kind
                            and (c[1], c[2], c[3]) in mentions], key=lambda c: mentions[c])
            pairs = [(c[1], c[2]) for c in cited]
            if pairs != sorted(pairs):
                bad = [f"{cited[i-1][0]}{cited[i-1][1]}{cited[i-1][2]} before "
                       f"{cited[i][0]}{cited[i][1]}{cited[i][2]}"
                       for i in range(1, len(cited)) if pairs[i] < pairs[i - 1]]
                add("mismatch", "M5b", f"{scope}: {kind.lower()} first citations out of order",
                    "; ".join(bad))
        never = [f"{c[1]}{c[2]}{c[3]}" for c in caps if (c[1], c[2], c[3]) not in mentions]
        if never:
            add("broken", "M5b", f"{scope}: float(s) never cited", ", ".join(never))

    # ── M6 numbers in prose vs tables ────────────────────────────────────────────────────────────
    tbl_vals = {m.group(0) for t in tbls for r in t["rows"] for c in r
                for m in re.finditer(r"\d+(?:\.\d+)?", c)}
    def _no_refs(t):
        return re.sub(r"(?:Section|Sections|Appendix)\s+[A-E]?\d+(?:\.\d+)?", " ", t)

    orphan = sorted({m.group(0) for b in prose
                     if not CAP_RE.match(b["text"]) and not b["style"].startswith("Heading")
                     for m in re.finditer(r"\d+\.\d+|\d+(?=\s*(?:%|pp|per cent))", _no_refs(b["text"]))
                     if m.group(0) not in tbl_vals and not re.match(r"^\d+\.\d+$", m.group(0)) is None
                     and float(m.group(0)) > 0.3}, key=float)
    if orphan:
        add("note", "M6", "prose numbers that appear in no table (they may live in the artefacts)",
            ", ".join(orphan[:20]))

    # ── M7 captions and notes ────────────────────────────────────────────────────────────────────
    for pos, b in enumerate(blocks):
        if b["kind"] != "tbl":
            continue
        cap = blocks[pos - 1]["text"].strip() if pos and blocks[pos - 1]["kind"] == "p" else ""
        note = blocks[pos + 1]["text"].strip() if pos + 1 < len(blocks) and \
            blocks[pos + 1]["kind"] == "p" else ""
        txt = " | ".join(c for r in b["rows"] for c in r)
        label = CAP_RE.match(cap).group(0).strip().rstrip(".") if CAP_RE.match(cap) else f"table {b['i']}"
        if "\u00b1" in txt and not re.search(r"\bSD\b|standard deviation|\u00b1 (on|is)",
                                              cap + " " + note + " " + txt):
            add("missing-def", "M7",
                f"{label} prints a ± that neither its caption nor its note defines", "")

    # ── M7b one source, ledger, render ───────────────────────────────────────────────────────────
    live = md5(args.docx)
    if os.path.exists(CHANGELOG) and os.path.abspath(args.docx) == os.path.abspath(DOCX):
        rows = [ln for ln in open(CHANGELOG, encoding="utf-8") if ln.startswith("| 20")]
        # A note row that records a FAILED declare-error or a manual write carries no md5; skip those
        # instead of crashing on them (round-10 fix: the round-9 note row made this gate unusable).
        marks = [m for ln in rows for m in re.findall(r"`([0-9a-f]{12})`", ln)]
        last = marks[-1] if marks else None
        if last and not live.startswith(last):
            add("provenance", "M7b", f"the live file ({live[:12]}) is not the ledger's last state "
                                     f"({last}) — a write bypassed the tooling", "")
        render = os.path.join(os.path.dirname(args.docx), "CAPRA_manuscripts.pdf")
        if os.path.exists(render) and os.path.getmtime(render) < os.path.getmtime(args.docx):
            add("provenance", "M7b", "the render is older than the manuscript of record", render)
        with zipfile.ZipFile(args.docx) as z:
            xml = z.read("word/document.xml").decode("utf-8")
            core = z.read("docProps/core.xml").decode("utf-8")
            if any("comment" in n.lower() for n in z.namelist()) or "<w:ins " in xml \
                    or "<w:del " in xml or "EE0000" in xml:
                add("provenance", "M7b", "tracked changes, comments or review colour in the file", "")
        meta = re.search(r"<dc:title>(.*?)</dc:title>", core)
        if meta and paras and meta.group(1) != paras[0]["text"]:
            add("provenance", "M7b", f"dc:title does not match the title paragraph: {meta.group(1)[:60]!r}",
                "")
        stray = [os.path.join(dp, f) for dp, _, fs in os.walk(REPO) for f in fs
                 if f.lower().endswith((".docx", ".pdf"))
                 and not any(part in ("backup", "archive", "figures", "ipm_study", ".git")
                             for part in os.path.join(dp, f).split(os.sep))
                 and not f.startswith(("fig_", "Figure_")) and "manuscript" in f.lower()]
        if stray:
            add("provenance", "M7b", "a manuscript-like copy lives in the repo", ", ".join(stray))

    # ── standing assertions ──────────────────────────────────────────────────────────────────────
    if args.assertions and os.path.exists(args.assertions):
        for a in json.load(open(args.assertions, encoding="utf-8")):
            for s in a.get("must", []):
                if s not in ASSERT_VIEW:
                    add("mismatch", "assert", f"{a['label']}: {s!r} is missing", "")
            for s in a.get("must_not", []):
                if s in ASSERT_VIEW:
                    add("mismatch", "assert", f"{a['label']}: {s!r} is still present", "")

    # ── report ───────────────────────────────────────────────────────────────────────────────────
    bad = [f for f in F if f["severity"] in FAILING]
    for f in F:
        print(f"{'FAIL' if f['severity'] in FAILING else 'note'}  [{f['severity']:11s}] "
              f"{f['module']:4s} {f['finding']}" + (f"  ::  {f['where']}" if f["where"] else ""))
    print(f"\n{len(bad)} failing / {len(F) - len(bad)} note  ·  "
          f"{len(paras)} paragraphs, {len(tbls)} tables, {len(refs)} references  ·  md5 {live[:12]}")
    if args.json:
        json.dump({"docx": args.docx, "md5": live, "findings": F}, open(args.json, "w", encoding="utf-8"),
                  indent=1, ensure_ascii=False)
        print(f"json: {args.json}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
