#!/usr/bin/env python3
"""Check every number Appendix C cites against the artefacts it was generated from.

The appendix (and Table C.1) was written by a generator that read `capra_lgroup_*.json`; this script reads
the SAME artefacts and the manuscript of record, and fails if a quoted value is not in the JSON (or is not
consistent with the JSON to the printed precision). It exists because the first version of Table C.1 shipped
with the parity-first test columns copied into the pooled and oracle rows — a generator bug that a check of
this kind catches in one run.

usage: python3 src/tools/lgroup_number_check.py        (set CAPRA_DOCX to the manuscript .docx)
"""
import json
import os
import re
import sys

DOCX = os.environ.get("CAPRA_DOCX")   # the manuscript .docx is not part of this release
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
J = {"base": os.path.join(ROOT, "outputs/textdetox/capra_lgroup_xlmr_base.json"),
     "large": os.path.join(ROOT, "outputs/textdetox/capra_lgroup_xlmr_large.json"),
     "envi": os.path.join(ROOT, "outputs/killtest/capra_lgroup_envi.json")}


def f2(x):
    return f"{x * 100:.2f}"


def main():
    from docx import Document
    from docx.text.paragraph import Paragraph

    if not DOCX:
        sys.exit("FAIL: no manuscript given — set CAPRA_DOCX to the manuscript .docx "
                 "(it is not part of this release)")
    art = {k: json.load(open(v)) for k, v in J.items()}
    doc = Document(DOCX)
    paras = [Paragraph(e, doc) for e in doc.element.body.xpath(".//w:p")]
    txt = "\n".join(p.text for p in paras)
    i = txt.find("Appendix C. More than two groups")
    j = txt.find("References", i)
    appx = txt[i:j] if i >= 0 and j > i else ""
    if not appx:
        sys.exit("FAIL: Appendix C is not in the manuscript (or is not before the References heading)")

    # Headings carry Word AUTO-NUMBERING in this manuscript, so the literal "7. " prefix is never in the
    # text: locate the section by the HEADING PARAGRAPH whose text starts with the title, not by a string
    # search (a tail mention of the same words otherwise moves the region start past the paragraphs the
    # gate is supposed to read). Region = that heading .. the References heading. A failed anchor must be
    # loud: falling back silently to the appendix is how this gate once passed while reading the wrong text.
    offs, pos = [], 0
    for p in paras:
        offs.append((pos, p.text.strip()))
        pos += len(p.text) + 1
    K2 = next((o for o, t in offs if re.match(r"^\d+\.\s*limitations\b", t, re.I)), -1)
    refs = next((o for o, t in offs if re.fullmatch(r"\d*\.?\s*references", t.strip(), re.I)), -1)
    print(f"lgroup region: section heading at {K2}, references at {refs}, appendix at {i}")
    assert K2 >= 0, "FAIL: could not locate the Limitations heading (auto-numbered headings have no '7. ' in the text)"
    end = refs if refs > K2 else j
    quoted = txt[K2:end] if 0 <= K2 < end else appx

    A, B, V = art["base"], art["large"], art["envi"]
    poT, pfT = A["pooled"]["test"], A["parity_first"]["test"]
    front = {float(t["delta"]): t for t in A["tradeoff_frontier"]}
    frontL = {float(t["delta"]): t for t in B["tradeoff_frontier"]}
    capA = {float(c["budget"]): c["design_max_gap"] for c in A["capacity_sweep"]}
    capB = {float(c["budget"]): c["design_max_gap"] for c in B["capacity_sweep"]}
    K = dict(zip(A["langs"], pfT["K"]))

    checks = [
        ("pooled test worst gap", f2(poT["max_gap"]), f2(poT["max_gap"])),
        ("parity-first test worst gap", f2(pfT["max_gap"]), f2(pfT["max_gap"])),
        ("share of queue in two groups (%)", f"{100*(K['zh']+K['ar']):.0f}", "94"),
        ("share left to the other five (%)", f"{100*sum(K[l] for l in K if l not in ('zh','ar')):.1f}", "5.7"),
        ("composition deviation L1", f"{pfT['comp_dev_l1']:.3f}", "1.314"),
        ("price of composition (pp)", f2(poT["max_gap"] - pfT["max_gap"]), "5.22"),
        ("frontier gain base (pp)", f2(front[0.0]["test_max_gap"] - front[0.6]["test_max_gap"]), "5.06"),
        ("frontier gain large (pp)", f2(frontL[0.0]["test_max_gap"] - frontL[0.6]["test_max_gap"]), "4.55"),
        ("frontier base end (pp)", f2(front[0.6]["test_max_gap"]), "17.68"),
        ("frontier large end (pp)", f2(frontL[0.6]["test_max_gap"]), "18.56"),
        ("7-way capacity base (pp at 35 %)", f2(capA[0.35]), "0.43"),
        ("7-way capacity large (pp at 35 %)", f2(capB[0.35]), "0.93"),
        ("seed SD base (pp)", f2(A["parity_first"]["seed_max_gap_sd"]), "1.28"),
    ]
    for lg in ("en", "ru", "zh", "ar"):
        checks.append((f"risk {lg} pooled -> parity-first",
                       f"{f2(poT['risk'][lg])} to {f2(pfT['risk'][lg])}",
                       f"{f2(poT['risk'][lg])} to {f2(pfT['risk'][lg])}"))
    checks.append(("two-group validation design (pp)", f2(V["parity_first"]["design"]["max_gap"]), "1.99"))
    checks.append(("two-group validation test (pp)", f2(V["parity_first"]["test"]["max_gap"]), "0.94"))

    # Numbers the appendix cuts of 2026-09-24 removed from the manuscript (DECISIONS 141-143): the tool
    # still cross-checks each value against the artefact, but does not require it to be printed.
    CUT = {
        "share left to the other five (%)": "not printed: left with the cut prose",
        "price of composition (pp)": "not printed: the appendix reports the price qualitatively",
        "frontier base end (pp)": "not printed: left with the cut prose",
        "frontier large end (pp)": "not printed: left with the cut prose",
        "7-way capacity base (pp at 35 %)": "not printed: stated as 'about 35 % of traffic'",
        "7-way capacity large (pp at 35 %)": "not printed: stated as 'about 35 % of traffic'",
    }
    bad, notes = [], []
    # the 30 % behaviour is no longer stated in prose (it went with the appendix cuts): check that the
    # artefact still supports the claim the cut prose made, not that the sentence is back
    if not (capA[0.30] > 0.02 and capB[0.30] > 0.02):
        bad.append(f"the cut prose claimed both models are still above 2 pp at 30 %, artefacts give "
                   f"{f2(capA[0.30])} and {f2(capB[0.30])} pp")
    else:
        notes.append("30 % behaviour: both models above 2 pp at 30 %, as the cut prose said")
    for name, derived, expected in checks:
        if derived != expected:
            bad.append(f"{name}: the artefact gives {derived}, the check expects {expected}")
        if name in CUT:
            notes.append(f"{name}: {expected} {CUT[name]}")
        elif expected not in quoted:
            bad.append(f"{name}: {expected!r} is not in the quoted text (Section 7 + appendix)")

    # Table C.1: rebuild every row from the artefacts and compare cell by cell.  The table is located by
    # its HEADER, not by position: `doc.tables[-1]` silently became a different table once later sections
    # gained tables of their own, and a positional selector would then compare against the wrong object
    # (or crash on the column count) while still reporting on Table C.1.
    def _hdr(tbl):
        return " ".join(c.text.strip().lower() for c in tbl.rows[0].cells)

    t = next((x for x in reversed(doc.tables)
              if "design (pp)" in _hdr(x) and "test (pp)" in _hdr(x)), None)
    if t is None:
        sys.exit("FAIL: Table C.1 is not in the manuscript — no table has the 'Design (pp)'/'Test (pp)' header")
    want = []
    for det, blk in (("XLM-R base", A["pooled"]), ("XLM-R base", A["parity_first"]), ("XLM-R base", A["oracle"]),
                     ("XLM-R large", B["pooled"]), ("XLM-R large", B["parity_first"]), ("XLM-R large", B["oracle"]),
                     ("XLM-R base, EN/VI (L = 2, validation)", V["parity_first"])):
        b = blk["test"]
        want.append([det, f2(blk["design"]["max_gap"]), f2(b["max_gap"]), f"{b['comp_dev']:.3f}",
                     f"{b['comp_dev_l1']:.3f}", f"{b['load']:.3f}"])
    def cell_text(cell):
        """Cell text INCLUDING equation glyphs: a cell whose "L = 2" is an m:oMath has no w:t for it."""
        from docx.oxml.ns import qn
        parts = []
        for para in cell.paragraphs:
            for child in para._p:
                if child.tag == qn("w:r"):
                    parts.append("".join(x.text or "" for x in child.findall(qn("w:t"))))
                else:
                    parts.append("".join(x.text or "" for x in child.iter(qn("m:t"))))
        return "".join(parts).strip()

    def norm(s_):
        return " ".join(s_.replace("=", " = ").split())

    for r, w in enumerate(want, start=1):
        got = [cell_text(t.rows[r].cells[c]) for c in (0, 2, 3, 4, 5, 6)]
        if [norm(g) for g in got] != [norm(x) for x in w]:
            bad.append(f"Table C.1 row {r}: {got} != {w}")

    print(f"appendix length: {len(appx)} chars | Table C.1: {len(t.rows)} rows")
    for n in notes:
        print("  note:", n)
    if bad:
        print("FAIL")
        for b in bad:
            print("  -", b)
        return 1
    print(f"PASS — {len(checks)} quoted numbers and {len(want)} table rows agree with the artefacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
