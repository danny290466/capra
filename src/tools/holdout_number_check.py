#!/usr/bin/env python3
"""Trace every number in the 6.2 hold-out (LOLO/LOSO) sentence back to the artefact.

The sentence was written from notes/lolo_loso.json, so the check re-derives each value from that file and
asserts the manuscript carries the same rounding. Anything hand-typed, mis-rounded or copied from the wrong
pair shows up here instead of in review.

    python3 tools/holdout_number_check.py            # check
    python3 tools/holdout_number_check.py --md       # print the derived list
"""
import argparse
import json
import os
import re
import sys

import docx
from decimal import Decimal, ROUND_HALF_UP

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(REPO, "notes", "lolo_loso.json")
TARGET = os.environ.get("CAPRA_DOCX")   # the manuscript .docx is not part of this release
ANCHOR = "With Chinese absent the detector keeps"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=ART)
    ap.add_argument("--file", default=TARGET)
    ap.add_argument("--md", action="store_true", help="print each derived value and whether it is present")
    a = ap.parse_args()
    if not a.file:
        sys.exit("FAIL: no manuscript given — pass --file <path.docx> or set CAPRA_DOCX "
                 "(the manuscript .docx is not part of this release)")

    d = json.load(open(a.json, encoding="utf-8"))
    sh, ct = d["shipped"], d["controls"]
    zh = ct["lolo_zh (train en,de,uk,ru,ar,hi; score zh)"]["en-zh"]
    ru = ct["lolo_ru (train en,de,uk,ar,hi,zh; score ru)"]["en-ru"]
    cyr_ru = ct["loso_cyr (train en,de,ar,hi,zh; score ru,uk)"]["en-ru"]
    cyr_uk = ct["loso_cyr (train en,de,ar,hi,zh; score ru,uk)"]["en-uk"]
    s_zh, s_ru, s_uk = sh["en-zh"], sh["en-ru"], sh["en-uk"]

    def f4(x):                       # accuracy / AUC
        return str(Decimal(str(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

    def f2(x):                       # gaps, capacities, loads
        return str(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    expected = [
        ("zh unseen accuracy", f4(zh["accuracy"]["zh"])),
        ("zh unseen auc", f4(zh["auc"]["zh"])),
        ("zh seen accuracy", f4(s_zh["accuracy"]["zh"])),
        ("zh seen auc", f4(s_zh["auc"]["zh"])),
        ("zh unseen pooled gap", f2(zh["pooled_gap_pp"])),
        ("zh seen pooled gap", f2(s_zh["pooled_gap_pp"])),
        ("zh unseen load@5", f'{zh["parity_load_by_risk_target_pct"]["0.05"]:.1f} %'),
        ("zh seen load@5", f'{s_zh["parity_load_by_risk_target_pct"]["0.05"]:.1f} %'),
        ("ru control acc", f4(ru["accuracy"]["ru"])),
        ("ru seen acc", f4(s_ru["accuracy"]["ru"])),
        ("ru control pooled gap", f2(ru["pooled_gap_pp"])),
        ("ru seen pooled gap", "+" + f2(s_ru["pooled_gap_pp"])),
        ("ru control parity-first gap", "+" + f2(ru["parity_first_gap_signed_pp"])),
        ("ru seen parity-first gap", "+" + f2(s_ru["parity_first_gap_signed_pp"])),
        ("ru control load@5", f'{ru["parity_load_by_risk_target_pct"]["0.05"]:.1f} %'),
        ("ru seen load@5", f'{s_ru["parity_load_by_risk_target_pct"]["0.05"]:.1f} %'),
        ("ru min capacity control", f'{ru["min_parity_capacity_pct"]:.1f} %'),
        ("ru min capacity seen", f'{s_ru["min_parity_capacity_pct"]:.1f} %'),
        ("cyr ru acc", f4(cyr_ru["accuracy"]["ru"])),
        ("cyr ru pooled gap", f2(cyr_ru["pooled_gap_pp"])),
        ("cyr ru load@5", f'{cyr_ru["parity_load_by_risk_target_pct"]["0.05"]:.1f} %'),
        ("cyr uk acc", f4(cyr_uk["accuracy"]["uk"])),
        ("cyr uk auc", f4(cyr_uk["auc"]["uk"])),
        ("uk seen acc", f4(s_uk["accuracy"]["uk"])),
        ("uk seen auc", f4(s_uk["auc"]["uk"])),
        ("cyr uk pooled gap seen", f2(s_uk["pooled_gap_pp"])),
        ("cyr uk pooled gap control", f2(cyr_uk["pooled_gap_pp"])),
        ("cyr uk load@5 seen", f'{s_uk["parity_load_by_risk_target_pct"]["0.05"]:.1f} %'),
        ("cyr uk load@5 control", f'{cyr_uk["parity_load_by_risk_target_pct"]["0.05"]:.1f} %'),
    ]

    doc = docx.Document(a.file)
    para = next((p.text for p in doc.paragraphs if ANCHOR in p.text), None)
    if para is None:
        sys.exit("FAIL: the 6.2 hold-out sentence is not in the manuscript")
    sentence = para[para.index(ANCHOR):]

    bad = []
    for label, want in expected:
        num = want.strip().rstrip(" %").lstrip("+")
        present = (want in sentence) or (num in sentence)
        if not present:
            bad.append((label, want))
        if a.md:
            print(f"  {'OK ' if present else 'MISS'} {label:<28} {want}")

    # no stray numbers: every decimal in the sentence must be one we derived
    derived = {w.strip().rstrip(" %").lstrip("+") for _l, w in expected}
    strays = sorted({m for m in re.findall(r"-?\d+\.\d+", sentence)} - derived)
    if strays:
        bad.append(("numbers in the sentence that the artefact does not produce", ", ".join(strays)))

    print(f"sentence length: {len(sentence)} chars | values derived: {len(expected)} | strays: {len(strays)}")
    if bad:
        for label, want in bad:
            print(f"FAIL  {label}: expected {want!r} in the sentence")
        sys.exit(1)
    print("VERIFY: OK — every number in the 6.2 sentence matches notes/lolo_loso.json")


if __name__ == "__main__":
    main()
