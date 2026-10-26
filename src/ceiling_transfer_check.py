#!/usr/bin/env python3
"""Round-4 external review, item 2.1 — does the risk ceiling that selects the constrained allocation
still hold on the held-out split?

`capra.py` selects the constrained solution (min D subject to r_bar <= ceiling, the ceiling being the
average risk of the pooled threshold at the same realised load) on the CALIBRATION split and reports
its design risk, its test disparity and the design-side flag `parity_first_under_ceiling`. What the
paper never printed is the reported quantity itself: the traffic-weighted average risk of the
constrained solution **on test**, its margin to the ceiling, and whether the ceiling is violated
there — a calibration-feasible point need not stay feasible out of sample.

Also reports the discretisation scale the review asks for (item 2.2): the review share is a sample
quantile, so a reported policy's realised calibration share differs from its design share by at most
one item, 1/n_cal, per language, and the sweep step in the allocation parameter is 0.005.

Writes outputs/r4_ceiling_transfer.json.
"""
import json

W = (0.5, 0.5)                       # traffic weights, equal in every reported configuration
ART = "outputs/killtest/capra.json"


def main():
    j = json.load(open(ART))
    out = {}
    print(f"artifact: {ART} (generated {j.get('generated')}) | detector {j.get('detector')} | pair {j.get('pair')}")
    print(f"{'budget':>6} {'variant':>11} {'ceiling':>8} {'r_bar_design':>12} {'r_bar_test':>11} "
          f"{'margin_test':>11} {'violates_test':>13} {'PF r_bar_test':>13}")
    for budget, blocks in j["budgets"].items():
        for variant, b in blocks.items():
            rc = b.get("risk_ceiling")
            if not rc:
                continue
            t = rc["test"]
            r_test = W[0] * t["en"]["risk"] + W[1] * t["vi"]["risk"]
            ceiling = rc["r_max_design"]
            margin = ceiling - r_test
            pf_test = rc["parity_first_r_bar_test"]
            row = {
                "budget": float(budget), "variant": variant,
                "ceiling_pp": 100 * ceiling,
                "r_bar_design_pp": 100 * rc["r_bar_design"],
                "r_bar_test_pp": 100 * r_test,
                "margin_test_pp": 100 * margin,
                "violates_test": bool(margin < 0),
                "parity_first_r_bar_design_pp": 100 * rc["parity_first_r_bar_design"],
                "parity_first_r_bar_test_pp": 100 * pf_test,
                "parity_first_under_ceiling_design": rc["parity_first_under_ceiling"],
                "alpha": rc["alpha"], "load_design": rc["load_design"],
                "load_test": t["load"],
                "test_en_risk_pp": 100 * t["en"]["risk"], "test_vi_risk_pp": 100 * t["vi"]["risk"],
                "test_en_coverage": t["en"]["coverage"], "test_vi_coverage": t["vi"]["coverage"],
            }
            out.setdefault(variant, {})[budget] = row
            print(f"{budget:>6} {variant:>11} {100*ceiling:8.2f} {100*rc['r_bar_design']:12.2f} "
                  f"{100*r_test:11.2f} {100*margin:11.2f} {str(margin < 0):>13} {100*pf_test:13.2f}")

    allmargins = [r["margin_test_pp"] for v in out.values() for r in v.values()]
    viol = [(v, b) for v, d in out.items() for b, r in d.items() if r["violates_test"]]
    summary = {
        "n_cells": len(allmargins),
        "margin_test_min_pp": min(allmargins), "margin_test_max_pp": max(allmargins),
        "violations_on_test": viol,
        "parity_first_under_ceiling_design_all_false": all(
            r["parity_first_under_ceiling_design"] is False for v in out.values() for r in v.values()),
    }
    print("\nSUMMARY", json.dumps(summary, indent=1))
    # discretisation scale (item 2.2): the review share is a sample quantile, so the realised
    # calibration share is the design share rounded to a multiple of 1/n_cal.
    disc = {v: {"n_cal_en": None} for v in out}
    try:
        z = json.load(open("outputs/killtest/baseline_NAT.json"))
        disc["n_cal_note"] = "see baseline_NAT.json pairs for per-language calibration counts"
        disc["n_cal_en_vi"] = [p.get("n_cal") for p in list(z.get("pairs", {}).values())[:1]]
    except Exception as exc:                              # noqa: BLE001
        disc["error"] = str(exc)
    summary["discretisation"] = {"sweep_step_alpha": 0.005,
                                 "share_quantum_note": "1/n_cal per language (5000 EN, 2672 VI dev items)"}
    with open("outputs/r4_ceiling_transfer.json", "w") as fh:
        json.dump({"per_variant": out, "summary": summary}, fh, indent=1)
    print("wrote outputs/r4_ceiling_transfer.json")


if __name__ == "__main__":
    main()
