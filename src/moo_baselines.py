#!/usr/bin/env python3
"""Round-3 external review, item 2.3 — "what makes CAPRA a new method rather than a
re-representation of a known multi-objective program?"

We run the two standard single-solution multi-objective methods on exactly the objects the paper
already has (the two calibrated risk curves, C = 5 %, equal traffic weights and equal costs, five
training seeds) and compare them with the CAPRA set:

  (a) weighted sum   min_a  lam * Dhat(a) + (1-lam) * Qhat(a)      -- one solution per lam
  (b) eps-constraint min_a  D(a)  s.t.  |K_Q(a) - w_b| <= eps       -- one solution per eps

Both are solved on the CALIBRATION split (as every rule in the paper is) and read back on TEST with
the raw empirical curve, so the numbers are comparable to Table 13/14.

What we report, per seed and aggregated:
  1. containment: the weighted-sum solution is a point of the CAPRA family (so the set generalises
     the method rather than competing with it);
  2. the weight is not a deployment-stated quantity: under three normalisations of the two
     objectives (max, range, raw-pp-vs-share) the set of lam that meets a composition tolerance
     of +-0.05 differs -- i.e. the answer depends on an arbitrary scaling choice;
  3. the eps-constraint needs a level the deployment must already know, and returns ONE point per
     eps: 51 tolerance levels are answered by one sweep of the family;
  4. the price list the set reports: D*(eps) for eps in {0.02, 0.05, 0.10, 0.20} against the
     unconstrained parity-first and the composition-first points.
"""
import json

import numpy as np

import sim_operational as S

TOL = 0.05
LAMS = np.round(np.linspace(0.0, 1.0, 51), 4)
EPSS = np.round(np.arange(0.0, 0.501, 0.01), 4)
NA = 2001


def realised_risk(conf, err, s):
    """Raw empirical kept-set risk at review share s (the convention Table B.1 calls 'raw empirical')."""
    n = len(conf)
    k = int(round(n * (1.0 - s)))
    k = min(max(k, 1), n)
    order = np.argsort(-conf, kind="mergesort")
    return float(err[order][:k].mean())


def realised_risk_at_tau(conf, err, tau):
    """Realised kept-set risk on test for a threshold calibrated elsewhere: reviewed = conf < tau."""
    kept = conf >= tau
    if not kept.any():
        return 0.0, 0.0
    return float(err[kept].mean()), float(kept.mean())


def build(seed):
    z = np.load(f"outputs/killtest/seed{seed}/probs.npz")
    d = {}
    for l in ("en", "vi"):
        cd, ed = S.conf_err(z[f"{l}_dev_probs"], z[f"{l}_dev_labels"])
        ct, et = S.conf_err(z[f"{l}_test_probs"], z[f"{l}_test_labels"])
        d[l] = {"cal_conf": cd, "cal_err": ed, "test_conf": ct, "test_err": et,
                "curve": S.envelope_curve(cd, ed), "n_cal": len(cd)}
    return d


def family(d, weights=(0.5, 0.5), budget=S.C, tol_grid=NA):
    """The CAPRA family: alpha -> (s_en, s_vi, D_design, K_Q_design)."""
    rows = []
    for a in np.linspace(0.0, 1.0, tol_grid):
        sa, sb = a * budget / weights[0], (1.0 - a) * budget / weights[1]
        if sa > 1 or sb > 1:
            continue
        ra = S.risk_at(d["en"]["curve"], d["en"]["n_cal"], sa)
        rb = S.risk_at(d["vi"]["curve"], d["vi"]["n_cal"], sb)
        kq = (weights[1] * sb) / (weights[0] * sa + weights[1] * sb) if (sa + sb) > 0 else 0.0
        rows.append({"alpha": float(a), "s_en": float(sa), "s_vi": float(sb),
                     "D_design": abs(ra - rb), "KQ_design": float(kq)})
    return rows


def read_test(d, s_en, s_vi):
    """Thresholds fixed on the calibration split at the design shares, realised on test."""
    tau_en = S.tau_from_share(d["en"]["cal_conf"], s_en)
    tau_vi = S.tau_from_share(d["vi"]["cal_conf"], s_vi)
    r_en, keep_en = realised_risk_at_tau(d["en"]["test_conf"], d["en"]["test_err"], tau_en)
    r_vi, keep_vi = realised_risk_at_tau(d["vi"]["test_conf"], d["vi"]["test_err"], tau_vi)
    s_en_t, s_vi_t = 1.0 - keep_en, 1.0 - keep_vi
    return {"s_en": float(s_en), "s_vi": float(s_vi),
            "s_en_test": float(s_en_t), "s_vi_test": float(s_vi_t),
            "KQ_test": float((0.5 * s_vi_t) / max(0.5 * s_en_t + 0.5 * s_vi_t, 1e-12)),
            "D_test": float(abs(r_en - r_vi))}


def normalisations(rows):
    D = np.array([r["D_design"] for r in rows])
    Q = np.array([abs(r["KQ_design"] - 0.5) for r in rows])
    return {
        "max": (D / max(D.max(), 1e-12), Q / max(Q.max(), 1e-12)),
        "range": ((D - D.min()) / max(D.max() - D.min(), 1e-12),
                  (Q - Q.min()) / max(Q.max() - Q.min(), 1e-12)),
        "raw_pp_vs_share": (100.0 * D, Q),          # the unit clash a deployment would face
    }, D, Q


def main():
    per_seed = {}
    for seed in S.SEEDS:
        d = build(seed)
        rows = family(d)
        variants, D, Q = normalisations(rows)
        out = {"n_family": len(rows)}

        # (2) weighted sum under three normalisations
        ws = {}
        for name, (Dh, Qh) in variants.items():
            picks = []
            for lam in LAMS:
                crit = lam * Dh + (1.0 - lam) * Qh
                i = int(np.argmin(crit))
                r = dict(rows[i])
                r.update(read_test(d, r["s_en"], r["s_vi"]))
                r["lam"] = float(lam)
                picks.append(r)
            feasible = [p for p in picks if abs(p["KQ_test"] - 0.5) <= TOL]
            ws[name] = {
                "lam_feasible": [p["lam"] for p in feasible],
                "n_lam_feasible": len(feasible),
                "lam_range_feasible": [min([p["lam"] for p in feasible], default=None),
                                       max([p["lam"] for p in feasible], default=None)],
                "KQ_at_lam0": picks[0]["KQ_test"], "D_at_lam0": picks[0]["D_test"],
                "KQ_at_lam1": picks[-1]["KQ_test"], "D_at_lam1": picks[-1]["D_test"],
            }
        out["weighted_sum"] = ws
        out["lam_feasible_sets_identical_across_normalisations"] = (
            ws["max"]["lam_feasible"] == ws["range"]["lam_feasible"] == ws["raw_pp_vs_share"]["lam_feasible"])

        # (3+4) eps-constraint: one solve per tolerance level
        eps_rows = []
        for eps in EPSS:
            cand = [r for r in rows if abs(r["KQ_design"] - 0.5) <= eps + 1e-12]
            if not cand:
                continue
            best = min(cand, key=lambda r: r["D_design"])
            r = dict(best)
            r.update(read_test(d, r["s_en"], r["s_vi"]))
            r["eps"] = float(eps)
            eps_rows.append(r)
        parity_first = min(rows, key=lambda r: r["D_design"])
        comp_first = min(rows, key=lambda r: abs(r["KQ_design"] - 0.5))
        out["eps_constraint"] = {
            "n_levels": len(eps_rows),
            "at": {f"{e:.2f}": next(({"KQ_test": r["KQ_test"], "D_test": r["D_test"],
                                      "s_en": r["s_en"], "s_vi": r["s_vi"]}
                                     for r in eps_rows if abs(r["eps"] - e) < 1e-9), None)
                   for e in (0.02, 0.05, 0.10, 0.20)},
            "D_test_over_eps": [[r["eps"], r["D_test"]] for r in eps_rows],
        }
        pf = dict(parity_first); pf.update(read_test(d, pf["s_en"], pf["s_vi"]))
        cf = dict(comp_first); cf.update(read_test(d, cf["s_en"], cf["s_vi"]))
        out["parity_first"] = {"KQ_test": pf["KQ_test"], "D_test": pf["D_test"]}
        out["composition_first"] = {"KQ_test": cf["KQ_test"], "D_test": cf["D_test"]}
        per_seed[seed] = out

        print(f"seed {seed}: family={out['n_family']} | lam feasible (max/range/raw) = "
              f"{ws['max']['n_lam_feasible']}/{ws['range']['n_lam_feasible']}/"
              f"{ws['raw_pp_vs_share']['n_lam_feasible']} of {len(LAMS)}"
              f" | identical={out['lam_feasible_sets_identical_across_normalisations']}")
        for e in (0.02, 0.05, 0.10, 0.20):
            r = out["eps_constraint"]["at"][f"{e:.2f}"]
            print(f"   eps={e:.2f}: KQ_test={r['KQ_test']:.3f} D_test={100*r['D_test']:.2f} pp "
                  f"(s_en={r['s_en']:.4f}, s_vi={r['s_vi']:.4f})")
        print(f"   parity-first: KQ={pf['KQ_test']:.3f} D={100*pf['D_test']:.2f} pp | "
              f"composition-first: KQ={cf['KQ_test']:.3f} D={100*cf['D_test']:.2f} pp")

    agg = {}
    for name in ("max", "range", "raw_pp_vs_share"):
        agg[name] = {"n_lam_feasible_mean": float(np.mean([per_seed[s]["weighted_sum"][name]["n_lam_feasible"]
                                                           for s in S.SEEDS])),
                     "n_lam_feasible": [per_seed[s]["weighted_sum"][name]["n_lam_feasible"] for s in S.SEEDS]}
    for e in (0.02, 0.05, 0.10, 0.20):
        vals = [100 * per_seed[s]["eps_constraint"]["at"][f"{e:.2f}"]["D_test"] for s in S.SEEDS]
        agg[f"D_eps_{e:.2f}_pp"] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                                    "values": vals}
    agg["parity_first_D_pp"] = {"mean": float(np.mean([100 * per_seed[s]["parity_first"]["D_test"] for s in S.SEEDS])),
                                "mean_would_be_unconstrained_argmin_design": True}
    agg["composition_first_D_pp"] = {"mean": float(np.mean([100 * per_seed[s]["composition_first"]["D_test"] for s in S.SEEDS]))}
    agg["all_normalisations_agree"] = all(per_seed[s]["lam_feasible_sets_identical_across_normalisations"]
                                          for s in S.SEEDS)
    print("\nAGGREGATE", json.dumps(agg, indent=1))
    with open("outputs/r3_moo_baselines.json", "w") as fh:
        json.dump({"per_seed": per_seed, "aggregate": agg, "lams": LAMS.tolist(),
                   "epss": EPSS.tolist(), "tol": TOL}, fh, indent=1)
    print("wrote outputs/r3_moo_baselines.json")


if __name__ == "__main__":
    main()
