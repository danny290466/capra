#!/usr/bin/env python3
"""CAPRA operational simulation: a time-varying arrival mix and a live review budget.

Answers the external-review question "does the framework still hold when traffic and the budget are
live": the manuscript's operating points are calibrated on a split and their realised load on new
traffic is only reported. Here the same policies run over a stream of windows whose language mix
drifts, under two regimes:

  fixed  -- the manuscript's regime: thresholds from the calibration split, load free to drift.
  capped -- the budget enforced inside every window: at most floor(C * n_window) items are reviewed,
            the most suspicious candidates first (lowest confidence), the rest are auto-decided.

Conventions follow the manuscript exactly:
  p1   = P(toxic), pred = 1[p1 >= 0.5], conf = max(p1, 1-p1)   (selective-prediction confidence)
  err  = 1[pred != gold]                                       (0/1 error of a kept item)
  review share s = P(conf < tau);  risk(s) = mean err over {conf >= tau}, the kept set.
  The reported curve is the achievable envelope (running minimum from the low-review end, Appendix B).
Frozen inputs: outputs/killtest/seed{42..46}/probs.npz (EN = Civil Comments test, VI = ViHSD test).
"""
import argparse
import glob
import json
import os

import numpy as np

SEEDS = ["42", "43", "44", "45", "46"]
C = 0.05                 # review capacity, share of mixed traffic
W = (0.5, 0.5)           # traffic weights, equal by default
N_WINDOW = 200
N_WINDOWS = 60
N_SCHED = 100            # arrival schedules per training seed
RHO = (1.0, 0.95, 0.90)  # reviewer accuracy: reviewed items are corrected with probability rho


def conf_err(probs, labels):
    p1 = probs[:, 1]
    pred = (p1 >= 0.5).astype(int)
    return np.maximum(p1, 1.0 - p1), (pred != labels).astype(float)


def envelope_curve(conf, err):
    """risk(k) for the top-k most confident items, with the running-minimum envelope."""
    order = np.argsort(-conf, kind="mergesort")
    e = err[order]
    k = np.arange(1, len(e) + 1)
    raw = np.cumsum(e) / k
    return np.minimum.accumulate(raw)          # envelope from the low-review end


def risk_at(curve, n, s):
    """risk at review share s (curve = running-min risk of the top-k kept set)."""
    k = int(round(n * (1.0 - s)))
    k = min(max(k, 1), n)
    return float(curve[k - 1])


def tau_from_share(cal_conf, s):
    return float(np.quantile(cal_conf, min(max(s, 0.0), 1.0), method="lower"))


def pooled_tau(cal, weights, target):
    """Single threshold on the traffic-weighted confidence mix, calibrated to `target` load."""
    conf = np.concatenate([cal[l]["conf"] for l in ("en", "vi")])
    wt = np.concatenate([np.full(len(cal[l]["conf"]), weights[i] / len(cal[l]["conf"]))
                         for i, l in enumerate(("en", "vi"))])
    order = np.argsort(conf, kind="mergesort")
    c, w = conf[order], wt[order]
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, target * cum[-1]))
    return float(c[min(idx, len(c) - 1)])


def sweep_parity(cal, curves, budget, weights):
    """Phase A: alpha in [0,1] = share of the budget spent on the reference language (equal costs)."""
    alphas = np.linspace(0.0, 1.0, 2001)
    best, best_obj, best_s = None, np.inf, None
    for a in alphas:
        sa, sb = a * budget / weights[0], (1 - a) * budget / weights[1]
        if sa > 1 or sb > 1:
            continue
        ra = risk_at(curves["en"], len(cal["en"]["conf"]), sa)
        rb = risk_at(curves["vi"], len(cal["vi"]["conf"]), sb)
        d = abs(ra - rb)
        if d < best_obj:
            best, best_obj, best_s = a, d, (sa, sb)
    assert best_s is not None, "no feasible alpha on the sweep grid"
    return best_s


def sweep_composition(cal, curves, budget, weights, target):
    alphas = np.linspace(0.0, 1.0, 2001)
    best, best_obj, best_s = None, np.inf, None
    for a in alphas:
        sa, sb = a * budget / weights[0], (1 - a) * budget / weights[1]
        if sa > 1 or sb > 1:
            continue
        kq = (weights[1] * sb) / (weights[0] * sa + weights[1] * sb) if (sa + sb) > 0 else 0.0
        d = abs(kq - target)
        if d < best_obj:
            best, best_obj, best_s = a, d, (sa, sb)
    assert best_s is not None, "no feasible alpha on the sweep grid"
    return best_s


def policies(cal, curves):
    """The four policies of the manuscript, each as {lang: tau} selected on the calibration split."""
    s_pooled = pooled_tau(cal, W, C)
    out = {"GLOBAL": {"en": s_pooled, "vi": s_pooled}}
    out["EQUAL-COV"] = {l: tau_from_share(cal[l]["conf"], C) for l in ("en", "vi")}
    for name, s in (("CAPRA-parity", sweep_parity(cal, curves, C, W)),
                    ("CAPRA-composition", sweep_composition(cal, curves, C, W, W[1]))):
        out[name] = {"en": tau_from_share(cal["en"]["conf"], s[0]),
                     "vi": tau_from_share(cal["vi"]["conf"], s[1])}
    return out


def schedule_mix(rng, n_windows, amp, burst):
    """Drifting traffic mix with short bursts; returns w_vi per window."""
    t = np.arange(n_windows)
    base = 0.5 + amp * np.sin(2 * np.pi * t / n_windows + rng.uniform(0, 2 * np.pi))
    extra = np.zeros(n_windows)
    if burst > 0 and n_windows >= 4:
        span = max(1, n_windows // 10)
        n_burst = min(2, max(1, n_windows // 3))
        starts = rng.choice(np.arange(1, max(2, n_windows - span)), size=n_burst, replace=False)
        for start in starts:
            extra[start:start + span] += rng.uniform(0.4 * burst, burst)
    return np.clip(base + extra, 0.15, 0.85)


def run_stream(seed_data, taus, mix, rng, regime, n_window):
    """Route a stream of windows; return per-item arrays, the window load and the reviewed mask."""
    langs, confs, errs, win_load, rev = [], [], [], [], []
    for w_vi in mix:
        n = n_window
        n_vi = int(round(n * w_vi)); n_en = n - n_vi
        idx_en = rng.integers(0, len(seed_data["en"]["conf"]), n_en)
        idx_vi = rng.integers(0, len(seed_data["vi"]["conf"]), n_vi)
        conf = np.concatenate([seed_data["en"]["conf"][idx_en], seed_data["vi"]["conf"][idx_vi]])
        err = np.concatenate([seed_data["en"]["err"][idx_en], seed_data["vi"]["err"][idx_vi]])
        lang = np.concatenate([np.zeros(n_en, dtype=int), np.ones(n_vi, dtype=int)])
        tau = np.where(lang == 0, taus["en"], taus["vi"])
        cand = conf < tau
        if regime == "capped":
            cap = int(np.floor(C * n))
            n_cand = int(cand.sum())
            if n_cand > cap:                       # most suspicious first
                order = np.argsort(np.where(cand, conf, np.inf), kind="mergesort")
                admitted = np.zeros(n, dtype=bool)
                admitted[order[:cap]] = True
                cand = admitted
        langs.append(lang); confs.append(conf); errs.append(err)
        win_load.append(cand.mean()); rev.append(cand)
    return (np.concatenate(langs), np.concatenate(confs), np.concatenate(errs),
            np.array(win_load), np.concatenate(rev))


def summarise(lang, err, reviewed, win_load):
    kept = ~reviewed
    out = {"load": float(reviewed.mean()), "max_window_load": float(win_load.max()),
           "frac_windows_over": float((win_load > C + 1e-9).mean()),
           "mean_overshoot_pp": float(100 * np.mean(np.clip(win_load - C, 0, None)[win_load > C + 1e-9]))
           if (win_load > C + 1e-9).any() else 0.0}
    for i, l in enumerate(("en", "vi")):
        out[f"load_{l}"] = float(reviewed[lang == i].mean())
    r = {l: float(err[(lang == i) & kept].mean()) for i, l in enumerate(("en", "vi"))}
    out["risk_en_pp"], out["risk_vi_pp"] = 100 * r["en"], 100 * r["vi"]
    # manuscript convention (Section 3.1 / Table 10): the average residual risk of the kept set is the
    # traffic-weighted average of the per-language kept-set risks, not the item-pooled rate.
    out["kept_risk_pp"] = 100 * (W[0] * r["en"] + W[1] * r["vi"])
    out["kept_risk_pooled_pp"] = 100 * float(err[kept].mean())
    out["disparity_pp"] = 100 * abs(r["en"] - r["vi"])
    out["kq_vi"] = float((reviewed & (lang == 1)).sum() / max(reviewed.sum(), 1))
    for rho in RHO:
        out[f"error_pp_rho{rho}"] = 100 * float((err[kept].sum() + (1 - rho) * err[reviewed].sum())
                                                / len(err))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/killtest")
    ap.add_argument("--out", default="outputs/operational_sim.json")
    ap.add_argument("--sched", type=int, default=N_SCHED)
    ap.add_argument("--amp", type=float, default=0.18)
    ap.add_argument("--burst", type=float, default=0.25)
    ap.add_argument("--window", type=int, default=N_WINDOW)
    ap.add_argument("--windows", type=int, default=N_WINDOWS)
    args = ap.parse_args()
    n_sched, n_window, n_windows = args.sched, args.window, args.windows
    mix_lo, mix_hi = 1.0, 0.0

    acc = {}
    for si, seed in enumerate(SEEDS):
        path = os.path.join(args.dir, f"seed{seed}", "probs.npz")
        z = np.load(path)
        data = {}
        for l in ("en", "vi"):
            cd, ed = conf_err(z[f"{l}_dev_probs"], z[f"{l}_dev_labels"])
            ct, et = conf_err(z[f"{l}_test_probs"], z[f"{l}_test_labels"])
            data[l] = {"conf": ct, "err": et,
                       "curve": envelope_curve(cd, ed), "cal": {"conf": cd, "err": ed}}
        cal = {l: data[l]["cal"] for l in ("en", "vi")}
        curves = {l: data[l]["curve"] for l in ("en", "vi")}
        taus = policies(cal, curves)

        for regime in ("fixed", "capped"):
            for name, t in taus.items():
                key = f"{name}|{regime}"
                acc.setdefault(key, [])
                for k in range(n_sched):
                    rng = np.random.default_rng(7000 + 13 * si + k)
                    mix = schedule_mix(rng, n_windows, args.amp, args.burst)
                    mix_lo, mix_hi = min(mix_lo, mix.min()), max(mix_hi, mix.max())
                    lang, conf, err, wl, reviewed = run_stream(data, t, mix, rng, regime, n_window)
                    acc[key].append(summarise(lang, err, reviewed, wl))
        print(f"seed {seed}: thresholds " +
              ", ".join(f"{n}=(en {t['en']:.3f}, vi {t['vi']:.3f})" for n, t in taus.items()))

    # run_stream already enforces the per-window cap in the "capped" regime; the mask it returns
    # is the routing that actually happened, so no mask is rebuilt here.
    report = {}
    for key, rows in acc.items():
        agg = {}
        for metric in rows[0]:
            v = np.array([r[metric] for r in rows])
            agg[metric] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1))}
        report[key] = agg

    # PAIRED dispersion of the price of enforcement (round-2 external review, item 4). fixed and capped
    # are run from the same rng draw for the same (seed, schedule), so the two lists line up entry by
    # entry and the difference can be reported with its own SD and interval instead of as the gap between
    # two marginal means. The reviewer asked for exactly this on the CAPRA-parity disparity change.
    paired = {}
    for name in ("GLOBAL", "EQUAL-COV", "CAPRA-parity", "CAPRA-composition"):
        fx, cp = acc[f"{name}|fixed"], acc[f"{name}|capped"]
        assert len(fx) == len(cp), (name, len(fx), len(cp))
        blk = {}
        for metric in ("load", "max_window_load", "mean_overshoot_pp", "kept_risk_pp", "disparity_pp",
                       "kq_vi", "error_pp_rho1.0"):
            d = np.array([c[metric] - f[metric] for f, c in zip(fx, cp)])
            blk[metric] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                           "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        blk["paired_runs"] = len(fx)
        paired[name] = blk
    report["_paired_capped_minus_fixed"] = paired

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    report["_mix"] = {"min": mix_lo, "max": mix_hi, "amp": args.amp, "burst": args.burst,
                      "n_windows": n_windows, "window": n_window, "schedules": n_sched,
                      "seeds": SEEDS, "C": C}
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)

    cols = ("load", "max_window_load", "frac_windows_over", "mean_overshoot_pp", "kept_risk_pp",
            "disparity_pp", "kq_vi", "error_pp_rho1.0", "error_pp_rho0.95", "error_pp_rho0.9")
    print("policy|regime|" + "|".join(cols))
    for key in sorted(report):
        if key.startswith("_"):
            continue
        name, regime = key.split("|")
        a = report[key]
        print(f"{name}|{regime}|" + "|".join(f"{a[m]['mean']:.2f}" for m in cols))
    print("mix range %.2f-%.2f | window %d x %d | %d schedules x %d seeds"
          % (mix_lo, mix_hi, n_window, n_windows, n_sched, len(SEEDS)))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
