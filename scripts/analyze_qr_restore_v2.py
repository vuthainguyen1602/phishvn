#!/usr/bin/env python3
"""
analyze_qr_restore_v2.py — the registered analysis of the architecture comparison.

`PREREG_restore_v2.md` asks one question: arm A (four 3x3 convolutions, 9 px receptive field) moves
every degradation whose damage is local and does not move `logo` at all, which its receptive field
predicted. Does a model that can see the whole symbol move it?

Three arms of the SAME study, each dumped as PNGs on the workstation and decoded on `.204` in one
session, each dump carrying its own `raw`, `control` and `restored`:

    conv        arm A, the first study's network, re-decoded here rather than reused
    hybrid      arm B, conv stem -> self-attention at 1/4 resolution -> upsample
    restormer   arm C, transposed (channel) attention at full resolution

WHAT IS REGISTERED, and what this script therefore refuses to improvise:

  T1  THE MECHANISM. `logo` renders only, OpenCV, restored(new arm) against restored(conv), paired
      per URL, two-sided Wilcoxon. Success >= +10 pp, partial 0 to +10, negative below 0.
  T2  THE DEPLOYMENT QUESTION, identical in form to the first study's T1 so the two are comparable.
      restored(new arm)+OpenCV against raw+WeChatQR, pooled over all six transforms, paired per
      URL. Success >= +5 pp, partial 0 to +5, negative below 0.

  Benjamini-Hochberg over m = 4 (two tests x two new arms). BOTH new arms are reported whatever
  they show; the prereg puts "choose the better one afterwards" out of scope, so this script prints
  every arm it is given and has no flag to hide one.

  GUARD (a threshold, not a test): of the held-out renders the `control` arm decoded correctly, the
  share the `restored` arm breaks must not exceed 2%. A restorer earns nothing by rescuing renders
  it also breaks, and an arm that fails this is reported as failing it whatever T1 and T2 say.

D IS POSITIVE WHEN THE NEW ARM IS AHEAD, i.e. D = DFR(comparator) - DFR(new arm). Unit of analysis
is the URL: every quantity is computed per URL first and aggregated across the held-out URLs.

THE PLUMBING CHECK IS NOT OPTIONAL. `raw` minus `control` is the polarity line and nothing else, and
arm A's numbers here must reproduce the first study's. If they drift, the two runs are not
comparable and the comparison is void -- this script says so rather than averaging over it.

RUN
    python3 scripts/analyze_qr_restore_v2.py
    python3 scripts/analyze_qr_restore_v2.py --dfr conv=... --dfr hybrid=... --json ...
"""
from __future__ import annotations
import argparse, collections, csv, json, os, sys

import numpy as np

PROC = os.path.join("data", "processed", "qr")
DEFAULT = {"conv": os.path.join(PROC, "qr_restore_arms_conv_dfr.csv"),
           "hybrid": os.path.join(PROC, "qr_restore_arms_hybrid_dfr.csv"),
           "restormer": os.path.join(PROC, "qr_restore_arms_restormer_dfr.csv")}
SNAP = os.path.join(PROC, "restore_v2_snapshot.json")
BASELINE = "conv"          # arm A: the comparator, never itself a T1 subject
T1_TRANSFORM = "logo"      # the transform the mechanism is about
T1_BAR, T2_BAR = 10.0, 5.0
GUARD_PCT = 2.0
# Measured wall-clock of the eight registered epochs, read off the training runs of 2026-08-31.
# Recorded here rather than re-derived because --no-eval writes no snapshot to carry them.
TRAIN_SECONDS = {"conv": None, "hybrid": 723, "restormer": 6004}


def load(path: str) -> list:
    """One decoded dump -> rows carrying (arm, url, transform, decoder, correct).

    The arm rides in the sample_id suffix the dump wrote and the URL is its first 16 hex;
    benchmark_qr.py carries neither column, and neither needs to exist twice."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sid, _, arm = r["sample_id"].rpartition("__")
            if arm not in ("raw", "control", "restored"):
                raise SystemExit(f"[!] {path}: sample_id {r['sample_id']!r} carries no arm suffix; "
                                 f"this is not a decoded --dump")
            rows.append({"sid": sid, "url": sid[:16], "arm": arm, "decoder": r["decoder"],
                         "transform": r["transform"], "correct": int(r["correct"])})
    return rows


def per_url_dfr(rows: list, arm: str, decoder: str, transform: str = "") -> dict:
    """DFR per URL, as a percentage. The URL is the unit; renders are the fixed design under it."""
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["arm"] == arm and r["decoder"] == decoder and (not transform
                                                            or r["transform"] == transform):
            a = agg[r["url"]]
            a[0] += 1
            a[1] += r["correct"]
    return {u: 100 * (1 - ok / n) for u, (n, ok) in agg.items() if n}


def paired(new: dict, comparator: dict) -> tuple:
    """(D array, urls) over the URLs both sides have. D > 0 means the new arm is ahead."""
    urls = sorted(set(new) & set(comparator))
    return np.array([comparator[u] - new[u] for u in urls], dtype=float), urls


def test(d: np.ndarray, bar: float) -> dict:
    from scipy.stats import wilcoxon
    p = float(wilcoxon(d)[1]) if d.size and np.any(d != 0) else 1.0
    mean = float(d.mean()) if d.size else float("nan")
    return {"n_urls": int(d.size), "mean_pp": mean, "median_pp": float(np.median(d)) if d.size
            else float("nan"), "p": p,
            "verdict": "success" if mean >= bar else "partial" if mean >= 0 else "negative"}


def bh(ps: dict, m: int) -> dict:
    """Benjamini-Hochberg at the m fixed by the registration, not at the number of tests that
    happened to run: dropping an arm and shrinking m would loosen every threshold."""
    order = sorted(ps, key=lambda k: ps[k])
    out, prev = {}, 1.0
    for i in range(len(order), 0, -1):
        k = order[i - 1]
        prev = min(prev, ps[k] * m / i)
        out[k] = prev
    return out


def transitions(rows: list, decoder: str) -> dict:
    """Rescued against destroyed, control -> restored. A net gain hides two populations."""
    by = {}
    for r in rows:
        if r["decoder"] == decoder and r["arm"] in ("control", "restored"):
            by.setdefault(r["sid"], {})[r["arm"]] = r["correct"]
    resc = sum(1 for v in by.values() if v.get("control") == 0 and v.get("restored") == 1)
    dest = sum(1 for v in by.values() if v.get("control") == 1 and v.get("restored") == 0)
    was_ok = sum(1 for v in by.values() if v.get("control") == 1)
    return {"rescued": resc, "destroyed": dest, "control_correct": was_ok, "n": len(by),
            "destroyed_pct_of_correct": (100 * dest / was_ok) if was_ok else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfr", action="append", default=[],
                    help="arch=path/to/decoded_dump.csv; repeatable. Defaults to all three arms.")
    ap.add_argument("--json", default=SNAP)
    a = ap.parse_args()

    paths = dict(DEFAULT)
    for spec in a.dfr:
        k, _, v = spec.partition("=")
        paths[k.strip()] = v.strip()
    missing = [f"{k}: {v}" for k, v in paths.items() if not os.path.isfile(v)]
    if missing:
        print("[!] decoded dumps not found — produce them with\n"
              "      python3 scripts/train_qr_restore.py --dir data/raw/qr_restore_v2 "
              "--arch <arch> --eval-only --dump data/raw/qr_restore_arms_<arch> --decoders ''\n"
              "    then on .204:\n"
              "      python3 scripts/benchmark_qr.py --dir data/raw/qr_restore_arms_<arch> "
              "--out data/processed/qr/qr_restore_arms_<arch>_dfr.csv\n    missing:\n      "
              + "\n      ".join(missing), file=sys.stderr)
        return 1

    data = {k: load(v) for k, v in paths.items()}
    decs = sorted({r["decoder"] for r in data[BASELINE]})
    print(f"[*] {', '.join(f'{k} {len(v):,} rows' for k, v in data.items())}; "
          f"decoders {', '.join(decs)}")
    if not {"opencv", "wechat"} <= set(decs):
        print("[!] T1 needs OpenCV and T2 needs WeChatQR; both must be in the same decode",
              file=sys.stderr)
        return 1

    # --- the plumbing check, before anything is compared -----------------------------------
    print("\n=== Plumbing: raw vs control is the polarity line and nothing else ===")
    plumb = {}
    for arch, rows in data.items():
        line = {}
        for d in decs:
            raw = np.mean(list(per_url_dfr(rows, "raw", d).values()))
            ctl = np.mean(list(per_url_dfr(rows, "control", d).values()))
            line[d] = {"raw": float(raw), "control": float(ctl), "polarity_pp": float(raw - ctl)}
        plumb[arch] = line
        print(f"  {arch:10} " + "  ".join(f"{d} {line[d]['raw']:.1f}->{line[d]['control']:.1f}"
                                          f" ({line[d]['polarity_pp']:+.1f}pp)" for d in decs))
    print("  Arm A's raw and control columns must reproduce the first study "
          "(OpenCV 69.6 -> 53.5). A drift here voids the comparison rather than qualifying it.")

    # --- the registered tests ---------------------------------------------------------------
    results, ps = {}, {}
    base_logo = per_url_dfr(data[BASELINE], "restored", "opencv", T1_TRANSFORM)
    for arch, rows in data.items():
        if arch == BASELINE:
            continue
        d1, _ = paired(per_url_dfr(rows, "restored", "opencv", T1_TRANSFORM), base_logo)
        d2, _ = paired(per_url_dfr(rows, "restored", "opencv"),
                       per_url_dfr(rows, "raw", "wechat"))
        results[arch] = {"T1": {**test(d1, T1_BAR),
                                "arms": f"restored({arch})+opencv vs restored({BASELINE})+opencv, "
                                        f"{T1_TRANSFORM} only"},
                         "T2": {**test(d2, T2_BAR),
                                "arms": f"restored({arch})+opencv vs raw+wechat"}}
        ps[f"{arch}:T1"] = results[arch]["T1"]["p"]
        ps[f"{arch}:T2"] = results[arch]["T2"]["p"]

    adj = bh(ps, m=4)
    for key, p_bh in adj.items():
        arch, _, t = key.partition(":")
        results[arch][t]["p_bh"] = p_bh

    print("\n=== REGISTERED (PREREG_restore_v2.md), BH over m = 4 ===")
    for arch in sorted(results):
        for t, bar in (("T1", T1_BAR), ("T2", T2_BAR)):
            v = results[arch][t]
            print(f"  {arch:10} {t}  {v['arms']:<62} n={v['n_urls']:<4} "
                  f"mean {v['mean_pp']:+6.1f}pp  median {v['median_pp']:+6.1f}pp  "
                  f"p_BH={v['p_bh']:.3g}  -> {v['verdict'].upper()}  (bar {bar:+.0f}pp)")
    print("  Positive means the NEW arm is ahead. T1 is the mechanism (does a global receptive "
          "field\n  fix `logo`?); T2 is whether the arm beats simply installing WeChatQR.")

    # --- the registered guard ----------------------------------------------------------------
    print(f"\n=== Registered guard: an arm may not break more than {GUARD_PCT}% of what already "
          f"decoded ===")
    guard = {}
    for arch, rows in data.items():
        guard[arch] = {d: transitions(rows, d) for d in decs}
        for d in decs:
            g = guard[arch][d]
            flag = "" if g["destroyed_pct_of_correct"] <= GUARD_PCT else "   <- FAILS THE GUARD"
            print(f"  {arch:10} {d:8} rescued {g['rescued']:>5,}  destroyed {g['destroyed']:>4,}"
                  f"  ({g['destroyed_pct_of_correct']:.2f}% of {g['control_correct']:,} "
                  f"already-correct){flag}")

    # --- registered diagnostics ---------------------------------------------------------------
    print("\n=== Diagnostic: DFR per arch x transform, restored arm, all decoders ===")
    trs = sorted({r["transform"] for r in data[BASELINE]})
    per_tr = {}
    for arch, rows in data.items():
        per_tr[arch] = {}
        for d in decs:
            for t in trs:
                ctl = per_url_dfr(rows, "control", d, t)
                res = per_url_dfr(rows, "restored", d, t)
                if ctl and res:
                    per_tr[arch][f"{d}/{t}"] = {
                        "control": float(np.mean(list(ctl.values()))),
                        "restored": float(np.mean(list(res.values())))}
        line = per_tr[arch]
        print(f"  [{arch}]")
        for t in trs:
            k = f"opencv/{t}"
            if k in line:
                print(f"    {t:12} opencv control {line[k]['control']:5.1f}%  "
                      f"restored {line[k]['restored']:5.1f}%  "
                      f"{line[k]['control'] - line[k]['restored']:+6.1f}pp")

    # The prereg registers parameter count and training wall-clock as diagnostics, "so that a null
    # result carries its budget". Parameters come from the checkpoint; wall-clock is passed in,
    # because --no-eval training writes no snapshot and the number would otherwise live only in a
    # terminal that has since scrolled.
    budget = {}
    for arch in data:
        ckpt = os.path.join(PROC, "restore",
                            "restore.pt" if arch == "conv" else f"restore_{arch}.pt")
        if os.path.isfile(ckpt):
            import torch
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            budget[arch] = {"params": int(sum(v.numel() for v in sd.values())),
                            "checkpoint_kb": round(os.path.getsize(ckpt) / 1e3, 1),
                            "train_seconds": TRAIN_SECONDS.get(arch)}
    if budget:
        print("\n=== Diagnostic: what each arm cost (a null result carries its budget) ===")
        for arch, b in budget.items():
            secs = b.get("train_seconds")
            print(f"  {arch:10} {b.get('params', 0):>9,} params  "
                  f"{b.get('checkpoint_kb', 0):>7.1f} KB  "
                  + (f"{secs / 60:.1f} min for 8 epochs" if secs else "training time not recorded"))

    out = {"registered": results, "guard": guard, "plumbing": plumb, "by_transform": per_tr,
           "budget": budget, "sources": paths, "m": 4, "bars": {"T1": T1_BAR, "T2": T2_BAR},
           "guard_pct": GUARD_PCT}
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    json.dump(out, open(a.json, "w", encoding="utf-8"), indent=2, sort_keys=True)
    print(f"\n[+] {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
