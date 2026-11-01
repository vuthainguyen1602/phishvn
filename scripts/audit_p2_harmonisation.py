#!/usr/bin/env python3
"""
audit_p2_harmonisation.py — what the host reduction does to each corpus of the transfer arm, and
how much the corpora share (revision 1, 2026-10-31).

WHY. The transfer arm reduces every corpus to the host before featurising (Section 4, Protocol
3). A reader cannot see from the text how many rows each corpus loses or merges on the way, how
often one host carries both labels, or whether the corpora share hosts at all, and each of these
bears on reading a transfer cell. This counts them.

Stages per external corpus: the distributed files (GramBeddings TSVs for ISCX-URL2016 and
PhishStorm, label 1 = phishing; the UCI CSV for PhiUSIIL, label 1 = legitimate), the 1:1
subsample that enters the study (`data/external/*_urllabel.csv`), and the host-level file the
features are computed from (`*_hostlabel.csv`, same rows reduced to the host). For each: rows,
class counts, unique URLs, unique hosts, unique registrable domains, rows whose host (or
registrable domain) also occurs elsewhere in the corpus, and hosts (domains) carrying both labels.
PhishVN is counted from data/processed/vn_compphish.csv at the same granularities.

Then the pairwise overlap of hosts and registrable domains between the four corpora, and the
per-feature standardised mean difference between corpora within each class, on the 13 features
that are not near-constant after the reduction.

Writes data/processed/p2/revision/harmonisation.csv, overlap.csv and feature_shift.csv.

RUN: python scripts/audit_p2_harmonisation.py
"""
from __future__ import annotations

import io
import itertools
import os
import sys
import zipfile

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
from train_url_baseline import COMPPHISH, add_label
from psl import registered_domain

EXT = "data/external"
OUT = "data/processed/p2/revision"
ALIGNED = {"PhishVN": "data/processed/vn_compphish.csv",
           "PhiUSIIL": "data/processed/external/phiusiil_compphish.csv",
           "ISCXURL2016": "data/processed/external/iscx_compphish.csv",
           "PhishStorm": "data/processed/external/phishstorm_compphish.csv"}
SUBSAMPLE = {"PhiUSIIL": "phiusiil", "ISCXURL2016": "iscx", "PhishStorm": "phishstorm"}


def host_of(u: str) -> str:
    u = str(u).strip().lower()
    u = u.split("://", 1)[1] if "://" in u else u
    return u.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].split("@")[-1].split(":")[0]


def distributed(name: str) -> pd.DataFrame:
    """The corpus as distributed, url + label in {phishing, benign}."""
    if name == "PhiUSIIL":
        with zipfile.ZipFile(os.path.join(EXT, "phiusiil.zip")) as z:
            raw = pd.read_csv(io.BytesIO(z.read("PhiUSIIL_Phishing_URL_Dataset.csv")),
                              usecols=["URL", "label"])
        return pd.DataFrame({"url": raw.URL, "label": np.where(raw.label == 1, "benign", "phishing")})
    stem = {"ISCXURL2016": "ISCXURL2016", "PhishStorm": "PhishStorm"}[name]
    # label, a tab, then the URL, which may itself contain tabs: split on the first one only
    recs = []
    for s in ("train", "test"):
        with open(os.path.join(EXT, f"{stem}_{s}.csv"), encoding="utf-8", errors="replace") as f:
            recs += [ln.rstrip("\n").split("\t", 1) for ln in f if "\t" in ln]
    raw = pd.DataFrame(recs, columns=["y", "url"])
    return pd.DataFrame({"url": raw.url, "label": np.where(raw.y.str.strip() == "1", "phishing", "benign")})


def profile(corpus: str, stage: str, df: pd.DataFrame, url_col: str = "url") -> dict:
    d = df.copy()
    d["host"] = d[url_col].map(host_of)
    d["rdom"] = d["host"].map(registered_domain)
    lab = d["label"].astype(str).str.lower()
    d["pos"] = lab.isin(["phishing", "1", "phish", "malicious"])
    row = {"corpus": corpus, "stage": stage, "rows": len(d), "phishing": int(d.pos.sum()),
           "benign": int((~d.pos).sum()), "unique_urls": d[url_col].nunique(),
           "unique_hosts": d.host.nunique(), "unique_rdoms": d.rdom.nunique()}
    for key in ("host", "rdom"):
        g = d.groupby(key).pos.agg(["size", "nunique"])
        row[f"rows_sharing_{key}"] = int(g.loc[g["size"] > 1, "size"].sum())
        row[f"{key}s_both_labels"] = int((g["nunique"] > 1).sum())
        row[f"rows_on_{key}s_both_labels"] = int(g.loc[g["nunique"] > 1, "size"].sum())
    return row


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows, hosts = [], {}
    for name, stem in SUBSAMPLE.items():
        rows.append(profile(name, "distributed", distributed(name)))
        rows.append(profile(name, "subsample", pd.read_csv(os.path.join(EXT, f"{stem}_urllabel.csv"))))
        rows.append(profile(name, "host_level", pd.read_csv(os.path.join(EXT, f"{stem}_hostlabel.csv"))))
    aligned = {}
    for name, path in ALIGNED.items():
        d = add_label(pd.read_csv(path))
        d["label"] = np.where(d.y == 1, "phishing", "benign")
        aligned[name] = d
        rows.append(profile(name, "featurised", d))
        hosts[name] = set(d.url.map(host_of))
    prof = pd.DataFrame(rows)
    prof.to_csv(os.path.join(OUT, "harmonisation.csv"), index=False)

    over = []
    rd = {k: {registered_domain(h) for h in v} for k, v in hosts.items()}
    for a, b in itertools.combinations(ALIGNED, 2):
        over.append({"a": a, "b": b, "shared_hosts": len(hosts[a] & hosts[b]),
                     "shared_rdoms": len(rd[a] & rd[b]),
                     "rdoms_a": len(rd[a]), "rdoms_b": len(rd[b])})
    pd.DataFrame(over).to_csv(os.path.join(OUT, "overlap.csv"), index=False)

    feats = [c for c in COMPPHISH if all(c in d for d in aligned.values())]
    active = [c for c in feats
              if all((pd.to_numeric(d[c], errors="coerce").fillna(0) != 0).mean() > 0.01
                     for d in aligned.values())]
    shift = []
    for a, b in itertools.permutations(ALIGNED, 2):
        for cls in (1, 0):
            xa = aligned[a].loc[aligned[a].y == cls, active].apply(pd.to_numeric, errors="coerce")
            xb = aligned[b].loc[aligned[b].y == cls, active].apply(pd.to_numeric, errors="coerce")
            sd = np.sqrt((xa.var() + xb.var()) / 2).replace(0, np.nan)
            smd = ((xb.mean() - xa.mean()) / sd).abs()
            for f in active:
                shift.append({"source": a, "target": b, "class": "phishing" if cls else "benign",
                              "feature": f, "abs_smd": float(smd[f])})
    pd.DataFrame(shift).to_csv(os.path.join(OUT, "feature_shift.csv"), index=False)

    pd.set_option("display.width", 220)
    print(prof.to_string(index=False))
    print(pd.DataFrame(over).to_string(index=False))
    s = pd.DataFrame(shift)
    print(f"\nactive features ({len(active)}): {', '.join(active)}")
    print(s.groupby(["source", "target", "class"]).abs_smd.agg(["mean", "max"]).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
