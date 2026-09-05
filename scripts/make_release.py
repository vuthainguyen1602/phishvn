#!/usr/bin/env python3
"""
make_release.py — Package a citable PhishVN release for a DOI repository (Mendeley Data / Zenodo).

Produces two clearly separated bundles under release/:
  * PhishVN_v<ver>_open.zip   — OPEN tier (CC BY 4.0): URL table, features, labels, splits, docs,
                                 LICENSE, CITATION.cff, an auto-generated README + MANIFEST (SHA-256).
                                 This is what you upload for open access and cite in the paper.
  * PhishVN_v<ver>_pages_GATED.zip (only with --include-pages) — the captured phishing HTML +
                                 screenshots. GATED / on-request only; contains live malicious page
                                 captures. Do NOT upload to the open tier.

RUN:
  python scripts/make_release.py --version 1.0.0
  python scripts/make_release.py --version 1.0.0 --include-pages
"""
from __future__ import annotations
import argparse
import csv
import datetime as _dt
import hashlib
import os
import re
import shutil
import sys
import zipfile
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

OPEN_FILES = [
    ("data/processed/dataset_url.csv", "data/dataset_url.csv"),
    ("data/processed/vn_compphish.csv", "data/vn_compphish.csv"),
    ("data/processed/splits/url_train.csv", "data/splits/url_train.csv"),
    ("data/processed/splits/url_val.csv", "data/splits/url_val.csv"),
    ("data/processed/splits/url_test.csv", "data/splits/url_test.csv"),
    # An ethics statement that argues release is the safe direction -- because a defender can
    # re-run our detector against our own attack set -- is only as good as the artefact actually
    # shipping. Both files are guardrail-checked before they are packed.
    ("data/processed/p3/p3_paraphrase.csv", "data/attacks/p3_paraphrase.csv"),
    ("data/processed/p3/p3_paraphrase_band.csv", "data/attacks/p3_paraphrase_band.csv"),
    # Typed where a conservative rule can type it; the datasheet explains why 89% is `unknown`.
    ("data/processed/abuse_type.csv", "data/abuse_type.csv"),
    ("data/docs/datasheet.md", "docs/datasheet.md"),
    ("data/docs/schema.md", "docs/schema.md"),
    ("data/docs/data_sources.md", "docs/data_sources.md"),
    ("LICENSE", "LICENSE"),
    ("CITATION.cff", "CITATION.cff"),
]


# ---- P4b: the infrastructure data article's deposit ---------------------------------------
# The article's "Repository structure" table IS the contract for this list: build_p4b refuses to
# run when the two disagree on a row count, because a deposit and the article describing it are
# supposed to be the same object. v1.0.0 freezes at P4's trigger; until then the version carries
# a -draft suffix so nothing built here can be mistaken for the deposit.
INFRA_FILES = [
    ("data/raw/host_infra/host_infra.csv", "data/host_infra.csv"),
    ("data/processed/infra/infra_dataset.csv", "data/infra_dataset.csv"),
    ("data/processed/infra/funnel.csv", "data/funnel.csv"),
    ("data/processed/infra/accrual.csv", "data/accrual.csv"),
    ("data/processed/infra/label_audit.csv", "data/label_audit.csv"),
    ("data/processed/infra/wildcard_probe.csv", "data/wildcard_probe.csv"),
    ("data/raw/ct_benign/seen_domains.txt", "data/ct_benign_seen.txt"),
    ("data/raw/ct_benign_vn/seen_domains.txt", "data/ct_benign_vn_seen.txt"),
    ("data/docs/infra/README_infra.md", "README.md"),
    ("data/docs/infra/CITATION_infra.cff", "CITATION.cff"),
    ("data/docs/infra/schema_infra.md", "docs/schema.md"),
    ("data/docs/infra/collection_protocol.md", "docs/collection_protocol.md"),
    ("data/docs/infra/datasheet_infra.md", "docs/datasheet.md"),
    ("data/docs/infra/CHANGELOG_infra.md", "docs/CHANGELOG.md"),
    ("LICENSE", "LICENSE"),
]
# What a FREEZE build must still do, not what a draft build lacks. The datasheet and changelog
# were absent until 2026-08-29 and the list named the files; now they ship in the draft too, and
# what remains at freeze is the work of making them true of the frozen cut. A reader of a draft
# deposit is better served by a datasheet that is current-but-provisional than by a promise.
INFRA_AT_FREEZE = ["refresh every count in docs/datasheet.md and docs/CHANGELOG.md",
                   "restate the capture window", "record collection changes made since"]
# The ethics statement promises records ABOUT pages, never page content. Anything that could
# carry markup or a rendered capture is a build-stopper, checked by column name and by content.
INFRA_FORBIDDEN_COLS = ("html", "body", "screenshot", "dom", "content", "raw", "text")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_stats(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lab = Counter(r.get("label", "") for r in rows)
    tier = Counter(r.get("tier", "") for r in rows)
    split = Counter(r.get("split", "") for r in rows)
    scen = Counter(r.get("scenario", "") for r in rows)
    return len(rows), lab, tier, split, scen


def _readme(version, stats):
    n, lab, tier, split, scen = stats
    top_scen = ", ".join(f"{k}={v}" for k, v in scen.most_common())
    return f"""# PhishVN v{version} — Vietnamese URL Phishing Dataset (open tier)

Time-stamped Vietnamese URL phishing dataset with a CompPhish-aligned 21-feature lexical schema,
impersonation-scenario labels, gold/silver label-confidence tiers, and a group-aware temporal split.
Released {_dt.date.today().isoformat()} under CC BY 4.0 (see LICENSE). Cite via CITATION.cff / the DOI.

## Contents
- `data/dataset_url.csv` — full record table ({n} rows).
- `data/vn_compphish.csv` — the same URLs re-featurised into the exact CompPhish schema.
- `data/splits/url_{{train,val,test}}.csv` — the official group-aware temporal split.
- `data/attacks/p3_paraphrase{{,_band}}.csv` — the 386 paraphrased lures of the evasion study and
  their Jaccard-controlled counterparts. Every lure carries the simulated link
  `http://sim.example.vn/x` and no other URL, contains no real brand token, and is stripped of
  diacritics: the set is for re-running a detector against a published attack, and resolves
  nowhere.
- `data/abuse_type.csv` — what KIND of abuse each listed domain is, where a conservative rule can
  say; 89% is `unknown` by design. See the datasheet on what the positive label means.
- `docs/` — datasheet, column schema, and data-source notes.
- `MANIFEST.txt` — SHA-256 checksums and row counts for every file.

## Composition
- Labels: {dict(lab)}
- Tiers: {dict(tier)}
- Splits: {dict(split)}
- Scenarios: {top_scen}

## Notes
- Benign = NCSC trusted-organisation registry (easy) + Tranco top-list sample (hard negatives).
- `is_https` is present for CompPhish compatibility but is a collection artefact — exclude it from
  modelling (all lexical features are computed on the scheme-stripped URL).
- The captured phishing HTML/screenshot subset is distributed separately as a GATED bundle
  (research-only; live malicious captures) — not part of this open tier.
- PII is redacted; no id↔PII mapping is included. Complies with Decree 13/2023/ND-CP and Vietnam's PDPL.
"""


def write_bundle(zip_path, files, extra_texts):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in files:
            z.write(src, arc)
        for arc, text in extra_texts.items():
            z.writestr(arc, text)


def _check_attack_guardrails():
    """The attack set ships only if every lure is still defused. This is the last gate before the
    bundle leaves the machine, so it re-checks rather than trusting the build that wrote the CSVs:
    the rules (simulated link only, no real brand token, no diacritics) are what make publishing an
    evasion corpus defensible, and the README about to be written asserts them."""
    from p3_jaccard_check import guardrail_problems  # scripts/, via the header bootstrap

    for src, _ in OPEN_FILES:
        if "p3_paraphrase" not in src:
            continue
        with open(src, newline="", encoding="utf-8") as f:
            bad = [(r.get("src_id", "?"), r.get("variant", "?"), p)
                   for r in csv.DictReader(f) if (p := guardrail_problems(r["text"]))]
        if bad:
            raise SystemExit(f"REFUSING to package {src}: {len(bad)} lure(s) fail the guardrail — "
                             + "; ".join(f"{i}/{v}: {', '.join(p)}" for i, v, p in bad[:5]))
        print(f"[ok] {src}: every lure passes the attack-set guardrail")


def _infra_rows(path):
    """Rows as the article counts them: data rows for a CSV, lines for a seen-set."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        n = sum(1 for _ in fh)
    return n - 1 if path.endswith(".csv") else n


def _infra_table_counts():
    """The row counts the P4b article prints, keyed by the deposit path it prints them for."""
    tex = os.path.join(ROOT, "papers", "P4b_infra_data", "sections", "tab_files.tex")
    want = {}
    for line in open(tex, encoding="utf-8"):
        m = re.match(r"\s*\\quad\\texttt\{([^}]+)\}\s*&\s*([\d{,}]+)\s+(?:rows|lines)", line)
        if m:
            want[m.group(1).replace("\\", "")] = int(m.group(2).replace("{,}", ""))
    return want


def _check_infra_guardrails(files):
    """No page content, ever: the deposit is records ABOUT pages (ethics statement, P4b)."""
    for src, arc in files:
        if not arc.endswith(".csv"):
            continue
        with open(src, encoding="utf-8", errors="replace") as fh:
            header = fh.readline().strip().lower().split(",")
            sample = [fh.readline() for _ in range(200)]
        bad = [c for c in header
               if any(w in c for w in INFRA_FORBIDDEN_COLS) and c not in
               ("registered_domain", "domain", "cname_present", "content_confirmed")]
        if bad:
            raise SystemExit(f"{arc}: column(s) {bad} could carry page content — "
                             "the ethics statement says the deposit carries none")
        if any("<html" in ln.lower() or "<!doctype" in ln.lower() for ln in sample if ln):
            raise SystemExit(f"{arc}: markup found in the first rows — page content must not ship")


def build_p4b(version, out):
    """Stage the P4b deposit as a directory (and a zip beside it), from the files the article
    names. Draft until P4's trigger freezes v1.0.0."""
    missing = [s for s, _ in INFRA_FILES if not os.path.exists(s)]
    if missing:
        raise SystemExit("Missing (run `make p4 p4b` first?): " + ", ".join(missing))
    _check_infra_guardrails(INFRA_FILES)

    want, seen = _infra_table_counts(), {}
    for src, arc in INFRA_FILES:
        if arc.startswith("data/"):
            seen[arc] = _infra_rows(src)
    drift = {k: (want[k], seen[k]) for k in want if k in seen and want[k] != seen[k]}
    if drift:
        raise SystemExit("the article and the deposit disagree on row counts "
                         f"(article, deposit): {drift} — run `make p4b` and rebuild the PDF")
    if set(want) - set(seen):
        raise SystemExit(f"the article lists files this build does not stage: {set(want) - set(seen)}")

    root = os.path.join(out, f"PhishVN-Infra_v{version}_open")
    if os.path.isdir(root):
        shutil.rmtree(root)
    for src, arc in INFRA_FILES:
        dst = os.path.join(root, arc)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    manifest = [f"PhishVN-Infra v{version} — MANIFEST (SHA-256)", ""]
    for _, arc in sorted(INFRA_FILES, key=lambda t: t[1]):
        f = os.path.join(root, arc)
        manifest.append(f"{_sha256(f)}  {arc}  ({os.path.getsize(f)} bytes)")
    if "draft" in version:
        manifest += ["", "DRAFT: v1.0.0 freezes at the time-stamped pre-specified trigger of the companion",
                     "study; these files are the build snapshot, not the deposit.",
                     "Written at freeze: " + ", ".join(INFRA_AT_FREEZE)]
    open(os.path.join(root, "MANIFEST.txt"), "w", encoding="utf-8").write(
        "\n".join(manifest) + "\n")

    zip_path = root + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _, names in os.walk(root):
            for n in sorted(names):
                f = os.path.join(dirpath, n)
                z.write(f, os.path.relpath(f, root))
    print(f"[+] {root}/")
    for arc, n in sorted(seen.items()):
        print(f"    {arc:<32} {n:>7,} rows")
    print(f"    {'MANIFEST.txt + docs + LICENSE':<32} {len(INFRA_FILES) - len(seen):>7} files")
    print(f"[+] {zip_path}  ({os.path.getsize(zip_path):,} bytes)")
    if "draft" in version:
        print("[i] DRAFT build; pending at freeze: " + ", ".join(INFRA_AT_FREEZE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--out", default=os.path.join(ROOT, "release"))
    ap.add_argument("--include-pages", action="store_true",
                    help="Also build the GATED HTML/screenshot bundle (research-only).")
    ap.add_argument("--p4b", action="store_true",
                    help="Build the infrastructure data article's deposit instead "
                         "(PhishVN-Infra: directory + zip under release/).")
    args = ap.parse_args()

    os.chdir(ROOT)
    if args.p4b:
        build_p4b("0.1.0-draft" if args.version == "1.0.0" else args.version, args.out)
        return
    missing = [s for s, _ in OPEN_FILES if not os.path.exists(s)]
    if missing:
        raise SystemExit("Missing (run `make data` first?): " + ", ".join(missing))

    _check_attack_guardrails()

    stats = _dataset_stats("data/processed/dataset_url.csv")
    readme = _readme(args.version, stats)

    # MANIFEST with checksums (computed on the real source files)
    manifest = [f"PhishVN v{args.version} — MANIFEST (SHA-256)", ""]
    for src, arc in OPEN_FILES:
        size = os.path.getsize(src)
        manifest.append(f"{_sha256(src)}  {arc}  ({size} bytes)")
    manifest_txt = "\n".join(manifest) + "\n"

    open_zip = os.path.join(args.out, f"PhishVN_v{args.version}_open.zip")
    write_bundle(open_zip, OPEN_FILES,
                 {"README.md": readme, "MANIFEST.txt": manifest_txt})
    print(f"[+] open tier  -> {open_zip}  ({os.path.getsize(open_zip)} bytes)")

    if args.include_pages:
        pages = []
        for d, arc_d in [("data/raw/landing", "pages/landing"),
                         ("data/raw/landing_shots", "pages/landing_shots")]:
            if os.path.isdir(d):
                for fn in sorted(os.listdir(d)):
                    pages.append((os.path.join(d, fn), f"{arc_d}/{fn}"))
        care = ("GATED / RESEARCH-ONLY.\n\nThese are captured LIVE PHISHING pages (HTML) and "
                "screenshots that clone real brands. Do NOT redistribute as attack material, do NOT "
                "open the HTML outside an isolated VM/container, and do NOT upload to an open-access "
                "tier. Share on request under a data-use agreement only. CC BY 4.0 does not waive "
                "these safety obligations.\n")
        gated_zip = os.path.join(args.out, f"PhishVN_v{args.version}_pages_GATED.zip")
        write_bundle(gated_zip, pages, {"HANDLE_WITH_CARE.txt": care})
        print(f"[+] gated tier -> {gated_zip}  ({len(pages)} artefacts, {os.path.getsize(gated_zip)} bytes)")

    print("Next: upload the OPEN zip to Mendeley Data (or Zenodo), get the DOI, then set it in "
          "CITATION.cff and papers/P1_dataset/preamble.tex (\\datadoi).")


if __name__ == "__main__":
    main()
