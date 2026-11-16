#!/usr/bin/env python3
"""
p3_compphish_baseline.py — CompPhish's released features for the P3 resubmission (Amendment 7).

Plan: papers/P3_multimodal/RESUBMISSION/PLAN.md, Amendment 7.

CompPhish version 4 (DOI 10.17632/fmbs4kp9wz.4) ships `feature extractor.zip`, whose module
`feature_Generator_with_brands` defines `extract_features(serial, url, label, html, mean_phishing_length,
mean_domain_length)`. That function is called unchanged. Only the loop around it is ours: the
released loop skips a page whose HTML file is missing and reads the files with relative paths.

What this module adds around the call:
  * the zip is checked against its pinned SHA-256 before it is unpacked or imported;
  * the module reads its brand, shortener and free-hosting lists from the working directory, at
    import and on every call, so both happen inside that directory;
  * HTML is decoded as UTF-8; bytes that are not UTF-8 are replaced and the page is flagged
    (`utf8_replaced`), where the released loop would have dropped it;
  * a page on which `extract_features` raises gets no features and is flagged (`extract_error`);
  * the two features that compare a length with the mean length of the phishing URLs
    (`url_more_than_mean`, `domain_more_than_mean`) read the labels of every page. They are
    extracted with placeholder means and then set per outer split from that split's training
    abuse-class pages only (`fold_thresholds`).

Feature sets (Amendment 7): CP_URL, the 47 URL features; CP_HTML, the 23 HTML features; CP_ALL.

RUN:
  python scripts/p3_compphish_baseline.py --verify-v4
      extracts the URL features of CompPhish's own 15,358 URLs and compares them with the released
      All_Features_threshold90.xlsx, as a check that the extractor is called correctly
IN:   data/external/compphish/ (the version-4 files, downloaded from Mendeley Data)
NEEDS: thefuzz 0.22.1, Levenshtein, beautifulsoup4, tldextract (5.2.0 pinned by CompPhish)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

# ---- pinned by Amendment 7 (2026-11-16). Changing one is an amendment.
CP_DIR = os.path.join(ROOT, "data", "external", "compphish")
SHA256 = {
    "feature extractor.zip": "0e72f280a189fdc91c65aa3035334676df26018da8d5b751a0d3f980007b858a",
    "Mapping_File.xlsx": "e6dda6a21d925f8aba36090f295b08e9808a5dfe8dd3276bf8cd82022855f14a",
    "All_HTML.zip": "12440e4f911fabf4ec2c712ae014cb43e638e9a6adc6c7c6a506a8a7df24fcf7",
    "All_Features_threshold90.xlsx": "6e87b7997c2d0e9e3f1e02fd5ee1240442385f9bea5f0a40e46e7b6e43ada21b",
}
PINNED_PACKAGES = {"thefuzz": "0.22.1", "tldextract": "5.2.0", "Levenshtein": "0.27.1"}

CP_URL = ["brand_subdomain", "brand_substring_domain", "brand_typo_dom", "fake_tld",
          "brand_in_path_query", "dots_path", "url_more_than_mean", "domain_more_than_mean",
          "gibberish", "domain_length", "url_length", "url_contains_ip", "presence_of_url_shortener",
          "special_characters_count", "no_of_at", "no_of_comma", "no_of_dollar", "no_of_semicolon",
          "no_of_space", "no_of_ampersand", "no_of_dots", "no_of_equal", "no_of_percent",
          "no_of_underscore", "no_of_hyphen", "no_of_questionmark", "no_of_colon",
          "no_of_slashes_inpath", "has_asterisk", "has_or", "has_embedded_url", "uses_https",
          "num_query_params", "numeric_proportion", "url_entropy", "hostname_digit_ratio",
          "has_redirections", "has_punycode", "count_www", "count_com",
          "url_has_high_risk_extension", "has_explicit_port", "total_words_url",
          "average_length_of_words", "longest_word_length", "shortest_word_length",
          "presence_of_free_hosting"]
CP_HTML = ["no_of_forms", "no_of_Images", "no_of_hyperlinks", "no_of_external_links",
           "no_of_internal_links", "no_of_null_links", "no_of_selfreference_links",
           "no_of_empty_links", "EmptyToExternalLinksRatio", "Risky_extensions_In_Html",
           "presence_of_shorten_url_html", "hidden_elements_html", "Script_loaded_from_ext_domain",
           "suspicious_inline_js", "is_footer_mismatch", "is_header_mismatch",
           "is_copyright_mismatch", "is_socials_mismatch", "is_favicon_mismatch",
           "has_missing_title", "title_mismatch_with_domain", "has_inline_event_handlers",
           "form_action_external"]
CP_ALL = CP_URL + CP_HTML
SETS = {"CP-URL": CP_URL, "CP-HTML": CP_HTML, "CP-all": CP_ALL}
LABEL_MEAN = {"url_more_than_mean": "url_length", "domain_more_than_mean": "domain_length"}
PLACEHOLDER_MEAN = 0.0      # every page gets 1; fold_thresholds overwrites both columns


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_pin(path: str) -> None:
    """Refuse a CompPhish file whose SHA-256 is not the pinned one."""
    name = os.path.basename(path)
    if name not in SHA256:
        raise SystemExit(f"[!] {name} is not a pinned CompPhish file")
    got = sha256(path)
    if got != SHA256[name]:
        raise SystemExit(f"[!] {name}: SHA-256 {got} is not the pinned {SHA256[name]} (Amendment 7)")


def package_drift() -> dict:
    """Installed versions that differ from the pins, as {package: (installed, pinned)}."""
    from importlib import metadata
    out = {}
    for pkg, pin in PINNED_PACKAGES.items():
        try:
            have = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            have = None
        if have != pin:
            out[pkg] = (have, pin)
    return out


@contextlib.contextmanager
def _cwd(path: str):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class Extractor:
    """CompPhish's extract_features, imported from the pinned zip and always called from inside
    the unpacked directory (the module opens its lists by relative path on every call)."""

    def __init__(self, cp_dir: str = CP_DIR):
        z = os.path.join(cp_dir, "feature extractor.zip")
        check_pin(z)
        root = os.path.join(cp_dir, "extractor_v4")
        self.dir = os.path.join(root, "feature extractor")
        if not os.path.isfile(os.path.join(self.dir, "feature_Generator_with_brands.py")):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(root)
        if self.dir not in sys.path:
            sys.path.insert(0, self.dir)
        with _cwd(self.dir):
            import feature_Generator_with_brands as mod  # noqa: E402 — the released module
        self.mod = mod

    def page(self, serial, url: str, html: str) -> dict:
        with _cwd(self.dir):
            f = self.mod.extract_features(serial, url, None, html, PLACEHOLDER_MEAN, PLACEHOLDER_MEAN)
        return {k: f[k] for k in CP_ALL}


def decode(raw: bytes) -> tuple[str, bool]:
    """UTF-8, as the released loop reads; bytes that are not UTF-8 are replaced and flagged."""
    try:
        return raw.decode("utf-8"), False
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace"), True


def extract_pages(pages: list[dict], read_html, extractor: Extractor | None = None) -> list[dict]:
    """pages: dicts with page_id and url; read_html(page) -> bytes. Returns one row per page:
    page_id, every CP_ALL feature (None if the extractor raised), utf8_replaced, extract_error."""
    ex = extractor or Extractor()
    out = []
    for p in pages:
        html, replaced = decode(read_html(p))
        row = {"page_id": p["page_id"], "utf8_replaced": int(replaced), "extract_error": ""}
        try:
            row.update(ex.page(p["page_id"], p["url"], html))
        except Exception as e:  # noqa: BLE001 — counted, as Amendment 7 says
            row.update({k: None for k in CP_ALL})
            row["extract_error"] = repr(e)[:200]
        out.append(row)
    return out


def fold_thresholds(rows: list[dict], train_idx, labels, positive="abuse") -> list[dict]:
    """Set the two label-mean flags for one outer split. The means are taken over the training
    rows whose label is `positive`; every row (train and test) is then compared with them.
    Returns new row dicts; the input is not changed. Rows with no features stay None."""
    tr = [i for i in train_idx if labels[i] == positive and rows[i].get("url_length") is not None]
    if not tr:
        raise ValueError("no training row of the positive class to take the mean from")
    means = {flag: sum(rows[i][src] for i in tr) / len(tr) for flag, src in LABEL_MEAN.items()}
    out = []
    for r in rows:
        r = dict(r)
        for flag, src in LABEL_MEAN.items():
            r[flag] = None if r.get(src) is None else int(r[src] >= means[flag])
        out.append(r)
    return out


def _verify_v4() -> int:
    """URL features of CompPhish's own URLs against its released feature file (empty HTML)."""
    import pandas as pd
    for name in ("Mapping_File.xlsx", "All_Features_threshold90.xlsx"):
        check_pin(os.path.join(CP_DIR, name))
    drift = package_drift()
    if drift:
        print(f"[warn] package versions differ from the CompPhish pins: {drift}")
    m = pd.read_excel(os.path.join(CP_DIR, "Mapping_File.xlsx"))
    pub = pd.read_excel(os.path.join(CP_DIR, "All_Features_threshold90.xlsx")).set_index("serial_number")
    pages = [{"page_id": int(r.serial_number), "url": r.url} for r in m.itertuples()]
    rows = extract_pages(pages, lambda p: b"")
    df = pd.DataFrame(rows).set_index("page_id")
    lab = m.set_index("serial_number").label.map({1: "abuse", 0: "legitimate"})
    df = pd.DataFrame(fold_thresholds(df.reset_index().to_dict("records"), range(len(df)),
                                      list(lab.loc[df.index]))).set_index("page_id")
    same = [c for c in CP_URL if c in pub.columns]
    agree = {c: float(((df[c].astype(float) - pub.loc[df.index, c].astype(float)).abs() < 1e-6).mean())
             for c in same}
    exact = [c for c, v in agree.items() if v == 1.0]
    print(f"pages {len(df)} | extractor errors {int((df.extract_error != '').sum())} | "
          f"URL features compared {len(same)} | identical on every page {len(exact)}")
    for c, v in sorted(agree.items(), key=lambda t: t[1]):
        if v < 1.0:
            print(f"  {c:28s} agree {v:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify-v4", action="store_true",
                    help="compare the extracted URL features of CompPhish's URLs with its release")
    a = ap.parse_args()
    if a.verify_v4:
        return _verify_v4()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
