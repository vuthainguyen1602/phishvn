#!/usr/bin/env python3
"""
extract_js_features.py — JavaScript modality for the P1b content/JS fusion (PhishFusion showed the
URL$\\times$JS interaction is under-exploited). Two representations of a page's inline+referenced JS:

  1. LIGHTWEIGHT obfuscation/behaviour features (always available, interpretable): script counts,
     total/avg JS length, Shannon entropy, hex/unicode-escape density, and call-frequency of the
     API surface phishing kits lean on (eval, unescape, atob, fromCharCode, document.write,
     setTimeout, addEventListener, XMLHttpRequest/fetch, location, cookie, form.submit, ...),
     plus inline event-handler and password-field counts.
  2. CODEBERT embedding (optional, --encoder codebert): frozen microsoft/codebert-base mean-pooled
     over the concatenated scripts, exactly the branch PhishFusion uses. Falls back to lightweight
     features if transformers/torch or the model download is unavailable.

Used by train_content_fusion.py; can also be run standalone to dump a feature CSV.
"""
from __future__ import annotations
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

SCRIPT_RE = re.compile(r"(?is)<script\b([^>]*)>(.*?)</script>")
SRC_RE = re.compile(r"""(?i)\bsrc\s*=\s*['"]?([^'">\s]+)""")
EVENT_RE = re.compile(r"(?i)\son[a-z]+\s*=")
PW_RE = re.compile(r"""(?i)<input[^>]*type\s*=\s*['"]?password""")
HEX_RE = re.compile(r"(\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|%[0-9a-fA-F]{2})")
QR_RE = re.compile(r"(?i)\bqrcode\b|\bqr-code\b|\bqr_code\b")

# API surface phishing kits disproportionately use (obfuscation, exfiltration, redirection)
API_TOKENS = [
    "eval", "unescape", "atob", "btoa", "fromcharcode", "document.write", "settimeout",
    "setinterval", "addeventlistener", "xmlhttprequest", "fetch(", "location.href",
    "location.replace", "window.location", "document.cookie", ".submit(", "createelement",
    "appendchild", "encodeuricomponent", "charcodeat", "string.fromcharcode", "escape(",
]

# The numeric feature vector, in a fixed order (train_content_fusion depends on this).
#
# js_has_qr is OPT-IN, behind PHISHVN_JS_QR=1. It was added 2026-08-30 for the quishing direction,
# and appending it unconditionally would have moved the JavaScript channel of a study already
# written against this schema: a thirteenth column changes what the model is fitted on, and that
# study reports a "JavaScript only" figure and a three-channel stack measured on twelve. The
# claims suite would have caught the drift, but only after a twenty-minute regeneration, and only
# if someone regenerated at all. So the published schema stays the published schema, and the new
# column is requested by whoever wants it. Drop the flag once those numbers can be restated
# deliberately rather than moved underneath a manuscript.
_JS_BASE = ["js_n_scripts", "js_n_external", "js_n_inline", "js_total_len", "js_avg_len",
            "js_entropy", "js_hex_density", "js_event_handlers", "js_password_fields",
            "js_has_iframe", "js_uniq_apis"]
JS_QR_ENABLED = os.environ.get("PHISHVN_JS_QR", "") not in ("", "0", "false", "False")
if JS_QR_ENABLED:
    JS_COLUMNS = (_JS_BASE[:-1] + ["js_has_qr"] + _JS_BASE[-1:]
                  + [f"js_api_{re.sub(chr(92)+chr(87), '', t)}" for t in API_TOKENS])
else:
    JS_COLUMNS = (_JS_BASE
                  + [f"js_api_{re.sub(chr(92)+chr(87), '', t)}" for t in API_TOKENS])


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def scripts_of(html: str):
    """Return (inline_code_concat, n_external, n_inline) from a page's <script> tags."""
    inline, ext = [], 0
    for attrs, body in SCRIPT_RE.findall(html or ""):
        if SRC_RE.search(attrs):
            ext += 1
        if body and body.strip():
            inline.append(body)
    return "\n".join(inline), ext, len(inline)


def external_js(js_dir: str) -> list:
    """Bodies of the EXTERNAL .js files captured under js_dir at crawl time
    (see watch_chongluadao.fetch_external_js). Empty list when there is no such dir."""
    import glob as _glob
    import os as _os
    out = []
    if js_dir and _os.path.isdir(js_dir):
        for p in sorted(_glob.glob(_os.path.join(js_dir, "*.js"))):
            try:
                out.append(open(p, encoding="utf-8", errors="ignore").read())
            except Exception:
                pass
    return out


def all_js(html: str, js_dir: str = "") -> str:
    """Concatenate a page's INLINE scripts with any EXTERNAL .js bodies captured under js_dir.
    Use this when the external JS was saved at crawl-at-detection time — it gives the JS branch
    the full script surface, not just inline."""
    parts = [scripts_of(html)[0]] + external_js(js_dir)
    return "\n".join(x for x in parts if x)


def js_lightweight(html: str, js_dir: str = "") -> dict:
    inline, n_ext, n_inl = scripts_of(html)
    ext_bodies = external_js(js_dir)
    # code-content features run over inline + captured external bodies; the structural counts
    # (n_scripts/n_external/n_inline) stay tag-derived so they mean the same with or without js_dir
    code = "\n".join([inline] + ext_bodies) if ext_bodies else inline
    n_blobs = n_inl + len(ext_bodies)
    low = code.lower()
    total = len(code)
    n_scripts = n_ext + n_inl
    feats = {
        "js_n_scripts": n_scripts,
        "js_n_external": n_ext,
        "js_n_inline": n_inl,
        "js_total_len": total,
        "js_avg_len": (total / n_blobs) if n_blobs else 0.0,
        "js_entropy": _entropy(code),
        "js_hex_density": (len(HEX_RE.findall(code)) / total) if total else 0.0,
        "js_event_handlers": len(EVENT_RE.findall(html or "")),
        "js_password_fields": len(PW_RE.findall(html or "")),
        "js_has_iframe": 1 if re.search(r"(?i)<iframe\b", html or "") else 0,
        # Only when PHISHVN_JS_QR=1 -- see the note on JS_COLUMNS. A key absent from the row and a
        # column absent from JS_COLUMNS have to agree, or train_content_fusion reads a ragged frame.
        **({"js_has_qr": 1 if QR_RE.search(html or "") or QR_RE.search(code) else 0}
           if JS_QR_ENABLED else {}),
    }
    apis = {"js_api_" + re.sub(r"\W", "", t): low.count(t) for t in API_TOKENS}
    feats["js_uniq_apis"] = sum(1 for v in apis.values() if v > 0)
    feats.update(apis)
    return {k: float(feats[k]) for k in JS_COLUMNS}


def js_vector(html: str, js_dir: str = "") -> list:
    f = js_lightweight(html, js_dir)
    return [f[c] for c in JS_COLUMNS]


# ---- optional CodeBERT branch (PhishFusion-style) ----
_CODEBERT = {}


def codebert_embed(codes: list, batch=16):
    """Frozen microsoft/codebert-base mean-pooled embedding per script blob. Returns an (n, 768)
    array, or None if transformers/torch or the model is unavailable."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        if not _CODEBERT:
            # prefer the local safetensors conversion (convert_phobert_safetensors.py) — the Hub
            # repo is bin-only, which transformers 5.x refuses under torch < 2.6
            local = os.path.join(ROOT, "models", "codebert-base-safetensors")
            src = local if os.path.isdir(local) else "microsoft/codebert-base"
            _CODEBERT["tok"] = AutoTokenizer.from_pretrained(src)
            _CODEBERT["mdl"] = AutoModel.from_pretrained(src).eval()
    except Exception as e:
        print(f"[i] CodeBERT unavailable ({type(e).__name__}); using lightweight JS features.")
        return None
    import numpy as np
    tok, mdl = _CODEBERT["tok"], _CODEBERT["mdl"]
    out = []
    with __import__("torch").no_grad():
        for i in range(0, len(codes), batch):
            chunk = [c[:4000] if c else "" for c in codes[i:i + batch]]
            enc = tok(chunk, truncation=True, padding=True, max_length=256, return_tensors="pt")
            h = mdl(**enc).last_hidden_state.mean(1)
            out.append(h.numpy())
    return np.vstack(out)
