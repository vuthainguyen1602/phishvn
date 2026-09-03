#!/usr/bin/env python3
"""
llm_content_baseline.py — P1b zero-/few-shot MLLM phishing detector baseline (screenshot + HTML).

The 2026 literature uses multimodal LLMs *as the detector*, while P1b benchmarks only trained
fusion detectors; this adds the missing comparison point on the SAME content_manifest_vi.csv. It
NEVER trains on PhishVN — zero-shot over the entire labelled subset, stricter than the trained
models' held-out 30%. --shots N adds a few-shot condition whose exemplars are excluded from
scoring. Refusals are recorded and counted, never retried on a fallback model.

INPUT  : data/interim/content_manifest_vi.csv  [domain,label,provenance,tier,dom_file,shot_file,...]
OUTPUT : data/processed/p3/llm_content_baseline.csv + a JSON verdict cache so re-runs cost nothing.
Provider: Claude via the Anthropic SDK; default claude-opus-5. Set ANTHROPIC_API_KEY.

RUN:
  python scripts/llm_content_baseline.py                       # full manifest, zero-shot
  python scripts/llm_content_baseline.py --limit 20 --dry-run  # wiring check, no API calls
  python scripts/llm_content_baseline.py --shots 6             # few-shot (3 phishing + 3 benign)
  python scripts/llm_content_baseline.py --model claude-sonnet-5 --effort medium
The evaluation framing, the cost ledger and the refusal policy: kept in the development repository, not shipped in this mirror
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import json
import os
import random
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)
MANIFEST = os.path.join(ROOT, "data", "interim", "content_manifest_vi.csv")
# Legacy single-run path, kept only so an older CSV on disk is still recognisable.
# The live output path is derived per condition in main(); see --out.
OUT_CSV = os.path.join(ROOT, "data", "processed", "p3", "llm_content_baseline.csv")
CACHE = os.path.join(ROOT, "data", "processed", "p3", "llm_content_baseline_cache.json")

# Screenshot-first, HTML auxiliary — the 2026 recommendation. Kept model-agnostic and lure-neutral
# (no VN brand hints) so the detector reasons from the page, not from our labelling heuristics.
SYSTEM = (
    "You are a phishing-website detector. You are shown a screenshot of a rendered web page and, when "
    "available, an excerpt of its HTML. Decide whether the page is a PHISHING page (impersonates a "
    "brand/service to harvest credentials, payment details, OTPs, or personal data; fake login/payment/"
    "verification flows; deceptive giveaways or account-suspension lures) or BENIGN (a legitimate site, "
    "parking page, error page, or unrelated content). Judge only from the evidence shown. Respond via "
    "the required JSON schema: verdict, a calibrated confidence in [0,1], and one short reason."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["phishing", "benign"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "confidence", "reason"],
    "additionalProperties": False,
}

HTML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
MAX_HTML_CHARS = 6000        # keep the auxiliary text bounded (cost + the screenshot is primary)
PHISH_WORDS = ("1", "phishing", "phish", "smishing", "spam", "malicious")

# USD per MTok (input, output) — public sticker prices, matched by model-ID prefix, for the
# summary's cost estimate only (billing truth lives in the Console).
PRICES = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
CACHE_READ_FACTOR = 0.1      # cache reads bill at ~0.1x the input price


def is_phishing_label(v) -> int:
    return 1 if str(v).strip().lower() in PHISH_WORDS else 0


def html_excerpt(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        html = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return ""
    text = WS_RE.sub(" ", HTML_TAG_RE.sub(" ", html)).strip()
    return text[:MAX_HTML_CHARS]


def image_block(path: str):
    """Base64 PNG image content block, or None when the screenshot is missing/unreadable."""
    if not path or not os.path.exists(path):
        return None
    try:
        data = base64.standard_b64encode(open(path, "rb").read()).decode("ascii")
    except OSError:
        return None
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def load_rows(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"manifest not found: {path} (sync captures from the Jetson first)")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cache_key(cond: str, dom_file: str, shot_file: str, uuid: str) -> str:
    """cond = model|effort|shots(|exemplar-hash) — any condition change is a different verdict."""
    basis = uuid or f"{dom_file}|{shot_file}"
    return hashlib.sha1(f"{cond}|{basis}".encode()).hexdigest()


def page_content(img, html: str) -> list:
    """The user-turn content for one page: screenshot first, then the bounded HTML excerpt."""
    content = []
    if img:
        content.append(img)
    content.append({
        "type": "text",
        "text": ("Classify this page. The screenshot is the primary signal; the HTML excerpt below is "
                 "auxiliary and may be empty.\n\n--- HTML excerpt ---\n" + (html or "(none)")),
    })
    return content


def build_fewshot(rows, shots: int, root: str):
    """Balanced labelled exemplars as prior (user, assistant) turns, plus the exemplar row ids to
    exclude from scoring. Deterministic (seed 42) over manifest rows that have a screenshot. The
    last exemplar block carries the cache_control breakpoint, so the whole exemplar prefix is
    written to the prompt cache once and cache-read on every later page."""
    if shots <= 0:
        return [], set()
    per_class = shots // 2
    rng = random.Random(42)
    pools = {1: [], 0: []}
    for r in rows:
        shot = os.path.join(root, r.get("shot_file", "") or "")
        if r.get("shot_file") and os.path.exists(shot):
            pools[is_phishing_label(r.get("label", ""))].append(r)
    if len(pools[1]) < per_class or len(pools[0]) < per_class:
        raise SystemExit(f"--shots {shots} needs {per_class} exemplars per class with screenshots; "
                         f"have phishing={len(pools[1])} benign={len(pools[0])}")
    picks = rng.sample(pools[1], per_class) + rng.sample(pools[0], per_class)
    rng.shuffle(picks)
    msgs, used = [], set()
    for r in picks:
        used.add(r.get("scan_uuid") or f"{r.get('dom_file', '')}|{r.get('shot_file', '')}")
        img = image_block(os.path.join(root, r["shot_file"]))
        html = html_excerpt(os.path.join(root, r.get("dom_file", "") or ""))
        label = "phishing" if is_phishing_label(r.get("label", "")) else "benign"
        msgs.append({"role": "user", "content": page_content(img, html)})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps(
            {"verdict": label, "confidence": 0.9, "reason": "labelled reference example"})}]})
    msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return msgs, used


def classify(client, model: str, effort: str, fewshot: list, img, html: str) -> dict:
    """One classification. Returns {verdict, confidence, reason, latency_s, usage...} or, on
    failure, {error, latency_s}. Refusals are recorded (error=refusal:<category>), never retried
    on another model — the baseline stays single-model and the refusal rate is reported."""
    t0 = time.monotonic()
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=4096,  # thinking is on by default on Opus 5 and counts against this cap
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=fewshot + [{"role": "user", "content": page_content(img, html)}],
            output_config={"effort": effort,
                           "format": {"type": "json_schema", "schema": SCHEMA}},
        )
    except Exception as e:  # network / 4xx-5xx — record and move on, don't abort the sweep
        return {"error": f"{type(e).__name__}: {str(e)[:120]}",
                "latency_s": round(time.monotonic() - t0, 2)}
    meta = {
        "latency_s": round(time.monotonic() - t0, 2),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cache_read_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
        "served_model": resp.model,
    }
    if resp.stop_reason == "refusal":
        cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
        return {"error": f"refusal:{cat}", **meta}
    try:
        text = next(b.text for b in resp.content if b.type == "text")
        return {**json.loads(text), **meta}
    except (StopIteration, ValueError) as e:
        return {"error": f"parse:{type(e).__name__}", **meta}


def metrics(y_true: list[int], y_pred: list[int]) -> dict:
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    return {
        "n": len(y_true),
        "n_phish": sum(y_true),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=MANIFEST)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"],
                    help="Reasoning effort per page (low keeps the sweep cheap; default low)")
    ap.add_argument("--shots", type=int, default=0,
                    help="Few-shot exemplar count (N/2 per class, seed 42, excluded from scoring; "
                         "0 = zero-shot)")
    ap.add_argument("--limit", type=int, default=0, help="Only the first N usable pages (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="Exercise loading/caching; make no API calls")
    ap.add_argument("--out", default="",
                    help="Where to write the per-page predictions. Default is derived from the "
                         "CONDITION, not fixed, so a second model does not silently overwrite the "
                         "first one's results — the comparison this baseline exists for needs the "
                         "runs side by side. The verdict cache was already keyed on the condition; "
                         "only the CSV was not.")
    args = ap.parse_args()

    out_csv = args.out or os.path.join(
        ROOT, "data", "processed",
        f"llm_content_baseline_{args.model.replace('/', '-')}_k{args.shots}.csv")

    rows = load_rows(args.inp)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    client = None
    fewshot, exemplar_ids = build_fewshot(rows, args.shots, ROOT)
    # verdicts depend on the full condition, not just the model — key the cache on all of it
    cond = f"{args.model}|effort={args.effort}|shots={args.shots}"
    if args.shots:
        cond += "|" + hashlib.sha1("|".join(sorted(exemplar_ids)).encode()).hexdigest()[:12]
        print(f"[i] few-shot: {args.shots} exemplars ({args.shots // 2}/class) embedded as cached "
              f"prefix turns and excluded from scoring")

    results, skipped = [], 0
    for r in rows:
        rid = r.get("scan_uuid") or f"{r.get('dom_file', '')}|{r.get('shot_file', '')}"
        if rid in exemplar_ids:
            continue  # never score a page the prompt already labels
        dom = os.path.join(ROOT, r.get("dom_file", "") or "")
        shot = os.path.join(ROOT, r.get("shot_file", "") or "")
        img = image_block(shot)
        html = html_excerpt(dom)
        if img is None and not html:
            skipped += 1        # neither modality present locally (captures not synced) — skip
            continue

        y = is_phishing_label(r.get("label", ""))
        key = cache_key(cond, r.get("dom_file", ""), r.get("shot_file", ""), r.get("scan_uuid", ""))
        if key in cache:
            out = cache[key]
        elif args.dry_run:
            out = {"verdict": "?", "confidence": 0.0, "reason": "dry-run", "dry": True}
        else:
            if client is None:
                import anthropic
                client = anthropic.Anthropic()          # resolves ANTHROPIC_API_KEY / ant profile
            out = classify(client, args.model, args.effort, fewshot, img, html)
            if "error" not in out:
                cache[key] = out
                json.dump(cache, open(CACHE, "w"))       # checkpoint after every successful call

        results.append({
            "domain": r.get("domain", ""), "label": y,
            "verdict": out.get("verdict", ""), "confidence": out.get("confidence", ""),
            "error": out.get("error", ""), "modality": "shot+html" if img else "html",
            "latency_s": out.get("latency_s", ""),
            "input_tokens": out.get("input_tokens", ""),
            "output_tokens": out.get("output_tokens", ""),
            "cache_read_tokens": out.get("cache_read_tokens", ""),
            "served_model": out.get("served_model", ""),
        })
        if args.limit and len([x for x in results if not x["error"]]) >= args.limit:
            break

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["domain", "label", "verdict", "confidence", "error",
                                          "modality", "latency_s", "input_tokens", "output_tokens",
                                          "cache_read_tokens", "served_model"])
        w.writeheader(); w.writerows(results)

    graded = [x for x in results if not x["error"] and x["verdict"] in ("phishing", "benign")]
    refusals = sum(1 for x in results if str(x["error"]).startswith("refusal:"))
    print(f"[i] rows={len(rows)} usable={len(results)} skipped(no capture locally)={skipped} "
          f"graded={len(graded)} errored={sum(1 for x in results if x['error'])} "
          f"(refusals={refusals})")
    if args.dry_run:
        print("[dry-run] wiring OK — set ANTHROPIC_API_KEY and drop --dry-run to score for real.")
        return
    if not graded:
        print("[i] nothing graded (no captures synced yet, or all errored). See", out_csv); return
    y_true = [x["label"] for x in graded]
    y_pred = [1 if x["verdict"] == "phishing" else 0 for x in graded]
    m = metrics(y_true, y_pred)
    shot_tag = f"{args.shots}-shot" if args.shots else "zero-shot"
    print(f"[+] {shot_tag} {args.model} (effort={args.effort}): F1={m['F1']:.3f} "
          f"P={m['precision']:.3f} R={m['recall']:.3f} acc={m['accuracy']:.3f} "
          f"(n={m['n']}, phish={m['n_phish']}) -> {out_csv}")

    # Cost/latency ledger — over graded pages that carry usage (older cache entries may not)
    lat = sorted(float(x["latency_s"]) for x in graded if x["latency_s"] != "")
    tok = [x for x in graded if x["input_tokens"] != ""]
    if lat:
        print(f"[i] latency: mean={sum(lat) / len(lat):.2f}s median={lat[len(lat) // 2]:.2f}s "
              f"p95={lat[int(len(lat) * 0.95)]:.2f}s (n={len(lat)})")
    if tok:
        t_in = sum(int(x["input_tokens"]) for x in tok)
        t_out = sum(int(x["output_tokens"]) for x in tok)
        t_cr = sum(int(x["cache_read_tokens"] or 0) for x in tok)
        line = f"[i] tokens: input={t_in:,} output={t_out:,} cache_read={t_cr:,} (n={len(tok)})"
        price = next((p for pre, p in PRICES.items() if args.model.startswith(pre)), None)
        if price:
            usd = (t_in * price[0] + t_cr * price[0] * CACHE_READ_FACTOR + t_out * price[1]) / 1e6
            line += f" ~= ${usd:.2f} at ${price[0]}/{price[1]} per MTok"
        print(line)


if __name__ == "__main__":
    main()
