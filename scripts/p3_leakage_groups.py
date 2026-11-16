#!/usr/bin/env python3
"""
p3_leakage_groups.py — the leakage groups behind the group splits of the P3 resubmission.

Plan: papers/P3_multimodal/RESUBMISSION/PLAN.md, "Splits and leakage", as amended by Amendment 6.

A page's group is the union of two relations:
  * same key: the tenant key of Amendment 1 for frame pages (and for top-up pages), the
    organisation for login arm 2;
  * same kit cluster.

Fingerprint: the page's start tags in document order (an <input> carries its type, so
"input:password"), nothing else: no text, no attribute values, no URLs. Its 5-shingles are hashed;
two pages' similarity is the exact Jaccard of their shingle sets. A page with fewer than MIN_TAGS
start tags is a stub (redirect, error or empty page) and joins no kit cluster.

Kit clusters, two methods:
  components  connected components of the graph with an edge wherever Jaccard >= TAU (the plan's
              wording: "pages whose DOM shingles link them form one group"). Primary.
  leader      pages visited in a fixed order (first_detected, scan_uuid, page_id); a page joins
              the leader it is most similar to if that similarity is >= TAU (ties: the earlier
              leader), otherwise it becomes a leader. Every member is within TAU of its own leader,
              so similar pages cannot chain unrelated ones together. Fallback.

Size guard (Amendment 6), applied once labels exist and before any model is fitted; it reads
group sizes per class, never a model score. A group is oversized if it holds more than
GUARD_SHARE of the pages of an analysed class.
  1. no oversized components group  -> components primary, leader reported beside it;
  2. else no oversized leader group -> leader primary, components reported beside it;
  3. else                           -> components primary, and outcome 1 is re-run without the
                                       oversized components groups as a reported sensitivity.

On the published study's 3,389 pages (many pages per tenant) one components group held 1,942
pages, 71% of the benign class. Kept to the earliest page per tenant, as the frame rule does, the
largest group held 88 pages (7.1% of benign, 3.6% of phishing) under either method.

RUN (descriptive check on the published study's pages, not the frame):
  python scripts/p3_leakage_groups.py --published
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from html.parser import HTMLParser

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
try:
    from _path import ROOT, add_script_dirs
    add_script_dirs()
except ImportError:  # flat public-mirror layout
    ROOT = os.path.dirname(_HERE)

# ---- fixed by Amendment 6 (2026-11-16), before the frame is built. Changing one is an amendment.
SHINGLE_K = 5
MIN_TAGS = 25
TAU = 0.8
GUARD_SHARE = 0.20            # one fold's share of a class under 5-fold splits
METHODS = ("components", "leader")


class _Tags(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "input":
            tag = "input:" + ((dict(attrs).get("type") or "text").lower())
        self.tags.append(tag)


def fingerprint(html: str) -> frozenset | None:
    """Hashed start-tag 5-shingles, or None for a stub (fewer than MIN_TAGS start tags)."""
    p = _Tags()
    try:
        p.feed(html or "")
    except Exception:  # noqa: BLE001 — a page the parser rejects keeps what it parsed so far
        pass
    if len(p.tags) < MIN_TAGS:
        return None
    k = SHINGLE_K
    return frozenset(hashlib.blake2b("/".join(p.tags[i:i + k]).encode(), digest_size=8).digest()
                     for i in range(len(p.tags) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def leader_clusters(prints: list, tau: float = TAU) -> list[int]:
    """prints: fingerprints in visiting order (None = stub). Returns a cluster id per page.
    A stub gets a cluster of its own."""
    leaders: list[tuple[int, frozenset]] = []   # (cluster id, leader fingerprint), in order made
    out, next_id = [], 0
    for fp in prints:
        best, best_j = None, -1.0
        if fp is not None:
            for cid, lfp in leaders:            # earliest leader first, so ties keep the earlier one
                j = jaccard(fp, lfp)
                if j >= tau and j > best_j:
                    best, best_j = cid, j
        if best is None:
            best = next_id
            next_id += 1
            if fp is not None:
                leaders.append((best, fp))
        out.append(best)
    return out


def component_clusters(prints: list, tau: float = TAU) -> list[int]:
    """Connected components of the graph with an edge wherever exact Jaccard >= tau. Stubs are
    singletons. Cluster ids are numbered by first appearance in the given order."""
    n = len(prints)
    uf = _UF(n)
    index: dict[bytes, list[int]] = {}
    for i, fp in enumerate(prints):
        if fp is None:
            continue
        cand = set()
        for h in fp:                            # pages sharing at least one shingle
            cand.update(index.get(h, ()))
            index.setdefault(h, []).append(i)
        for j in cand:
            if uf.find(i) != uf.find(j) and jaccard(fp, prints[j]) >= tau:
                uf.union(i, j)
    ids, out = {}, []
    for i in range(n):
        out.append(ids.setdefault(uf.find(i), len(ids)))
    return out


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def assign_groups(pages: list[dict], read_html, method: str = "components") -> list[dict]:
    """pages: dicts with page_id, key (tenant key or organisation), first_detected, scan_uuid.
    read_html(page) -> str; method in METHODS. Returns, in input order, {page_id, kit, group, stub}; group ids are
    0..G-1 numbered by first appearance in visiting order."""
    order = sorted(range(len(pages)), key=lambda i: (pages[i].get("first_detected") or "",
                                                     pages[i].get("scan_uuid") or "",
                                                     pages[i]["page_id"]))
    prints = [fingerprint(read_html(pages[i])) for i in order]
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    kit_of_pos = (component_clusters if method == "components" else leader_clusters)(prints)
    n = len(pages)
    uf = _UF(n)
    first_by = {}
    for pos, i in enumerate(order):
        for tag in (("kit", kit_of_pos[pos]), ("key", pages[i]["key"])):
            if tag in first_by:
                uf.union(first_by[tag], pos)
            else:
                first_by[tag] = pos
    gid, out = {}, [None] * n
    for pos, i in enumerate(order):
        root = uf.find(pos)
        gid.setdefault(root, len(gid))
        out[i] = {"page_id": pages[i]["page_id"], "kit": kit_of_pos[pos], "group": gid[root],
                  "stub": prints[pos] is None}
    return out


def largest_share(groups: list[dict], labels: list[str]) -> dict:
    """Per analysed class, the share of its pages held by its largest group, and that group."""
    from collections import Counter
    per_class = Counter(labels)
    cell = Counter(zip((g["group"] for g in groups), labels))
    out = {}
    for c, n in per_class.items():
        gid, v = max(((g, v) for (g, cc), v in cell.items() if cc == c), key=lambda t: t[1])
        out[c] = {"share": v / n, "group": gid}
    return out


def oversized(groups: list[dict], labels: list[str], share: float = GUARD_SHARE) -> list[int]:
    """Groups holding more than `share` of the pages of some analysed class, sorted."""
    from collections import Counter
    per_class = Counter(labels)
    cell = Counter(zip((g["group"] for g in groups), labels))
    return sorted({g for (g, c), v in cell.items() if v / per_class[c] > share})


def primary_method(components: list[dict], leader: list[dict], labels: list[str],
                   share: float = GUARD_SHARE) -> dict:
    """The size guard of Amendment 6. Reads group sizes per class only, never a model score.
      1. no components group over `share` of a class -> components is primary;
      2. else, no leader group over it               -> leader is primary;
      3. else                                         -> components stays primary, and its
         oversized groups are dropped in a reported sensitivity run (`drop_groups`)."""
    big_c = oversized(components, labels, share)
    if not big_c:
        return {"primary": "components", "rule": 1, "drop_groups": []}
    if not oversized(leader, labels, share):
        return {"primary": "leader", "rule": 2, "drop_groups": []}
    return {"primary": "components", "rule": 3, "drop_groups": big_c}


def _published() -> int:
    """Descriptive check on the published study's 3,389 pages (data/interim/content_manifest.csv).
    Capture time comes from the scan UUID (v7); pages without one are visited last."""
    import uuid
    import json
    import pandas as pd
    from make_subdomain_assets import normalize, tenant_key
    man = pd.read_csv(os.path.join(ROOT, "data", "interim", "content_manifest.csv"))
    hosted = json.load(open(os.path.join(ROOT, "data", "processed", "subdomain",
                                         "input_manifest.json")))["hosted_suffixes"]

    def when(u):
        try:
            x = uuid.UUID(str(u))
        except ValueError:
            return "~"
        return f"{x.int >> 80:015d}" if x.version == 7 else "~"

    pages = [{"page_id": str(i), "key": tenant_key(normalize(r.domain), hosted),
              "first_detected": when(r.scan_uuid), "scan_uuid": str(r.scan_uuid),
              "path": r.dom_file, "label": r.label} for i, r in enumerate(man.itertuples())]
    read = lambda p: open(p["path"], encoding="utf-8", errors="replace").read()  # noqa: E731
    first = {}
    for p in sorted(pages, key=lambda p: (p["first_detected"], p["scan_uuid"])):
        first.setdefault(p["key"], p)
    for tag, subset in (("all pages", pages), ("earliest page per tenant", list(first.values()))):
        labels = [p["label"] for p in subset]
        res = {m: assign_groups(subset, read, m) for m in METHODS}
        for method, g in res.items():
            df = pd.DataFrame(g).assign(label=labels)
            size = df.groupby("group").size()
            per = df.groupby(["group", "label"]).size().unstack(fill_value=0)
            shares = ", ".join(f"{c} {per[c].max()}/{per[c].sum()} ({per[c].max() / per[c].sum():.1%})"
                               for c in per.columns)
            print(f"{tag:26s} {method:10s} pages {len(df)} | groups {len(size)} | largest "
                  f"{size.nlargest(5).tolist()} | largest per class: {shares} | "
                  f"mixed-class groups {int((per.min(axis=1) > 0).sum())}")
        print(f"{'':26s} guard -> {primary_method(res['components'], res['leader'], labels)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--published", action="store_true",
                    help="describe the groups on the published study's pages")
    a = ap.parse_args()
    if a.published:
        return _published()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
