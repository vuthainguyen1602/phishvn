#!/usr/bin/env python3
"""
train_qr_restore.py — a restoration network placed in front of the decoder, and the evaluation
that says whether it earned its place.

THE IDEA. Every degraded image in the synthetic set has its undegraded original, so the supervision
is exact rather than annotated: 1 clean plus N degraded per (URL x error-correction level x module
size). A small convolutional network learns degraded -> clean, and the decoder reads the restored
image instead of the raw one.

FIVE CONSTRAINTS, each written here because each is a way the result could look real and not be.

1. SPLIT BY URL, never by image. Many images share one payload and one module pattern; an
   image-level split puts the same QR on both sides and the network memorises patterns it will meet
   again at test time. The split key is url_sha1, hashed, so it is stable across runs and across
   machines.
2. THE BASELINE IS THE STRONGEST DECODER. Restoring an image so OpenCV succeeds where WeChatQR
   already succeeded untouched shows nothing about the threat. Reported: raw vs restored for each
   decoder, and restored-OpenCV against raw-WeChat.
3. SCORE DECODE SUCCESS, NOT PIXEL SIMILARITY. A QR decodes or it does not. L1 is the training
   loss because it is differentiable; DFR is the metric because it is the question. They are
   allowed to disagree, and if they do, the metric wins.
4. MOTION BLUR IS THE HARD CASE. It defeats all three decoders at 100%, so it is both the strongest
   argument for restoration and the likeliest thing to be unrecoverable. Reported separately.
5. NOTHING BUT THE NETWORK MAY DIFFER BETWEEN THE ARMS. Three arms are decoded, not two:

       raw        the degraded PNG exactly as the benchmark saw it
       control    the same PNG through the network's own plumbing and NOTHING else
       restored   the same PNG through that plumbing and the network

   The network's effect is `control` minus `restored`. `raw` minus `control` should be exactly zero
   and is printed rather than assumed, because it is the check that the plumbing is honest.

WHY THE PLUMBING IS A CONSTRAINT AND NOT AN IMPLEMENTATION DETAIL. The first version of this file
resized every image to a fixed 128x128 tile. At the wild render scale a code is about 130 px across,
so that resize was resampling a 2 px module down to 1.75 px -- destroying, before the network saw
anything, exactly the resolution the study exists to measure, and handing the network a target it
could not have reached. The network is fully convolutional and now runs at the NATIVE raster:
padding to a common canvas for batching, cropping back afterwards, which is lossless by
construction and verified by the `raw` vs `control` line rather than argued for here. Training is on
random crops, which is what makes native resolution affordable.

WHAT THIS MODEL CLASS CAN AND CANNOT DO, stated before the numbers rather than after them. The
network is four 3x3 convolutions: a 9 px receptive field, which at 2 px per module is 4.5 modules.
Three consequences follow, and they are predictions, not excuses:

  * A LOCAL correction is within reach -- contrast, salt-and-pepper, a blur narrower than the
    receptive field.
  * A GLOBAL DECISION IS NOT. `invert` is the sharp case: a patch of an inverted QR is
    indistinguishable from a patch of an ordinary one, and deciding polarity needs the quiet zone
    or the finder patterns, which no 9 px window contains. A local model must therefore fail on
    inversion however long it trains -- while WeChatQR, which detects the symbol first, reads
    inverted codes almost perfectly (7.8% DFR against 100% for the other two).
  * A GEOMETRIC one is not either. Nothing in a stack of convolutions inverts a rotation or a
    homography; that needs a rectifier that locates the finder patterns and solves for the
    transform. `rotate` and `perspective` are excluded from the training set for that reason, and
    the run warns if they are present.

RUN
    # 1. build the pairs. --calibration modules matters: under the frozen px calibration, blur and
    #    motion are 100% DFR at every strength, and a restorer cannot be shown to help on an input
    #    that carries no signal.
    python3 scripts/gen_synthetic_qr.py --n 200 --box 2 --calibration modules \
        --transforms blur,motion,saltpepper,contrast,invert,logo --out data/raw/qr_restore
    # 2. train and evaluate (workstation: torch, MPS)
    python3 scripts/train_qr_restore.py --dir data/raw/qr_restore --epochs 8
    # 3. evaluate the checkpoint again without retraining
    python3 scripts/train_qr_restore.py --dir data/raw/qr_restore --eval-only
    # 4. constraint 2 needs all three decoders, which live on the second Jetson and torch does not.
    #    Dump the arms as PNGs here, decode them there:
    python3 scripts/train_qr_restore.py --dir data/raw/qr_restore --eval-only \
        --dump data/raw/qr_restore_arms
    # then, on .204:  python3 scripts/benchmark_qr.py --dir data/raw/qr_restore_arms \
    #                     --out data/processed/qr/qr_restore_arms_dfr.csv
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, sys, tempfile, time

import numpy as np
from PIL import Image

# A conv stack cannot invert these, and including them teaches it to emit the blurred average of
# every rotation it saw -- which lowers L1 and destroys every code it touches.
GEOMETRIC = ("rotate", "perspective")
# The model classes PREREG_restore_v2.md registers. `conv` is the arm the first study ran.
ARCHS = ("conv", "hybrid", "restormer")
WHITE = 255


def split_key(url_sha1: str, holdout: float) -> str:
    """Deterministic per-URL assignment. Hashing rather than shuffling means the same URL lands in
    the same split on every machine and every rerun, without carrying a split file around."""
    h = int(hashlib.sha1(("split:" + url_sha1).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < holdout else "train"


def build_pairs(rows: list) -> list:
    """Pair each degraded image with the clean one from its own (url, EC, module size) group."""
    key = lambda r: (r["url_sha1"], r["ec_level"], r.get("box_size", ""))
    clean = {key(r): r["image"] for r in rows if r["transform"] == "clean"}
    return [(r, clean[key(r)]) for r in rows if r["transform"] != "clean" and key(r) in clean]


class Images:
    """Every image, once, as uint8 at its native size. 20,000 renders at the wild scale is about
    340 MB held this way and 1.4 GB held as float32, and re-reading them from disk every epoch
    costs more than the convolutions do."""

    def __init__(self, root: str):
        self.root, self.cache = root, {}

    def __call__(self, rel: str) -> np.ndarray:
        a = self.cache.get(rel)
        if a is None:
            a = np.asarray(Image.open(os.path.join(self.root, rel)).convert("L"), np.uint8)
            self.cache[rel] = a
        return a


def polarity(a: np.ndarray) -> np.ndarray:
    """One line, and it decides whether the network can learn anything at all.

    A QR is about half black, so a patch of an INVERTED code is indistinguishable from a patch of an
    ordinary one; only the quiet zone and the finder patterns carry the polarity, and neither fits
    in a 9 px receptive field. Train on a set containing `invert` without this stage and the same
    local input carries two opposite targets: the L1-optimal answer is the average, the network
    emits flat grey at an L1 of 0.367 and never moves, and every arm reads 100% DFR. That is not a
    result about restoration -- it is a mislabelled training set.

    Deciding polarity is a GLOBAL question with a global answer: is the image mostly dark? So it
    belongs in the plumbing, where it is one comparison, and not in the network, where it is
    impossible. It is applied identically to the control arm, so it can never be mistaken for
    something the network did -- and the control's own numbers then say what a single line of
    preprocessing is worth against a decoder that fails on 100% of inverted codes."""
    return 255 - a if a.mean() < 128 else a


def pad_to(a: np.ndarray, canvas: int) -> np.ndarray:
    """White padding to a square canvas: batching needs one shape, and a QR's quiet zone is white,
    so the padding is the same thing the border already is. Lossless -- the crop back returns the
    original raster exactly."""
    h, w = a.shape
    out = np.full((canvas, canvas), WHITE, np.uint8)
    out[:h, :w] = a
    return out


def make_model(torch, nn, ch: int = 32, arch: str = "conv"):
    """The model class is the ONE thing PREREG_restore_v2.md varies. Three arms, one factory.

    Every arm is resolution-agnostic: trained on 64 px crops, run on whole images, and neither of
    the two attention arms carries a learned position table that would fix an input size. Position
    comes from a depthwise 3x3 (arm B) or is unnecessary because attention is over channels rather
    than pixels (arm C).
    """
    if arch not in ARCHS:
        raise SystemExit(f"--arch must be one of {', '.join(ARCHS)}")

    class Restore(nn.Module):
        """Deliberately small, and fully convolutional so that training on crops and running on
        whole images is the same function. The claim is that restoration helps at all, not that a
        large network helps; a big model would make the result about capacity and about how long it
        was trained."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
                nn.Conv2d(ch, 1, 3, padding=1), nn.Sigmoid())

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        """Arm B's bottleneck block: pre-norm self-attention over every token of the 1/4-resolution
        map, then an MLP. The map is 1/4 of a 232 px canvas at inference -- 58x58 = 3,364 tokens --
        which is why the downsample is not optional: at full resolution this attention is 16x the
        memory for the same field of view, and the field of view is the whole point.

        Positional information arrives as a depthwise 3x3 added to the tokens (a convolutional
        positional encoding). A learned table would tie the model to one input size and break the
        crop-train / whole-image-run equivalence every arm here relies on."""

        def __init__(self, dim, heads=4, mlp=2):
            super().__init__()
            self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
            self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp), nn.GELU(), nn.Linear(dim * mlp, dim))

        def forward(self, x):
            x = x + self.cpe(x)
            b, c, h, w = x.shape
            t = x.flatten(2).transpose(1, 2)
            t = t + self.attn(*(self.n1(t),) * 3, need_weights=False)[0]
            t = t + self.mlp(self.n2(t))
            return t.transpose(1, 2).reshape(b, c, h, w)

    class Hybrid(nn.Module):
        """Arm B. Conv stem -> two attention blocks at 1/4 resolution -> upsample with a stem skip.
        The global residual means the identity is the easy function to learn, which is what keeps a
        restorer from damaging renders that already decoded."""

        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU())
            self.down = nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.ReLU(),
                                      nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.ReLU())
            self.blocks = nn.Sequential(Block(ch), Block(ch))
            self.up = nn.Sequential(nn.Conv2d(ch, ch * 4, 3, padding=1), nn.PixelShuffle(2),
                                    nn.ReLU(),
                                    nn.Conv2d(ch, ch * 4, 3, padding=1), nn.PixelShuffle(2),
                                    nn.ReLU())
            self.head = nn.Conv2d(ch * 2, 1, 3, padding=1)

        def forward(self, x):
            # /4 needs an input divisible by 4. The canvas is a multiple of 8 and the crop is 64,
            # so this pads only in the odd case rather than on every call.
            h, w = x.shape[-2:]
            ph, pw = (-h) % 4, (-w) % 4
            xp = nn.functional.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            s = self.stem(xp)
            y = self.up(self.blocks(self.down(s)))
            y = self.head(torch.cat([y, s], 1))
            return torch.sigmoid(y + _logit(xp))[..., :h, :w]

    class MDTA(nn.Module):
        """Arm C's attention: transposed, i.e. over CHANNELS. The attention matrix is C x C rather
        than HW x HW, so a global spatial context costs linear rather than quadratic in pixels --
        which is what makes full-resolution attention affordable on a 232 px canvas at all."""

        def __init__(self, dim, heads=4):
            super().__init__()
            self.heads = heads
            self.temp = nn.Parameter(torch.ones(heads, 1, 1))
            self.qkv = nn.Conv2d(dim, dim * 3, 1)
            self.dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3)
            self.proj = nn.Conv2d(dim, dim, 1)

        def forward(self, x):
            b, c, h, w = x.shape
            q, k, v = self.dw(self.qkv(x)).chunk(3, dim=1)
            q, k, v = (t.reshape(b, self.heads, c // self.heads, h * w) for t in (q, k, v))
            q, k = (nn.functional.normalize(t, dim=-1) for t in (q, k))
            a = (q @ k.transpose(-2, -1)) * self.temp
            out = (a.softmax(dim=-1) @ v).reshape(b, c, h, w)
            return self.proj(out)

    class GDFN(nn.Module):
        """Gated feed-forward: one branch gates the other, so the block can suppress a region
        instead of only adding to it -- the operation an overlay needs."""

        def __init__(self, dim, mult=2):
            super().__init__()
            hid = int(dim * mult)
            self.pw = nn.Conv2d(dim, hid * 2, 1)
            self.dw = nn.Conv2d(hid * 2, hid * 2, 3, padding=1, groups=hid * 2)
            self.out = nn.Conv2d(hid, dim, 1)

        def forward(self, x):
            a, b = self.dw(self.pw(x)).chunk(2, dim=1)
            return self.out(nn.functional.gelu(a) * b)

    class LN2d(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.n = nn.LayerNorm(dim)

        def forward(self, x):
            return self.n(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    class Restormer(nn.Module):
        """Arm C. Four MDTA+GDFN blocks at full resolution, global residual, no downsampling."""

        def __init__(self, depth=4):
            super().__init__()
            self.inp = nn.Conv2d(1, ch, 3, padding=1)
            self.blocks = nn.ModuleList()
            for _ in range(depth):
                self.blocks.append(nn.ModuleList([LN2d(ch), MDTA(ch), LN2d(ch), GDFN(ch)]))
            self.out = nn.Conv2d(ch, 1, 3, padding=1)

        def forward(self, x):
            y = self.inp(x)
            for n1, at, n2, ff in self.blocks:
                y = y + at(n1(y))
                y = y + ff(n2(y))
            return torch.sigmoid(self.out(y) + _logit(x))

    def _logit(x, eps=1e-4):
        """The global residual is taken in logit space, so that "predict zero" is exactly the
        identity after the sigmoid. Added in pixel space it would not be."""
        x = x.clamp(eps, 1 - eps)
        return torch.log(x / (1 - x))

    return {"conv": Restore, "hybrid": Hybrid, "restormer": Restormer}[arch]()


def save_png(arr: np.ndarray, path: str, binarize: bool) -> None:
    """One array at its native raster -> one file the decoders can open. Both the control and the
    restored arm come through here, so whatever this costs is charged to both."""
    a = (arr >= 0.5).astype(np.float32) if binarize else arr
    Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def restore_batch(model, torch, dev, imgs: Images, chunk: list, canvas: int, a_polarity=True):
    """Degraded -> (control array, restored array), both at the native size of each image."""
    pol = polarity if a_polarity else (lambda z: z)
    padded = (np.stack([pad_to(pol(imgs(r["image"])), canvas) for r, _ in chunk])
              .astype(np.float32) / 255)
    if model is None:
        out = padded
    else:
        with torch.no_grad():
            out = model(torch.tensor(padded).unsqueeze(1).to(dev)).squeeze(1).cpu().numpy()
    ctrl, rest = [], []
    for j, (r, _c) in enumerate(chunk):
        h, w = imgs(r["image"]).shape
        ctrl.append(padded[j, :h, :w])
        rest.append(out[j, :h, :w])
    return ctrl, rest


def evaluate(a, te: list, model, torch, decs: dict, imgs: Images, canvas: int) -> dict:
    """Decode all three arms over the held-out URLs and write one row per (image, decoder, arm)."""
    dev = next(model.parameters()).device
    rows_out = []
    dump = a.dump
    if dump:
        os.makedirs(dump, exist_ok=True)
    tmp = {k: tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
           for k in ("control", "restored")}
    t0, n, dump_rows = time.time(), 0, []
    for i in range(0, len(te), a.batch):
        chunk = te[i:i + a.batch]
        ctrl, rest = restore_batch(model, torch, dev, imgs, chunk, canvas, a.polarity == "auto")
        for j, (r, _clean) in enumerate(chunk):
            raw_path = os.path.join(a.dir, r["image"])
            save_png(ctrl[j], tmp["control"], a.binarize)
            save_png(rest[j], tmp["restored"], a.binarize)
            want = r["url_sha1"]
            base = {k: r.get(k, "") for k in ("sample_id", "url_sha1", "ec_level", "box_size",
                                              "transform", "strength", "calibration", "modules")}
            for arm, path in (("raw", raw_path), ("control", tmp["control"]),
                              ("restored", tmp["restored"])):
                if dump:
                    # ALL THREE arms, the raw one included. The comparison the dump exists for is
                    # restored+OpenCV against RAW+WeChatQR, so a dump without the raw arm would
                    # travel to a machine that cannot compute the thing it was sent to compute --
                    # and pairing it against a raw arm decoded here, on a different OpenCV build,
                    # would be comparing two machines. benchmark_qr.py reads an index.csv, so the
                    # arm rides in the sample_id and every file goes next to the others.
                    sub_dir = os.path.join(dump, r["url_sha1"][:2])
                    os.makedirs(sub_dir, exist_ok=True)
                    rel = os.path.join(r["url_sha1"][:2], f"{r['sample_id']}__{arm}.png")
                    if arm == "raw":
                        shutil.copyfile(raw_path, os.path.join(dump, rel))
                    else:
                        save_png(ctrl[j] if arm == "control" else rest[j],
                                 os.path.join(dump, rel), a.binarize)
                    dump_rows.append({**base, "sample_id": f"{r['sample_id']}__{arm}",
                                      "image": rel, "url_len": r.get("url_len", ""),
                                      "label": r.get("label", ""), "tld": r.get("tld", ""),
                                      "qr_version": r.get("qr_version", "")})
                for dname, dfn in decs.items():
                    got = dfn(path)
                    ok = any(hashlib.sha1(t.encode("utf-8", "replace")).hexdigest()[:16] == want
                             for t in got)
                    rows_out.append({**base, "arm": arm, "decoder": dname,
                                     "decoded": int(bool(got)), "correct": int(ok)})
            n += 1
            if n % 500 == 0:
                el = time.time() - t0
                print(f"    [{n:,}/{len(te):,}] {el:.0f}s, {n/max(el,1):.0f} img/s")
    for p in tmp.values():
        os.unlink(p)
    if dump:
        with open(os.path.join(dump, "index.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dump_rows[0].keys()))
            w.writeheader()
            w.writerows(dump_rows)
        print(f"[+] {dump}/  ({len(dump_rows):,} PNGs + index.csv; the arm is the sample_id "
              f"suffix after '__'). Decode there with benchmark_qr.py.")

    if not rows_out:
        # --dump-only: nothing was decoded here, so there is no eval CSV to write and no report to
        # make. Writing an empty one would leave a file that looks like a run with no findings.
        return {}
    os.makedirs(os.path.dirname(a.eval_out) or ".", exist_ok=True)
    with open(a.eval_out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"[+] {a.eval_out}  ({len(rows_out):,} rows)")
    return report(rows_out, a, list(decs))


def registered_tests(per_url, decs_l: list) -> dict:
    """PREREG_restore.md, T1 and T2. Both are cross-decoder and neither can be computed on a
    machine that lacks WeChatQR, which is the whole reason --dump exists.

    T1  (restorer + OpenCV) against RAW WeChatQR. If a defender can simply install a better decoder,
        a network that only matches it is not a contribution.
    T2  (restorer + WeChatQR) against raw WeChatQR: is there anything left to win once the strongest
        decoder is already in place?

    D is positive when the restorer is ahead, paired per URL, two-sided Wilcoxon, BH over m = 2."""
    if not {"opencv", "wechat"} <= set(decs_l):
        return {"registered": "not computable: needs opencv and wechat in the same run"}
    from scipy.stats import wilcoxon
    F = lambda k: 100 * (1 - per_url[k][1] / per_url[k][0]) if per_url.get(k) else float("nan")
    urls = sorted({k[2] for k in per_url})
    out, ps = {}, {}
    for name, (lo, hi) in (("T1", (("restored", "opencv"), ("raw", "wechat"))),
                           ("T2", (("restored", "wechat"), ("raw", "wechat")))):
        d = np.array([F((hi[0], hi[1], u)) - F((lo[0], lo[1], u)) for u in urls])
        d = d[~np.isnan(d)]
        p = float(wilcoxon(d)[1]) if np.any(d != 0) else 1.0
        mean = float(d.mean())
        verdict = "success" if mean >= 5 else "partial" if mean >= 0 else "negative"
        out[name] = {"n_urls": int(d.size), "mean_pp": mean, "median_pp": float(np.median(d)),
                     "p": p, "verdict": verdict,
                     "arms": f"{lo[0]}+{lo[1]} vs {hi[0]}+{hi[1]}"}
        ps[name] = p
    # Benjamini-Hochberg, m = 2 fixed at registration.
    order = sorted(ps, key=lambda k: ps[k])
    prev = 1.0
    for i, k in ((2, order[1]), (1, order[0])):
        prev = min(prev, ps[k] * 2 / i)
        out[k]["p_bh"] = prev
    print("\n=== REGISTERED (PREREG_restore.md): does the restorer beat the decoder you could "
          "just install? ===")
    for k in ("T1", "T2"):
        v = out[k]
        print(f"  {k}  {v['arms']:<32} n={v['n_urls']:<5} mean {v['mean_pp']:+6.1f}pp  "
              f"median {v['median_pp']:+6.1f}pp  p_BH={v['p_bh']:.3g}  -> {v['verdict'].upper()}")
    print("  Positive means the restorer is ahead. Registered: >= +5 pp success, 0 to +5 partial, "
          "below 0 negative\n  (and a negative T1 says a defender should install WeChatQR rather "
          "than train anything).")
    return {"registered": out}


def report(rows_out: list, a, decs_l: list) -> dict:
    """Every table this study reports, from a list of (image, arm, decoder, correct) rows --
    whichever machine produced them. The local run reaches it through evaluate(); a dump decoded on
    a host with all three decoders reaches it through --from-dfr, and both must be read the same
    way or the cross-machine comparison is not one."""
    import collections
    arms = ("raw", "control", "restored")
    dfr = collections.defaultdict(lambda: [0, 0])
    per_tr = collections.defaultdict(lambda: [0, 0])
    per_url = collections.defaultdict(lambda: [0, 0])
    for r in rows_out:
        for k, d in (((r["arm"], r["decoder"]), dfr),
                     ((r["arm"], r["decoder"], r["transform"]), per_tr),
                     ((r["arm"], r["decoder"], r["url_sha1"]), per_url)):
            d[k][0] += 1
            d[k][1] += r["correct"]
    F = lambda d, k: 100 * (1 - d[k][1] / d[k][0]) if d[k][0] else float("nan")

    print("\n=== DFR by arm (held-out URLs only) ===")
    pre = "polarity" if a.polarity == "auto" else "plumbing"
    print(f"  {'decoder':<9} " + " ".join(f"{x:>10}" for x in arms) + f"   {pre:>8}   network")
    res = {"by_arm": {}}
    for dn in decs_l:
        v = {arm: F(dfr, (arm, dn)) for arm in arms}
        print(f"  {dn:<9} " + " ".join(f"{v[x]:9.1f}%" for x in arms)
              + f"  {v['raw']-v['control']:+8.1f}pp {v['control']-v['restored']:+8.1f}pp")
        res["by_arm"][dn] = v
    if a.polarity == "auto":
        print("  'polarity' (raw - control) is what ONE LINE of preprocessing bought, before any "
              "network:\n  an image whose mean is dark is inverted. 'network' (control - restored) "
              "is the network's own\n  effect, measured against a control that already has the "
              "polarity fix. Positive is better.")
    else:
        print("  'plumbing' (raw - control) must be 0.0: padding and cropping are lossless, and a "
              "non-zero\n  value here means the arms differ by something other than the network. "
              "'network'\n  (control - restored) is the effect being measured; positive is "
              "better.")
        bad = [dn for dn in decs_l
               if abs(res["by_arm"][dn]["raw"] - res["by_arm"][dn]["control"]) > 1e-9]
        if bad:
            print(f"  [!] PLUMBING IS NOT NEUTRAL for {', '.join(bad)} — every number below is "
                  f"contaminated by it.")
        res["plumbing_neutral"] = not bad

    print("\n=== DFR by transform: control -> restored (the network's own effect) ===")
    trs = sorted({r["transform"] for r in rows_out})
    res["by_transform"] = {}
    for dn in decs_l:
        print(f"  [{dn}]")
        for tr in trs:
            c, s = F(per_tr, ("control", dn, tr)), F(per_tr, ("restored", dn, tr))
            note = ("   <- geometric, outside the model class" if tr in GEOMETRIC else
                    "   <- global polarity, outside the model class" if tr == "invert" else
                    "   <- the hard case (constraint 4)" if tr == "motion" else "")
            print(f"    {tr:<12} control {c:6.1f}%  restored {s:6.1f}%  {c-s:+6.1f}pp{note}")
            res["by_transform"].setdefault(dn, {})[tr] = {"control": c, "restored": s}

    # ---- what changed, image by image. A net gain hides two populations and a defender cares
    # which: codes the network rescued, and codes it destroyed that the control could read.
    print("\n=== Transitions (control -> restored), per decoder ===")
    idx = {(r["arm"], r["decoder"], r["sample_id"]): r["correct"] for r in rows_out}
    ids = sorted({r["sample_id"] for r in rows_out})
    res["transitions"] = {}
    for dn in decs_l:
        got = lambda arm, s: idx.get((arm, dn, s))
        fixed = sum(1 for s in ids if not got("control", s) and got("restored", s))
        broke = sum(1 for s in ids if got("control", s) and not got("restored", s))
        print(f"  {dn:<9} rescued {fixed:>6,}   destroyed {broke:>6,}   of {len(ids):,} renders")
        res["transitions"][dn] = {"rescued": fixed, "destroyed": broke, "n": len(ids)}

    # ---- constraint 2: the comparison that would be flattering, made honestly.
    if "opencv" in decs_l and "wechat" in decs_l:
        r_o, r_w = F(dfr, ("restored", "opencv")), F(dfr, ("raw", "wechat"))
        print(f"\n=== The baseline that matters: restored+OpenCV {r_o:.1f}%  vs  "
              f"raw+WeChatQR {r_w:.1f}%  ({r_w-r_o:+.1f}pp) ===")
        print("  A restorer only matters if it beats the strongest decoder someone could simply "
              "install instead.")
        res["vs_wechat"] = {"restored_opencv": r_o, "raw_wechat": r_w, "delta_pp": r_w - r_o}
    else:
        print("\n[i] Constraint 2's comparison needs WeChatQR, which is not installed here. "
              "Re-run with --dump\n    and decode the dump on the second Jetson, which has all "
              "three decoders.")

    # ---- paired per URL, because the unit of analysis is the URL and not the render.
    try:
        from scipy.stats import wilcoxon
        res["paired"] = {}
        print("\n=== Paired per URL (control vs restored), two-sided Wilcoxon ===")
        urls = sorted({r["url_sha1"] for r in rows_out})
        for dn in decs_l:
            d = np.array([F(per_url, ("control", dn, u)) - F(per_url, ("restored", dn, u))
                          for u in urls])
            p = float(wilcoxon(d)[1]) if np.any(d != 0) else 1.0
            print(f"  {dn:<9} n={len(urls):<5} mean {d.mean():+6.1f}pp  "
                  f"median {np.median(d):+6.1f}pp  p={p:.3g}")
            res["paired"][dn] = {"n_urls": len(urls), "mean_pp": float(d.mean()),
                                 "median_pp": float(np.median(d)), "p": p}
    except ImportError:
        print("\n[i] scipy absent; paired test skipped")

    res.update(registered_tests(per_url, decs_l))
    return res


def save_examples(a, te: list, model, torch, imgs: Images, canvas: int, path: str) -> None:
    """Four columns per row: degraded | control | restored | clean. A figure is not evidence, but a
    reader who cannot see what the network does to a QR cannot judge the tables either.

    The CONTROL column is there for the same reason the control arm is: without it the polarity fix
    reads as something the network did, and the inverted row would be the most impressive picture on
    the page for the wrong reason. One example per transform, each at the strongest setting it has,
    and each row cropped to its own images rather than to the global canvas -- a code is 130 px and
    the canvas is as wide as the longest URL's version, so padding every cell to the canvas is
    mostly white space."""
    dev = next(model.parameters()).device
    pick = []
    for tr in sorted({r["transform"] for r, _ in te}):
        cand = [p for p in te if p[0]["transform"] == tr]
        pick.append(max(cand, key=lambda p: float(p[0]["strength"])))
    ctrl, rest = restore_batch(model, torch, dev, imgs, pick, canvas, a.polarity == "auto")
    gut = 6
    rows = []
    for j, (r, c) in enumerate(pick):
        cells = [imgs(r["image"]), (ctrl[j] * 255).astype(np.uint8),
                 (rest[j] * 255).astype(np.uint8), imgs(c)]
        side = max(max(x.shape) for x in cells)
        row = np.full((side, len(cells) * (side + gut) - gut), WHITE, np.uint8)
        for k, cell in enumerate(cells):
            row[:cell.shape[0], k * (side + gut):k * (side + gut) + cell.shape[1]] = cell
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    grid = np.full((sum(r.shape[0] for r in rows) + gut * (len(rows) - 1), width), WHITE, np.uint8)
    y = 0
    for r in rows:
        grid[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + gut
    Image.fromarray(grid, mode="L").save(path)
    print(f"[+] {path}  (degraded | control | restored | clean; rows: "
          + ", ".join(f"{r['transform']}@{r['strength']}" for r, _ in pick) + ")")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("data", "raw", "qr_restore"))
    ap.add_argument("--arch", default="conv", choices=list(ARCHS),
                    help="model class; the one variable PREREG_restore_v2.md changes")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crop", type=int, default=64,
                    help="random crop, in pixels of the NATIVE raster; 0 trains on whole images")
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("data", "processed", "qr", "restore"))
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--decoders", default="opencv,pyzbar,wechat")
    ap.add_argument("--eval-limit", type=int, default=4000,
                    help="cap held-out renders decoded; decoding, not training, is the slow half")
    ap.add_argument("--eval-out", default=os.path.join("data", "processed", "qr",
                                                       "qr_restore_eval.csv"))
    ap.add_argument("--json", default=os.path.join("data", "processed", "qr",
                                                   "restore_snapshot.json"))
    ap.add_argument("--eval-only", action="store_true", help="load --out/restore.pt and evaluate")
    ap.add_argument("--no-eval", action="store_true", help="train and stop")
    ap.add_argument("--from-dfr", default="",
                    help="skip everything and report on a benchmark_qr.py CSV of a --dump, "
                         "decoded on a host that has all three decoders")
    ap.add_argument("--dump", default="",
                    help="also write the control and restored arms as PNGs + index.csv, for "
                         "decoding on a host that has all three decoders")
    # Written for PREREG_restore_v2.md. This workstation has OpenCV but neither PyZbar nor
    # WeChatQR, and T1 of that registration is an OpenCV comparison -- so decoding here "just to
    # check" would read half a confirmatory test before the registered run. --dump-only writes the
    # arms and decodes nothing, which is the only honest way to prepare a dump on this machine.
    ap.add_argument("--dump-only", action="store_true",
                    help="write the dump and decode NOTHING; the decode happens on the host that "
                         "has all three decoders")
    # Off by default: a threshold is a decision about the OUTPUT, and applying it only to the
    # restored arm would hand the network a free win the control never got. When on, both arms get
    # it, which is the only way the comparison stays about the network.
    # Default on, because with it off any set containing `invert` trains a network that emits
    # flat grey -- see polarity(). Off is kept because it is the strict neutrality check: with
    # --polarity off the control arm is the raw arm, bit for bit, and the run says so.
    ap.add_argument("--polarity", choices=["auto", "off"], default="auto",
                    help="normalise global polarity before BOTH the control and the network")
    ap.add_argument("--binarize", action="store_true",
                    help="threshold control AND restored at 0.5 before decoding")
    a = ap.parse_args()

    if a.from_dfr:
        # No torch, no images, no training: a dump was decoded elsewhere and this reads it back.
        # The arm is the sample_id suffix the dump wrote, and the URL is its first 16 hex --
        # benchmark_qr.py carries neither column, and neither needs to exist twice.
        with open(a.from_dfr, newline="", encoding="utf-8") as fh:
            rows = []
            for r in csv.DictReader(fh):
                sid, _, arm = r["sample_id"].rpartition("__")
                if arm not in ("raw", "control", "restored"):
                    print(f"[!] {a.from_dfr} is not a dump decode: sample_id {r['sample_id']!r} "
                          f"carries no arm suffix", file=sys.stderr)
                    return 1
                rows.append({**r, "sample_id": sid, "url_sha1": sid[:16], "arm": arm,
                             "correct": int(r["correct"]), "decoded": int(r["decoded"])})
        decs_l = sorted({r["decoder"] for r in rows})
        print(f"[*] {len(rows):,} rows from {a.from_dfr}; decoders {', '.join(decs_l)}; "
              f"{len({r['url_sha1'] for r in rows})} URLs")
        res = report(rows, a, decs_l)
        res.update({"source": a.from_dfr, "decoders": decs_l})
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, sort_keys=True)
        print(f"[+] {a.json}")
        return 0

    idx = os.path.join(a.dir, "index.csv")
    if not os.path.isfile(idx):
        print(f"[!] {idx} not found. Build a SAVED (not --stream) set first:\n"
              f"    python3 scripts/gen_synthetic_qr.py --n 200 --box 2 "
              f"--calibration modules \\\n"
              f"        --transforms blur,motion,saltpepper,contrast,invert,logo "
              f"--out {a.dir}", file=sys.stderr)
        return 1
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        print(f"[!] {e}. Neither Jetson has torch; run this on the workstation.", file=sys.stderr)
        return 1
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    with open(idx, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pairs = build_pairs(rows)
    if a.limit:
        pairs = pairs[:a.limit]
    present = sorted({r["transform"] for r, _ in pairs})
    geo = [t for t in present if t in GEOMETRIC]
    if geo:
        print(f"[!] the set contains {', '.join(geo)}: a convolution stack cannot invert a "
              f"rotation or a homography.\n    They are kept and reported, but a failure on them "
              f"is a statement about the model class, not about restoration.", file=sys.stderr)
    cal = {r.get("calibration", "") for r, _ in pairs}
    if cal <= {"px", ""} and {"blur", "motion"} & set(present):
        print("[!] this set carries the frozen px calibration, where blur and motion are 100% DFR "
              "at every\n    strength: the restorer cannot be shown to help on an input that "
              "carries no signal.\n    Regenerate with --calibration modules.", file=sys.stderr)

    tr = [p for p in pairs if split_key(p[0]["url_sha1"], a.holdout) == "train"]
    te = [p for p in pairs if split_key(p[0]["url_sha1"], a.holdout) == "test"]
    n_url_tr = len({p[0]["url_sha1"] for p in tr})
    n_url_te = len({p[0]["url_sha1"] for p in te})
    overlap = {p[0]["url_sha1"] for p in tr} & {p[0]["url_sha1"] for p in te}
    print(f"[*] {len(pairs):,} pairs; train {len(tr):,} ({n_url_tr} URLs), "
          f"test {len(te):,} ({n_url_te} URLs); transforms {', '.join(present)}")
    # Assert rather than trust: a leak here would make every number below meaningless, and it is
    # the single most likely way this goes wrong.
    assert not overlap, f"URL leak between splits: {len(overlap)} shared"
    print("[*] no URL appears in both splits")

    imgs = Images(a.dir)
    canvas = max(max(imgs(r["image"]).shape) for r, _ in pairs)
    canvas += (-canvas) % 8          # a multiple of 8, so a deeper model could stride later
    print(f"[*] native raster, canvas {canvas}x{canvas} px, crop {a.crop or canvas}")

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device {dev}")
    model = make_model(torch, nn, arch=a.arch).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[*] arch {a.arch}, {nparam:,} parameters")
    # One checkpoint per arch: the arms are compared against each other, so an arm that overwrote
    # another's weights would be comparing a model with itself and never say so.
    ckpt = os.path.join(a.out, "restore.pt" if a.arch == "conv" else f"restore_{a.arch}.pt")

    train_secs = 0.0
    if a.eval_only:
        if not os.path.isfile(ckpt):
            print(f"[!] {ckpt} not found; train first", file=sys.stderr)
            return 1
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        print(f"[*] loaded {ckpt}")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.L1Loss()
        t_train = time.time()
        order = np.arange(len(tr))
        for ep in range(1, a.epochs + 1):
            model.train()
            rng.shuffle(order)
            tot, nb, t0 = 0.0, 0, time.time()
            for i in range(0, len(order), a.batch):
                chunk = [tr[k] for k in order[i:i + a.batch]]
                xs, ys = [], []
                for r, c in chunk:
                    d, cl = imgs(r["image"]), imgs(c)
                    if a.polarity == "auto":
                        d = polarity(d)      # the target is already the right way up
                    # CROP INSIDE THE SYMBOL, not inside the canvas. A code is about 130 px at the
                    # wild render scale and the canvas is as wide as the longest URL's version --
                    # 232 px here -- so a window drawn uniformly over the canvas lands in white
                    # padding two times in three. Trained that way the network learns that white is
                    # usually the right answer, reaches an L1 of 0.04 by saying so, and returns an
                    # image no decoder can read. The crop is drawn from the image's own extent, and
                    # only images smaller than the window are padded at all.
                    k = a.crop
                    if k and (d.shape[0] > k or d.shape[1] > k):
                        y0 = int(rng.integers(0, max(1, d.shape[0] - k + 1)))
                        x0 = int(rng.integers(0, max(1, d.shape[1] - k + 1)))
                        d, cl = d[y0:y0 + k, x0:x0 + k], cl[y0:y0 + k, x0:x0 + k]
                    side = max(k, d.shape[0], d.shape[1]) if k else canvas
                    xs.append(pad_to(d, side))
                    ys.append(pad_to(cl, side))
                x = torch.tensor(np.stack(xs).astype(np.float32) / 255).unsqueeze(1).to(dev)
                y = torch.tensor(np.stack(ys).astype(np.float32) / 255).unsqueeze(1).to(dev)
                opt.zero_grad()
                loss = lossf(model(x), y)
                loss.backward()
                opt.step()
                tot += float(loss)
                nb += 1
            print(f"    epoch {ep}/{a.epochs}  L1 {tot/max(nb,1):.4f}  {time.time()-t0:.0f}s")
        train_secs = round(time.time() - t_train, 1)
        os.makedirs(a.out, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print(f"[+] {ckpt}")

    if a.no_eval:
        print("[i] --no-eval: the checkpoint is saved and nothing has been measured. "
              "L1 is not the question; DFR is.")
        return 0

    model.eval()
    os.makedirs(a.out, exist_ok=True)
    save_examples(a, te, model, torch, imgs, canvas, os.path.join(a.out, "examples.png"))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from benchmark_qr import load_decoders
    decs = load_decoders([] if a.dump_only
                         else [x.strip() for x in a.decoders.split(",") if x.strip()])
    if not decs and not a.dump_only:
        print("[!] no decoder available; the checkpoint is saved but unevaluated", file=sys.stderr)
        return 1
    # Deterministically thinned, not truncated: te is ordered by the index, so te[:N] would be the
    # first URLs and their first transforms rather than a cross-section of the held-out set.
    if a.eval_limit and len(te) > a.eval_limit:
        step = len(te) / a.eval_limit
        te = [te[int(i * step)] for i in range(a.eval_limit)]
    print(f"[*] evaluating {len(te):,} held-out renders x {len(decs)} decoder(s) x 3 arms: "
          f"{', '.join(decs)}")

    res = evaluate(a, te, model, torch, decs, imgs, canvas)
    if a.dump_only:
        print(f"[*] dump-only: nothing was decoded here. Next, on the host with all three "
              f"decoders:\n    python3 scripts/benchmark_qr.py --dir {a.dump} "
              f"--out data/processed/qr/{os.path.basename(a.dump)}_dfr.csv")
        return 0
    res.update({"arch": a.arch, "params": nparam, "train_seconds": train_secs,
                "dir": a.dir, "epochs": a.epochs, "crop": a.crop, "canvas": canvas,
                "polarity": a.polarity,
                "holdout": a.holdout, "binarize": bool(a.binarize), "eval_renders": len(te),
                "train_urls": n_url_tr, "test_urls": n_url_te, "transforms": present,
                "decoders": list(decs), "calibration": sorted(c for c in cal if c)})
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    with open(a.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)
    print(f"[+] {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
