"""The comparison of each result file with the one the paper reports (`expected/`).

`EXPECTED` is set by `setup()`. Every check is appended to `CHECKS`; `save()` keeps a notebook's
checks under data/processed/p2/notebook_checks/ so the last notebook can add up all four."""
import json, math
from pathlib import Path

import numpy as np
import pandas as pd

RES = "data/processed/p2"
EXPECTED = Path("expected")
THR_TOL = 1e-3      # columns read at a threshold chosen on validation rows, or PR-AUC on small windows
ROUND4_TOL = 1.5e-4  # tables rounded to 4 decimals: one step in the last digit is 1e-4
CNN_TOL = 0.03      # networks the paper trained on Apple's GPU (MPS), refitted here on the CPU

CHECKS = []
TIME_COLS = {"fit_seconds", "seconds", "fit_s", "time_s", "elapsed_s", "wall_seconds", "fit_time"}
MET_TOL = 1e-4     # metrics of one run: parallel tree ensembles move the last digits, and one test row at
                   # the 0.5 threshold can flip; the paper prints three decimals
CURVE_TOL = 1e-4   # seed-mean precision on the 200-point recall grid, rounded to 6 decimals

def check(step, name, ok, detail=""):
    CHECKS.append({"step": step, "check": name, "detail": detail, "match": bool(ok)})
    return ok

def report(step):
    rows = [c for c in CHECKS if c["step"] == step]
    bad = [c for c in rows if not c["match"]]
    print(f"[{step}] {len(rows) - len(bad)}/{len(rows)} checks match")
    for c in rows:
        print(f"    {'ok      ' if c['match'] else 'MISMATCH'} {c['check']}: {c['detail']}")

def _num_diff(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    both = np.isnan(x) & np.isnan(y)
    one = np.isnan(x) ^ np.isnan(y)
    d = np.where(both, 0.0, np.abs(x - y))
    return (np.inf if one.any() else float(np.nanmax(d))) if d.size else 0.0

def check_table(step, name, tol, path=None, col_tol=None, shared_columns=False):
    # col_tol: a wider tolerance for named columns; shared_columns: compare on the columns both
    # tables have (for a saved table written before the code added a column)
    new = pd.read_csv(path or f"{RES}/{name}")
    old = pd.read_csv(EXPECTED / name)
    extra = sorted(set(new.columns) - set(old.columns))
    if shared_columns and extra and not set(old.columns) - set(new.columns):
        new = new[list(old.columns)]
    if set(new.columns) != set(old.columns) or len(new) != len(old):
        return check(step, name, False, f"shape/columns differ: {new.shape} {sorted(set(new.columns) ^ set(old.columns))} vs {old.shape}")
    cols = [c for c in old.columns if c not in TIME_COLS]
    num = [c for c in cols if pd.api.types.is_float_dtype(old[c])]
    key = [c for c in cols if c not in num]
    # stable sort: rows sharing a key (a PR curve's 200 points) keep their order
    a = new[cols].sort_values(key, kind="mergesort").reset_index(drop=True) if key else new[cols]
    b = old[cols].sort_values(key, kind="mergesort").reset_index(drop=True) if key else old[cols]
    if key and not a[key].astype(str).equals(b[key].astype(str)):
        return check(step, name, False, "row keys differ")
    col_tol = col_tol or {}
    plain = [c for c in num if c not in col_tol]
    d = _num_diff(a[plain].values, b[plain].values) if plain else 0.0
    ok = d <= tol
    detail = f"{len(new)} rows, max |difference| {d:.2g} (tolerance {tol:g})"
    for c, t in col_tol.items():
        dc = _num_diff(a[[c]].values, b[[c]].values)
        ok &= dc <= t
        detail += f"; {c} {dc:.2g} (tolerance {t:g})"
    if shared_columns and extra:
        detail += f"; columns only in the notebook's table: {extra}"
    return check(step, name, ok, detail)

def check_matrix(step, name, tol=0.0, path=None):
    new = pd.read_csv(path or f"{RES}/{name}", index_col=0)
    old = pd.read_csv(EXPECTED / name, index_col=0)
    if new.shape != old.shape or list(new.index) != list(old.index) or list(new.columns) != list(old.columns):
        return check(step, name, False, "shape or corpus order differs")
    d = _num_diff(new.values, old.values)
    return check(step, name, d <= tol + 1e-12, f"{new.shape[0]}x{new.shape[1]}, max |difference| {d:.2g} (tolerance {tol:g})")

def _plain(o):
    def conv(v):
        if hasattr(v, "item"):
            return v.item()
        if hasattr(v, "tolist"):
            return v.tolist()
        return str(v)
    return json.loads(json.dumps(o, sort_keys=True, default=conv, allow_nan=True))

def flat(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flat(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from flat(v, f"{prefix}[{i}]")
    else:
        yield prefix, d

def check_json(step, mine, saved_name, key=None, tol=0.0):
    saved = json.loads((EXPECTED / saved_name).read_text())
    new, ref = dict(flat(_plain(mine))), dict(flat(saved if key is None else saved[key]))
    keys = sorted(set(new) | set(ref))
    bad, worst = [], 0.0
    for k in keys:
        a, b = new.get(k, "<missing>"), ref.get(k, "<missing>")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            d = 0.0 if (a == b or (math.isnan(a) and math.isnan(b))) else abs(a - b)
            worst = max(worst, d)
            if not d <= tol:
                bad.append(k)
        elif a != b:
            bad.append(k)
    label = f"{saved_name}[{key}]" if key else saved_name
    check(step, label, not bad, f"{len(keys)} fields, {len(bad)} differ, max |difference| {worst:.2g} (tolerance {tol:g})")
    for k in bad[:8]:
        print(f"    differs: {k}: saved={ref.get(k, '<missing>')!r} notebook={new.get(k, '<missing>')!r}")

def save(notebook: str):
    out = Path(RES) / "notebook_checks"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(CHECKS, columns=["step", "check", "detail", "match"]).assign(notebook=notebook) \
        .to_csv(out / f"{notebook}.csv", index=False)


def summary(notebook: str):
    save(notebook)
    tbl = pd.DataFrame(CHECKS, columns=["step", "check", "detail", "match"])
    by_step = tbl.groupby("step", sort=False)["match"].agg(checks="size", matched="sum")
    print(by_step.to_string())
    print(f"total: {int(tbl['match'].sum())}/{len(tbl)} checks match")
    return by_step
