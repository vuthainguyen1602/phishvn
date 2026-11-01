"""
p2_supp.py — which of the URL benchmark's generated assets live in the Supplementary Material, and how text that
lives in the article refers to them (revision 1, 2026-10-31).

The revision moves detail tables out of the article. A generated paragraph that stays in the
article and says ``Table~\\ref{tab:maxf1}'' would print ``??'' once that table is only in
supplementary.tex, so every generated file destined for the article passes through
`article_refs`, which turns a reference to a supplementary label into ``Supplementary
Table~\\SuppRef{...}'' (numbers filled in from the supplement's own build). Files that are
themselves supplementary are left alone: inside supplementary.tex a plain \\ref resolves, and
references back into the article resolve through xr.
"""
from __future__ import annotations

import os
import re

# label -> the generated or hand-written file that carries it in supplementary.tex
SUPP = {
    "tab:maxf1": "tab_maxf1", "tab:priorcontrol": "tab_prior_control", "tab:confusion": "tab_confusion",
    "tab:strict": "tab_strict", "tab:tier-sensitivity": "tab_tier_sensitivity",
    "tab:labelnoise": "tab_label_noise", "tab:nowww": "tab_nowww", "tab:trivialfloor": "tab_trivial_floor",
    "tab:xdataset": "tab_xdataset", "tab:shap": "tab_shap", "tab:pruned": "tab_pruned",
    "tab:pruned_models": "tab_pruned_models", "tab:combined": "tab_combined",
    "tab:shiftmatrix": "tab_shiftmatrix", "tab:stacking": "tab_stacking", "tab:hpo": "tab_hpo",
    "tab:twosided": "tab_twosided", "tab:harmonisation": "tab_rev_harmonisation",
    "tab:coralcells": "tab_rev_coral", "tab:attrstab": "tab_rev_attr", "tab:hpoall": "tab_rev_hpoall",
    "fig:shiftmatrix": None, "fig:repairattempts": None, "fig:prauc_designs": None,
    "fig:prauc_cost": None, "alg:forecast": None,
}
SUPP_FILES = {v for v in SUPP.values() if v}
_KIND = {"tab": "Table", "fig": "Figure", "alg": "Algorithm"}


def article_refs(text: str, path: str) -> str:
    """Rewrite references to supplementary labels in a file that the article inputs."""
    if os.path.splitext(os.path.basename(path))[0] in SUPP_FILES:
        return text

    def one(m: re.Match) -> str:
        word, label = m.group(1), m.group(2)
        if label not in SUPP:
            return m.group(0)
        kind = _KIND[label.split(":")[0]]
        return f"Supplementary {kind}~\\SuppRef{{{label}}}"

    text = re.sub(r"(Table|Figure|Algorithm)~\\ref\{([^}]+)\}", one, text)
    # a bare \ref to a supplementary label (e.g. inside "Tables~\ref{a} and~\ref{b}")
    return re.sub(r"\\ref\{([^}]+)\}",
                  lambda m: f"\\SuppRef{{{m.group(1)}}}" if m.group(1) in SUPP else m.group(0), text)
