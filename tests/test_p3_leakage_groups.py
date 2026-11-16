"""Leakage groups of the P3 resubmission (RESUBMISSION/PLAN.md "Splits and leakage", Amendment 6).

Run: pytest -q. The fingerprint, both kit-cluster methods, the key union and the three rules of the
size guard are pinned on hand-made pages and shingle sets.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
try:
    import _path
    _path.add_script_dirs()
except ImportError:  # flat public-mirror layout
    pass
import p3_leakage_groups as lg


def page(tags, text="x"):
    return "<html>" + "".join(f"<{t}>{text}</{t}>" if t != "input" else '<input type="password">'
                              for t in tags) + "</html>"


BODY = ["div", "p", "span", "a", "ul", "li", "img", "form", "input", "button"] * 3


def test_constants_are_the_amended_ones():
    assert (lg.SHINGLE_K, lg.MIN_TAGS, lg.TAU, lg.GUARD_SHARE) == (5, 25, 0.8, 0.20)


def test_stub_has_no_fingerprint():
    assert lg.fingerprint(page(["div"] * 10)) is None
    assert lg.fingerprint("") is None


def test_fingerprint_ignores_text_and_attribute_values():
    a = lg.fingerprint(page(BODY, "Ngân hàng A"))
    b = lg.fingerprint(page(BODY, "Bank B").replace("<div>", '<div class="z" id="q">'))
    assert a is not None and a == b


def test_input_type_is_part_of_the_structure():
    a = lg.fingerprint(page(BODY))
    b = lg.fingerprint(page(BODY).replace('type="password"', 'type="text"'))
    assert a != b


# Three shingle sets: A~B and B~C at Jaccard 9/11 >= 0.8, A~C at 8/12 < 0.8.
A, B, C = (frozenset(range(i, i + 10)) for i in (0, 1, 2))


def test_jaccard_of_the_chain():
    assert lg.jaccard(A, B) >= lg.TAU and lg.jaccard(B, C) >= lg.TAU and lg.jaccard(A, C) < lg.TAU


def test_components_chain_and_leader_does_not():
    assert lg.component_clusters([A, B, C]) == [0, 0, 0]
    lead = lg.leader_clusters([A, B, C])
    assert lead[0] == lead[1] and lead[2] != lead[0]


def test_leader_ties_go_to_the_earlier_leader():
    x, y = frozenset(range(0, 10)), frozenset(range(100, 110))
    probe = frozenset(range(0, 10))
    assert lg.leader_clusters([x, y, probe]) == [0, 1, 0]


def test_stubs_are_singletons_under_both_methods():
    for f in (lg.component_clusters, lg.leader_clusters):
        out = f([None, A, None, A])
        assert out[0] != out[2] and out[1] == out[3] and len(set(out)) == 3


def _pages(rows):
    return [{"page_id": pid, "key": key, "first_detected": fd, "scan_uuid": pid, "html": h}
            for pid, key, fd, h in rows]


def test_same_key_joins_pages_of_different_structure():
    other = ["table", "tr", "td", "th", "section"] * 6
    ps = _pages([("p1", "a.vn", "1", page(BODY)), ("p2", "a.vn", "2", page(other)),
                 ("p3", "b.vn", "3", page(other))])
    g = lg.assign_groups(ps, lambda p: p["html"])
    assert g[0]["group"] == g[1]["group"] == g[2]["group"]      # p2 links by key, p3 by kit


def test_groups_do_not_depend_on_input_order():
    other = ["table", "tr", "td", "th", "section"] * 6
    rows = [("p1", "a.vn", "1", page(BODY)), ("p2", "b.vn", "2", page(other)),
            ("p3", "c.vn", "3", page(BODY)), ("p4", "d.vn", "4", page(["div"] * 5))]
    fwd = {r["page_id"]: r["group"] for r in lg.assign_groups(_pages(rows), lambda p: p["html"])}
    rev = {r["page_id"]: r["group"] for r in lg.assign_groups(_pages(rows[::-1]), lambda p: p["html"])}
    assert fwd == rev
    assert fwd["p1"] == fwd["p3"] != fwd["p2"] and fwd["p4"] not in (fwd["p1"], fwd["p2"])


def test_unknown_method_is_refused():
    with pytest.raises(ValueError):
        lg.assign_groups(_pages([("p1", "a", "1", "")]), lambda p: p["html"], "single-link")


def _g(ids):
    return [{"group": i} for i in ids]


def test_guard_rule_1_components_when_nothing_is_oversized():
    labels = ["abuse"] * 10 + ["legitimate"] * 10
    comp = _g([0, 0] + list(range(1, 19)))                       # 2/10 = 20%, not over
    assert lg.primary_method(comp, comp, labels) == {"primary": "components", "rule": 1,
                                                     "drop_groups": []}


def test_guard_rule_2_leader_when_it_removes_the_oversized_group():
    labels = ["abuse"] * 10 + ["legitimate"] * 10
    comp = _g([0, 0, 0] + list(range(1, 18)))                    # 3/10 abuse in group 0
    lead = _g(list(range(20)))
    assert lg.primary_method(comp, lead, labels)["primary"] == "leader"


def test_guard_rule_3_drop_sensitivity_when_both_are_oversized():
    labels = ["abuse"] * 10 + ["legitimate"] * 10
    comp = _g(list(range(10)) + [50] * 5 + list(range(10, 15)))  # 5/10 legitimate in group 50
    out = lg.primary_method(comp, comp, labels)
    assert out == {"primary": "components", "rule": 3, "drop_groups": [50]}
