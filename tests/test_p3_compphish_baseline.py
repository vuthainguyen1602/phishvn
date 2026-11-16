"""CompPhish's released features as a P3 baseline (RESUBMISSION/PLAN.md, Amendment 7).

Run: pytest -q. The pins, the feature sets, the decoding and error rules and the per-fold label
means are pinned on hand-made rows. One test calls the real extractor and is skipped when the
pinned zip or its dependencies are not on this machine.
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
import p3_compphish_baseline as cb


def test_feature_sets_partition_the_seventy():
    assert (len(cb.CP_URL), len(cb.CP_HTML), len(cb.CP_ALL)) == (47, 23, 70)
    assert not set(cb.CP_URL) & set(cb.CP_HTML)
    assert len(set(cb.CP_ALL)) == 70
    assert set(cb.LABEL_MEAN) <= set(cb.CP_URL) and set(cb.LABEL_MEAN.values()) <= set(cb.CP_URL)


def test_a_file_with_the_wrong_hash_is_refused(tmp_path):
    p = tmp_path / "Mapping_File.xlsx"
    p.write_bytes(b"not the released file")
    with pytest.raises(SystemExit, match="not the pinned"):
        cb.check_pin(str(p))


def test_an_unpinned_file_is_refused(tmp_path):
    p = tmp_path / "something_else.zip"
    p.write_bytes(b"x")
    with pytest.raises(SystemExit, match="not a pinned"):
        cb.check_pin(str(p))


def test_decode_keeps_utf8_and_flags_the_rest():
    assert cb.decode("Ngân hàng".encode("utf-8")) == ("Ngân hàng", False)
    text, flagged = cb.decode(b"ab\xff")
    assert flagged and text.startswith("ab")


class _Fake:
    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)

    def page(self, serial, url, html):
        if serial in self.fail_on:
            raise RuntimeError("boom")
        return {k: len(url) if k == "url_length" else 0 for k in cb.CP_ALL}


def test_a_page_the_extractor_rejects_is_kept_empty_and_flagged():
    pages = [{"page_id": "a", "url": "http://a.vn"}, {"page_id": "b", "url": "http://b.vn"}]
    rows = cb.extract_pages(pages, lambda p: b"<html></html>", _Fake(fail_on={"b"}))
    assert rows[0]["extract_error"] == "" and rows[0]["url_length"] == len("http://a.vn")
    assert "boom" in rows[1]["extract_error"] and all(rows[1][k] is None for k in cb.CP_ALL)


def _rows(lengths):
    return [{"url_length": u, "domain_length": d} for u, d in lengths]


def test_fold_means_come_from_training_positives_only():
    rows = _rows([(10, 4), (30, 8), (100, 50), (1000, 900)])
    labels = ["abuse", "abuse", "legitimate", "abuse"]
    out = cb.fold_thresholds(rows, train_idx=[0, 1, 2], labels=labels)
    # mean over rows 0 and 1 only: url 20, domain 6. Row 3 (test) and row 2 (benign) do not count.
    assert [r["url_more_than_mean"] for r in out] == [0, 1, 1, 1]
    assert [r["domain_more_than_mean"] for r in out] == [0, 1, 1, 1]
    assert "url_more_than_mean" not in rows[0]                  # input untouched


def test_test_rows_cannot_move_the_threshold():
    labels = ["abuse", "abuse", "abuse"]
    a = cb.fold_thresholds(_rows([(10, 4), (30, 8), (25, 6)]), [0, 1], labels)
    b = cb.fold_thresholds(_rows([(10, 4), (30, 8), (99999, 99999)]), [0, 1], labels)
    assert a[0] == b[0] and a[1] == b[1]


def test_rows_without_features_stay_empty_and_no_positive_is_an_error():
    rows = _rows([(10, 4), (None, None)])
    out = cb.fold_thresholds(rows, [0, 1], ["abuse", "abuse"])
    assert out[1]["url_more_than_mean"] is None
    with pytest.raises(ValueError):
        cb.fold_thresholds(rows, [0], ["legitimate", "abuse"])


def test_real_extractor_returns_the_seventy_in_order():
    z = os.path.join(cb.CP_DIR, "feature extractor.zip")
    if not os.path.exists(z):
        pytest.skip("CompPhish version-4 extractor not downloaded")
    for mod in ("thefuzz", "Levenshtein", "bs4", "tldextract"):
        pytest.importorskip(mod)
    ex = cb.Extractor()
    with cb._cwd(ex.dir):
        f = ex.mod.extract_features(1, "https://a.example.com/x?y=1", 0, "<html></html>", 0, 0)
    assert [k for k in f if k not in ("serial_number", "url", "label")] == cb.CP_ALL
    row = ex.page(1, "https://a.example.com/x?y=1", "<form action='x'></form>")
    assert row["no_of_forms"] == 1 and row["no_of_questionmark"] == 1
