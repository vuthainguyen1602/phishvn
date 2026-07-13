"""Basic unit tests for the data pipeline (pure functions). Run: pytest -q"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import normalize_merge as nm  # noqa: E402


def test_url_features_suspicious_tld():
    f = nm.url_features("http://vcb-1.sub.xyz-login.top/verify")
    assert f["domain"] == "vcb-1.sub.xyz-login.top"
    assert f["tld"] == "top"
    assert f["suspicious_tld"] == 1
    assert f["has_ip"] == 0
    assert f["num_subdomains"] >= 1


def test_url_features_benign():
    f = nm.url_features("https://vksnghean.gov.vn")
    assert f["tld"] == "vn"
    assert f["suspicious_tld"] == 0


def test_redact_pii():
    t = nm.redact_pii("Lien he 0912345678, email a@b.com, TK 0123456789012")
    assert "<PHONE>" in t and "<EMAIL>" in t and "<NUM>" in t
    assert "0912345678" not in t


def test_map_scenario():
    assert nm.map_scenario("Ngân hàng Vietcombank") == "bank"
    assert nm.map_scenario("Bộ Công An") == "gov"
    assert nm.map_scenario("Shopee khuyến mãi") == "ecommerce"
    assert nm.map_scenario("Một tổ chức lạ") == "other"
