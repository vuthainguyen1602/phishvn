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


def test_clean_title():
    from watch_chongluadao import clean_title
    assert clean_title("Ngân hàng \x00bảo mật\r\n") == "Ngân hàng  bảo mật"
    assert clean_title(None) == ""
    assert len(clean_title("x" * 999)) == 200


def test_brand_token_extraction():
    import build_brand_tokens as bbt
    assert bbt.domain_token("vietcombank.com.vn") == "vietcombank"
    assert bbt.domain_token("tiki.vn") == "tiki"
    assert bbt.domain_token("www.moet.gov.vn") == "moet"
    assert bbt.name_token("Công ty Cổ phần Thế Giới Di Động") == "thegioididong"
    assert bbt.token_mode("airlines", 3) is None       # deny list
    assert bbt.token_mode("tiki", 3) == "word"          # short -> boundary match
    assert bbt.token_mode("thegioididong", 3) == "substring"


def test_host_of_normalises():
    import vn_filter as vf
    assert vf.host_of("https://Sub.Example.VN:443/path?q=1") == "sub.example.vn"
    assert vf.host_of("http://tiki.vn./") == "tiki.vn"
    assert vf.host_of("vietcombank.com.vn/login") == "vietcombank.com.vn"
    assert vf.host_of("") == "" and vf.host_of(None) == ""


def test_is_vn_target_signals():
    import vn_filter as vf
    assert vf.is_vn_target("vksnghean.gov.vn")            # .vn TLD alone
    assert vf.is_vn_target("vietcombank-secure.top")      # unaccented brand token on foreign TLD
    assert vf.is_vn_target("dichvucong-online.xyz")       # service token
    assert vf.is_vn_target("tcb-login.com")               # \btcb\b bounded by the hyphen
    assert not vf.is_vn_target("atcb.com")                # boundary holds — no match inside a word
    assert not vf.is_vn_target("qqzzxx.top")
    assert not vf.is_vn_target("") and not vf.is_vn_target(None)


def test_is_vietnamese_text_threshold():
    import vn_filter as vf
    assert vf.is_vietnamese_text("Cảnh báo: tài khoản của bạn sẽ bị khóa")
    assert not vf.is_vietnamese_text("Your account will be suspended today")
    assert not vf.is_vietnamese_text("")
    # density is strictly-greater-than 0.008: 8 marked chars in 1000 is a miss, 9 is a hit
    assert not vf.is_vietnamese_text("a" * 992 + "ă" * 8)
    assert vf.is_vietnamese_text("a" * 991 + "ă" * 9)


def test_brand_tokens_extend_is_vn_target(tmp_path):
    import json
    import build_brand_tokens as bbt
    import vn_filter as vf

    rows = [{"org_name": "Tiki", "domain": "tiki.vn"},
            {"org_name": "Vietcombank", "domain": "vietcombank.com.vn"}]
    result = bbt.build(rows, min_len=3)
    assert [t["token"] for t in result["tokens"]] == ["tiki"]
    assert "vietcombank" in result["covered_by_static"]  # static VN_TOKENS has "vietcom"

    p = tmp_path / "brand_tokens.json"
    p.write_text(json.dumps(result), encoding="utf-8")
    rx = vf.load_brand_regex(p)
    assert rx.search("tiki-khuyenmai.top")
    assert not rx.search("batiki.com")                   # word boundary holds
