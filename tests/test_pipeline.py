"""Basic unit tests for the data pipeline (pure functions). Run: pytest -q"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
try:
    import _path
    _path.add_script_dirs()
except ImportError:  # flat public-mirror layout: scripts/ itself is already on sys.path
    pass
import normalize_merge as nm


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
    # Density is strictly-greater-than 0.008: 8 marked chars in 1000 is a miss, 9 is a hit. The
    # marked character has to be one the language gate accepts on its own -- these fixtures used
    # `ă` repeated, which is exactly the Romanian signature and now fails the gate before the
    # density is ever reached, so the boundary they exist to pin would have gone untested.
    assert not vf.is_vietnamese_text("a" * 992 + "ế" * 8)
    assert vf.is_vietnamese_text("a" * 991 + "ế" * 9)
    # and the gate is what rejects a single non-Vietnamese-exclusive mark, whatever its density
    assert not vf.is_vietnamese_text("a" * 900 + "ă" * 100)


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


def test_chongluadao_entry_re_tolerates_attributes_after_class():
    """The denylist page gained title="<url>" between the class value and the closing > in a
    late-July 2026 redeploy. A regex pinned to '">' silently matched nothing for three weeks while
    the page still returned 200, so both markup shapes are pinned here."""
    from watch_chongluadao import ENTRY_RE
    from fetch_phishing_feeds import CLD_ENTRY_RE

    current = ('<td class="_urlCell_kkxgc_119"><div class="_cellWrap_kkxgc_124"><span '
               'data-hk="000000010000000000031311" class="_urlText_kkxgc_131" '
               'title="http://facebook-page-login-boost-protect.ph">'
               'http://facebook-page-login-boost-protect.ph</span>')
    legacy = ('<span class="_urlText_ab12">https://vcb-verify.top/login</span>')

    for rx in (ENTRY_RE, CLD_ENTRY_RE):
        assert rx.findall(current) == ["http://facebook-page-login-boost-protect.ph"]
        assert rx.findall(legacy) == ["https://vcb-verify.top/login"]
        # footer/nav anchors sit outside the table cells and must stay excluded
        assert rx.findall('<a href="https://chongluadao.vn">home</a>') == []


def test_chongluadao_empty_parse_raises_instead_of_returning_empty():
    """An empty parse means the markup moved, not that the denylist emptied. Returning [] reads as
    success to both callers and is what hid the breakage; each fetcher must raise."""
    import pytest
    import fetch_phishing_feeds as ffp
    import watch_chongluadao as wcl

    class _Resp:
        text = "<html><body>no denylist table here</body></html>"

        def raise_for_status(self):
            return None

    ffp_get, wcl_get = ffp._get, wcl.requests.get
    ffp._get = lambda *a, **k: _Resp()
    wcl.requests.get = lambda *a, **k: _Resp()
    try:
        with pytest.raises(RuntimeError):
            ffp.src_chongluadao()
        with pytest.raises(RuntimeError):
            wcl.fetch_denylist_webpage()
    finally:
        ffp._get, wcl.requests.get = ffp_get, wcl_get


def test_host_of_drops_www():
    """Feeds list both www. and bare forms of the same site; keeping them apart doubled the
    ChongLuaDao arm (2,810 of its 5,826 rows were a www. twin) and spent scan quota re-capturing
    pages already held."""
    from vn_filter import host_of
    assert host_of("https://www.Example.COM:8443/a/b") == "example.com"
    assert host_of("http://x.vn/") == "x.vn"
    assert host_of("www.vietcombank-verify.top") == "vietcombank-verify.top"
    assert host_of("wwwx.vn") == "wwwx.vn"          # only the www. label, not any www prefix
    assert host_of("www.www.a.com") == "www.a.com"  # one label, so a doubled prefix still collapses


def test_is_vn_target_separators_and_idn():
    """Hyphenated brand/lure spellings and punycode were invisible to the name filter."""
    from vn_filter import is_vn_target, name_segments
    for d in ["vietin-bank.com", "ngan-hang-he-thong.com",
              "khach-hang-ca-nhan.site", "nang-cap-khach-hang.com"]:
        assert is_vn_target(d), d
    # punycode decodes, then loses diacritics, so unaccented tokens can match
    assert name_segments("xn--cngngitvit-xkb2932gjea6c.vn") == [["congdongitviet"], ["vn"]]
    assert is_vn_target("ng\u00e2nh\u00e0ng".encode("idna").decode() + ".com")
    # hyphen-dependent patterns in VN_TOKENS must survive the addition
    assert is_vn_target("homecredit-vn.xyz")


def test_brand_tokens_are_actually_loaded():
    """A mutation audit found this the one way to break the filter that the suite still passed:
    BRAND_TOKENS silently None costs ~6% of the admissions on a Vietnamese-targeting feed.

    Skipped where the token file is absent, which is exactly the public mirror: it ships
    vn_filter.py and excludes data/processed/, so the filter there runs on the hand-curated core
    alone and says so on stderr. That is the mirror's documented state, not a regression for its
    suite to fail on -- asserting here would have made `pytest -q` red on every clone. The guard
    still bites wherever the file exists, which is every tree that actually admits rows."""
    import vn_filter
    if not vn_filter.BRAND_TOKENS_PATH.exists():
        pytest.skip(f"brand tokens not exported here ({vn_filter.BRAND_TOKENS_PATH}); "
                    "the filter runs on the hand-curated core, as the mirror documents")
    assert vn_filter.BRAND_TOKENS is not None, "brand tokens did not load; see the stderr warning"
    assert vn_filter.BRAND_TOKENS.search("vietinbank-verify.top")


def test_host_of_parses_real_url_shapes():
    """host_of fed a scheme in caps, credentials, a port, an IPv6 literal or a bare query returned
    'https', 'user', '[2001' and the query string respectively."""
    from vn_filter import host_of
    assert host_of("HTTPS://Example.VN/path") == "example.vn"
    assert host_of("https://user:pw@host.com:443/p") == "host.com"
    assert host_of("https://user@vietcombank.com.vn/login") == "vietcombank.com.vn"
    assert host_of("http://[2001:db8::1]:8080/x") == "[2001:db8::1]"
    assert host_of("example.com?q=1") == "example.com"
    assert host_of("//example.vn/x") == "example.vn"
    assert host_of("sub.example.vn.") == "sub.example.vn"
    assert host_of("") == ""


def test_fused_match_requires_segment_alignment():
    """A token that lands inside the join is not a match. These are domains the live collector
    actually admitted on 2026-08-18 from the global feeds -- 33 spurious admissions in one hour,
    each spending a urlscan submission from the 40-per-run cap. Validating the widening against
    ChongLuaDao alone could not have caught any of them: every entry there is Vietnamese-targeting,
    so a false positive cannot appear."""
    from vn_filter import is_vn_target
    # The join between a label's tail and the TLD is a dot, and a dot is never crossed: these are
    # real admissions from 2026-08-18, IT hosts that matched `techcom` as `tech` + `com`.
    for d in ["dmg-tech.com", "rdp.dmg-tech.com", "poligone-tech.com", "quadro-tech.com",
              "wu-test.finovate-tech.com"]:
        assert not is_vn_target(d), d
    for d in ["app-net--coins-us.pages.dev",        # appnet, a registry token
              "app-nettflix.netlify.app",           # appnet
              "raghul-designer.github.io",          # ldesign, starts mid-segment
              "bel-design.ru",                      # ldesign
              "0q00vc-bq.myshopify.com",            # vcb across the join
              "likblnbvc-b1246e.ingress-florina.ewp.live",   # vcb
              "onlinebdo.c-ccd.workers.dev",        # cccd: aligned but 4 chars, below the floor
              "bright-kelpie-6cdb91.netlify.app",   # ghtk
              "codashope-eventi.duckdns.org",       # shopee across the join
              "rcifunqcsdivqsen-dot-millinium.ey.r.appspot.com"]:  # sendo
        assert not is_vn_target(d), d


def test_fused_match_keeps_split_brands_and_lures():
    """The case the rule exists for: a hyphen inserted INSIDE the brand or the phrase. A blunt
    length floor would drop the 6-character half of this list, which is most of the Vietnamese
    banks -- the alignment rule is what makes the short ones safe to keep."""
    from vn_filter import is_vn_target
    for d in ["mb-bank-verify.com", "tp-bank-secure.xyz", "sea-bank.vip", "lp-bank.top",
              "vietin-bank.com", "ngan-hang-he-thong.com", "khach-hang-ca-nhan.site",
              "nang-cap-khach-hang.com", "cong-thanh-toan-the24h.com"]:
        assert is_vn_target(d), d
    # Known and accepted losses: ncb-bank.pw matches only through `cbbank` sitting inside `ncbbank`,
    # which straddles the join, and tuoi-tre.com through a registry token the fused path excludes.
    for d in ["ncb-bank.pw", "tuoi-tre.com"]:
        assert not is_vn_target(d), d


def test_is_vn_target_does_not_collapse_across_labels():
    """Separators are stripped per label. Collapsing the whole host fuses a label's tail with the
    TLD -- 8kbetviet.com -> 8kbetvietcom matches 'vietcom' -- which on the 2026-08-18 ChongLuaDao
    mirror produced 88 hits that were mostly this artefact."""
    from vn_filter import is_vn_target
    for d in ["8kbetviet.com", "ablefinstech.com", "discord-app.net", "coopm.art"]:
        assert not is_vn_target(d), d


def test_is_vn_target_vn_only_brands():
    """Brands the registry extension never supplied. Global lenders with a VN branch stay out, by
    the same rule the registry guard uses: they attract worldwide, not VN-targeting, phishing."""
    from vn_filter import is_vn_target
    for d in ["seabank-online.com", "lpbank-verify.top", "baoviet-nhantho.com", "fecredit-vay.top"]:
        assert is_vn_target(d), d
    for d in ["mirae-asset.top", "homecredit-global.com", "google.com", "paypal-merchant.ru"]:
        assert not is_vn_target(d), d


def test_is_vietnamese_text_requires_vietnamese_exclusive_letters():
    """Density over VN_CHARS alone does not identify Vietnamese: the single-tone vowels are shared
    with French/Portuguese/Spanish. The dotPH parking page rendering its category list in a Romance
    language, and Chrome's localised error screen, both cleared the old threshold."""
    from vn_filter import is_vietnamese_text
    assert is_vietnamese_text("Vui lòng đăng nhập tài khoản ngân hàng để xác minh thông tin")
    assert not is_vietnamese_text(
        "Redirecting... Category Search This domain is available to be registered. "
        "Éducation Technologie et Informatique Notícias e Política Estilo e Moda")
    assert not is_vietnamese_text(
        "lhe.vn Ce site est inaccessible Vérifiez si l'adresse lhe.vn est correcte. "
        "DNS_PROBE_FINISHED_NXDOMAIN Actualiser")
    assert not is_vietnamese_text("Please sign in to your account to verify your information")
    assert not is_vietnamese_text("")


def test_is_vietnamese_text_rejects_the_other_diacritic_languages():
    """VN_ONLY_CHARS is not Vietnamese-only: a-breve is Romanian, d-stroke is Croatian and Serbian,
    and the dot-below vowels are Yoruba and Igbo. A localised browser error screen in any of those
    would pass exactly as the French one did. Two routes separate them -- a stacked letter, which
    no other language forms, or four distinct exclusive letters, which none of their inventories
    reaches."""
    from vn_filter import is_vietnamese_text
    assert is_vietnamese_text("Vui lòng đăng nhập tài khoản ngân hàng để xác minh")
    assert is_vietnamese_text("Cảnh báo: tài khoản của bạn sẽ bị khóa")      # no stacked letter
    for other in [
        "Această pagină nu funcționează. Verificați dacă adresa este scrisă corect",
        "Đakovo je grad u Slavoniji i sjedište Đakovačko-osječke nadbiskupije",
        "Ẹ kú àárọ̀ ọmọ mi ẹ ṣé púpọ̀ fún ìrànlọ́wọ́ yín lónìí àti fún gbogbo",
        "Ị bịara n ụlọ anyị taa maka ọrụ ndị a ga-eme n izu na-abịa daalụ",
    ]:
        assert not is_vietnamese_text(other), other[:40]


def test_is_vietnamese_text_reads_decomposed_vietnamese():
    """Vietnamese written with combining marks decomposes to bare Latin plus combining codepoints,
    none of which are in either character set, so a decomposed page scored as not Vietnamese."""
    import unicodedata
    from vn_filter import is_vietnamese_text
    vn = "Vui lòng đăng nhập tài khoản ngân hàng để xác minh thông tin"
    assert is_vietnamese_text(unicodedata.normalize("NFD", vn))
    assert is_vietnamese_text(unicodedata.normalize("NFC", vn))


def test_is_vn_target_tolerates_non_string_input():
    """A float NaN is what an empty pandas cell becomes, and it used to raise inside .lower(),
    crashing whatever loop was scanning a CSV."""
    from vn_filter import is_vn_target
    for bad in (float("nan"), None, 0, 3.14, [], {}):
        assert is_vn_target(bad) is False
    assert is_vn_target("abc.vn") is True


def test_not_content_catches_wildcard_parking_and_browser_errors():
    """Browser errors are matched on Chrome's error CODE: it survives localisation, and the captures
    arrive in French and Portuguese as often as English.

    build_content_manifest.py is the content channel's, so it stays out of the public mirror
    until that paper and its data are released; there the import is a miss, not a break."""
    m = pytest.importorskip("build_content_manifest",
                            reason="content-channel script, not exported to the public mirror")
    assert m.NOT_CONTENT.search("This domain is available to be registered. Click here to register.")
    assert m.NOT_CONTENT.search("Esta página não está a funcionar ERR_EMPTY_RESPONSE")
    assert m.NOT_CONTENT.search("Ce site est inaccessible DNS_PROBE_FINISHED_NXDOMAIN")
    assert m.NOT_CONTENT.search("Impossible de traiter cette demande HTTP ERROR 500")
    # a real Vietnamese login lure must not be swept up
    assert not m.NOT_CONTENT.search("Vui lòng đăng nhập tài khoản ngân hàng để xác minh")
