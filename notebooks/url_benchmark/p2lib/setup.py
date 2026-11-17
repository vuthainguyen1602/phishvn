"""Step 0 of every notebook: read the settings, download and verify the data, move into the work tree.

The paper's code reads and writes relative paths under data/processed/. `setup()` builds that tree
in url_work/ next to the notebooks (or PHISHVN_WORK_DIR) and makes it the working directory, so
every later step finds its inputs and the earlier notebooks' results where it expects them."""
import os, json, shutil, hashlib, zipfile, urllib.request
from pathlib import Path
from types import SimpleNamespace

from . import checks

MENDELEY_ZIP = ("https://data.mendeley.com/public-files/datasets/b97hxbxtpd/files/"
                "d5724d4d-9c2b-46b7-810e-cf290c3bfd87/file_downloaded")
ZIP_SHA256 = "308351e9d0c0fca13a81f2f63524a7dee08ccba6c775848d125b3da5525d0ab5"
PHISHVN_SHA256 = {"dataset_url.csv": "8e1c6325a82177014ac892a9aaf8c4d3cbbc986b47fd1ecac8899c865fb6eade",
                  "vn_compphish.csv": "ce64567f4ab955b2"}
ZENODO_DOI = "10.5281/zenodo.24301295"
ZENODO_RECORD = "24301295"  # the published data package; missing data files are downloaded from it
EXPECTED_URL = "https://raw.githubusercontent.com/vuthainguyen1602/phishvn/master/notebooks/url_benchmark/expected"

HOME = Path(__file__).resolve().parent.parent   # the folder that holds the notebooks
SETTINGS = ["PHISHVN_WORK_DIR", "PHISHVN_DATA_DIR", "PHISHVN_EXPECTED_DIR"]


def load_env_file(path):
    # KEY=value lines; a variable already set in the shell wins
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k not in os.environ:
                os.environ[k] = v
                loaded.append(k)
    return loaded


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fetch(url, dest):
    # Mendeley refuses Python's default user agent
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PhishVN walkthrough notebook)"})
    Path(dest).write_bytes(urllib.request.urlopen(req, timeout=300).read())


def setup():
    print("loaded from .env:", load_env_file(HOME / ".env") or "none")
    config = {k: os.environ.get(k, "") for k in SETTINGS}
    for k, v in config.items():
        print(f"{k:22s} {v or '(default)'}")

    work = Path(config["PHISHVN_WORK_DIR"] or HOME / "url_work")
    proc = work / "data" / "processed"
    (proc / "external").mkdir(parents=True, exist_ok=True)
    (proc / "p2").mkdir(parents=True, exist_ok=True)

    # PhishVN and its 21 URL features, from the Mendeley Data package
    zip_path = work / "PhishVN_v3.1.0_open.zip"
    if not zip_path.exists():
        fetch(MENDELEY_ZIP, zip_path)
    assert sha(zip_path) == ZIP_SHA256, "the downloaded package is not the published one"
    with zipfile.ZipFile(zip_path) as z:
        for name in PHISHVN_SHA256:
            (proc / name).write_bytes(z.read(f"data/{name}"))
    for name, digest in PHISHVN_SHA256.items():
        assert sha(proc / name).startswith(digest), name

    # the three external corpora, from the study's Zenodo package
    data = Path(config["PHISHVN_DATA_DIR"] or HOME / "url_data")
    if not (data / "SHA256SUMS").exists():
        assert ZENODO_RECORD, f"download the data package (DOI {ZENODO_DOI}) and unzip it into {data}"
        data.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(f"https://zenodo.org/api/records/{ZENODO_RECORD}",
                                         headers={"User-Agent": "Mozilla/5.0 (PhishVN walkthrough notebook)"})
            meta = json.loads(urllib.request.urlopen(req, timeout=120).read())
            for f in meta["files"]:
                fetch(f["links"]["self"], data / f["key"])
        except Exception:
            for fn in ["SHA256SUMS", "phiusiil_compphish.csv", "iscx_compphish.csv", "phishstorm_compphish.csv", "README.md"]:
                fetch(f"https://zenodo.org/records/{ZENODO_RECORD}/files/{fn}?download=1", data / fn)
    for line in (data / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert sha(data / name) == digest, f"{name} differs from the published package"
        if name.endswith("_compphish.csv"):
            shutil.copyfile(data / name, proc / "external" / name)

    # the result files the paper reports, to compare with
    expected = Path(config["PHISHVN_EXPECTED_DIR"] or HOME / "expected")
    if not (expected / "manifest.json").exists():
        expected = work / "expected"
        expected.mkdir(exist_ok=True)
        fetch(f"{EXPECTED_URL}/manifest.json", expected / "manifest.json")
        for f in json.loads((expected / "manifest.json").read_text()):
            (expected / f).parent.mkdir(parents=True, exist_ok=True)
            if not (expected / f).exists():
                fetch(f"{EXPECTED_URL}/{f}", expected / f)
    shutil.copyfile(expected / "first_seen_validation_summary.json", proc / "first_seen_validation_summary.json")

    os.chdir(work)
    os.environ["P2_UNWEIGHTED"] = "1"      # a few functions read the weighting switch from the environment
    checks.EXPECTED = expected
    checks.CHECKS.clear()
    print("working in", work, "| expected:", expected)
    return SimpleNamespace(home=HOME, work=work, expected=expected)


def require(*paths, notebook):
    # a notebook that reads an earlier notebook's results stops here if they are missing
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"run notebook {notebook} first: missing {', '.join(missing)}")
