# Walkthrough notebooks

Each folder reruns one paper's experiments step by step, prints the results, and compares each result file with
the one the paper reports, which ship in the folder's `expected/` directory. The explanations are in Vietnamese and the code in
English.

| Folder | Paper | Data |
|---|---|---|
| `sms/` | Vietnamese SMS phishing: text against URL | downloaded by the notebook from Hugging Face at a pinned revision, with PhoBERT |
| `qr/` | QR-code phishing in Vietnam | the study's data package on Zenodo, placed in `qr/qr_data/` |
| `suffix_blindspot/` | the `.vn` suffix blind spot of URL phishing detectors | downloaded by the notebook from the PhishVN deposit on Mendeley Data |
| `feed_measurement/` | measuring the phishing-intelligence feeds that cover Vietnam | downloaded by the notebook from the PhishVN deposit on Mendeley Data |
| `url_benchmark/` | what a phishing-URL benchmark measures: configuration, duplicates and corpus shift | PhishVN from Mendeley Data; the three external corpora from the study's Zenodo package, placed in `url_benchmark/url_data/` |
| `stream_construction/` | how the construction of a time-ordered stream sets the staleness gap of a phishing-URL detector | downloaded by the notebook from the PhishVN deposit on Mendeley Data, and four columns of PhreshPhish from Hugging Face at a pinned revision |
| `content_paraphrase/` | page-content fusion and paraphrase evasion of Vietnamese phishing detectors | the study's data package on Zenodo, placed in `content_paraphrase/content_data/`; models and the real-SMS corpus from Hugging Face at pinned revisions |

To run one: `cd sms` (or `qr`, `suffix_blindspot`, `feed_measurement`, `url_benchmark`, `content_paraphrase`, `stream_construction`), `pip install -r requirements.txt` with Python 3.12, open the notebook and run
all cells. The steps that train small neural networks reproduce exactly only on the pinned library versions.
