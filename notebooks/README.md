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

To run one: `cd sms` (or `qr`, `suffix_blindspot`, `feed_measurement`), `pip install -r requirements.txt` with Python 3.12, open the notebook and run
all cells. The steps that train small neural networks reproduce exactly only on the pinned library versions.
