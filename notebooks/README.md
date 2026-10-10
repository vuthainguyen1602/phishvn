# Walkthrough notebooks

Each folder rebuilds one paper's results step by step and compares every number with the results the paper
prints, which ship in the folder's `expected/` directory. The explanations are in Vietnamese and the code in
English.

| Folder | Paper | Data |
|---|---|---|
| `sms/` | Vietnamese SMS phishing: text against URL | downloaded by the notebook from Hugging Face at a pinned revision, with PhoBERT |

To run one: `cd sms`, `pip install -r requirements.txt` with Python 3.12, open the notebook and run
all cells. The steps that train small neural networks reproduce exactly only on the pinned library versions.
