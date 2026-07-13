.PHONY: install data url text eval test release clean
install:      ## install python deps
	pip install -r requirements.txt
data:         ## build processed dataset from data/raw
	python scripts/normalize_merge.py --raw data/raw --out data/processed
url:          ## train URL baseline
	python scripts/train_url_baseline.py --in data/processed/dataset_url.csv --out models/url_rf.joblib
text:         ## fine-tune PhoBERT on SMS content
	python scripts/train_text_phobert.py --in data/processed/dataset_sms.csv --out models/phobert_sms
eval:         ## robustness evaluation (needs a trained text model)
	python scripts/eval_robustness.py --model models/phobert_sms --data data/processed/dataset_sms.csv
test:         ## run unit tests
	pytest -q
release:      ## package a citable open-tier release (add PAGES=1 for the gated bundle)
	python scripts/make_release.py --version $(or $(VERSION),1.0.0) $(if $(PAGES),--include-pages,)
clean:
	rm -rf models data/processed/*.csv data/processed/splits/*.csv
