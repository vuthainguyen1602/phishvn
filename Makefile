.PHONY: install data url assets release verify clean
install:      ## install python deps
	pip install -r requirements.txt
data:         ## build the URL dataset from data/raw
	python scripts/normalize_merge.py --raw data/raw --out data/processed
url:          ## train URL baselines (multi-seed + bootstrap CI)
	python scripts/train_url_baseline.py --in data/processed/dataset_url.csv --out models/url_rf.joblib
assets:       ## regenerate the paper figure + tables from data
	python scripts/make_p1a_assets.py
release:      ## package the citable open-tier release (PAGES=1 for the gated bundle)
	python scripts/make_release.py --version $(or $(VERSION),1.0.0) $(if $(PAGES),--include-pages,)
verify:       ## run unit tests
	pytest -q
clean:
	rm -rf models data/processed/*.csv data/processed/splits/*.csv
