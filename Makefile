.PHONY: install data url benchmark assets release verify clean
install:      ## install python deps
	pip install -r requirements.txt
data:         ## build the URL dataset from data/raw
	python scripts/normalize_merge.py --raw data/raw --out data/processed
url:          ## train URL baselines (multi-seed + bootstrap CI)
	python scripts/train_url_baseline.py --in data/processed/dataset_url.csv --out models/url_rf.joblib
benchmark:    ## multi-protocol URL benchmark: both protocols, then the cross-corpus matrix
	python scripts/run_p2_benchmark.py
	python scripts/run_p2_temporal_strict.py
	# the three external corpora are featurised first with align_compphish.py (see its docstring)
	P2_UNWEIGHTED=1 python scripts/run_cross_dataset.py --corpora PhishVN=data/processed/vn_compphish.csv \
	  PhiUSIIL=data/processed/external/phiusiil_compphish.csv \
	  ISCXURL2016=data/processed/external/iscx_compphish.csv \
	  PhishStorm=data/processed/external/phishstorm_compphish.csv --seeds 5 \
	  --out data/processed/p2/cross_dataset_F1.csv
assets:       ## regenerate the paper figure + tables from data
	python scripts/make_p1_assets.py
release:      ## package the citable open-tier release (PAGES=1 for the gated bundle)
	python scripts/make_release.py --version $(or $(VERSION),1.0.0) $(if $(PAGES),--include-pages,)
verify:       ## run unit tests
	pytest -q
clean:
	rm -rf models data/processed/*.csv data/processed/splits/*.csv
