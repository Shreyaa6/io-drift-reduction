.PHONY: data prepare train evaluate plot test
PYTHON ?= python3

data:
	$(PYTHON) scripts/download_data.py
prepare:
	$(PYTHON) scripts/prepare.py
train:
	$(PYTHON) scripts/train.py
evaluate:
	$(PYTHON) scripts/evaluate.py
plot:
	$(PYTHON) scripts/plot_results.py
test:
	$(PYTHON) -m unittest discover -s tests -v
