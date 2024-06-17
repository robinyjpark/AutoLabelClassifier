# system python interpreter. used only to create virtual environment
PY = python3
VENV = venv
BIN=$(VENV)/bin

$(VENV): requirements.txt setup.py
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade -r requirements.txt
	$(BIN)/pip install -e .

.PHONY: test
test: $(VENV) 
	$(BIN)/auto_label_classifier label --data example_data/example_reports.csv --condition stenosis --definition "Stenosis is any narrowing or compression of the spinal canal or nerves, including disc protrusions, impingement of nerve roots, or compromise of recesses." --device cuda:0 --output example_data/example_output.csv