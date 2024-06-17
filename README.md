# Automated Spinal MRI Labelling from Reports Using a Large Language Model

This repository contains code for experiments in "Automated Spinal MRI Labelling from Reports Using a Large Language Model" (submitted to MICCAI 2024). The data used are not publicly available and thus are not provided in this repository.

The code has been tested on python3.10 on ubuntu linux. If you use or adapt this software for your own work, please cite:


## Installation/Quick Start 

1. Build + activate python virtual environment to run the code in this repository:
`make venv; source venv/bin/activate`
Alternatively you can try to install the repo on top of an existing venv with `pip install .`

2. Try to run the example by `make test`. This will attempt to label sample reports in the `example_data` directory.

## CLI Usage

The pipeline for labelling reports can be run from the command line using the `auto_label_classifier` command. More information
can be found by running `auto_label_classifier --help`, however a typical usage pattern would be:

```
auto_label_classifier label --data INPUT_REPORTS_CSV \ # path to csv with reports, should have two columns: 'report' contining a string of the report text and 'pat_id' containing a unique identifier for the report.
                            --output OUTPUT_CSV_PATH \ # path to save the output csv with the labels
                            --condition CONDITION_TO_LABEL_FOR \ # The condition one is trying to label, e.g. 'Stenosis'
                            --definition CONDITION_DEFINITION \ # A more-fine grained definition of the condition, e.g. 'Stenosis is any narrowing or compression of the spinal canal or nerves, including disc protrusions, impingement of nerve roots, or compromise of recesses.'
                            --model_name MODEL_NAME # The path to a model or Huggingface name, e.g. 'HuggingFaceH4/zephyr-7b-beta'
```

## Reproducing the paper

The experiments conducted for the paper are in the `scripts` and `notebooks` directories.

To-dos:
- [ ] Add citation + bibtex to README (important)
- [ ] Add interface to llama model (important)
- [ ] Add single stage option to report labelling pipeline
- [ ] CLI interface for instruction finetuning the model
- [ ] Add option to specify command line arguments by yaml file
