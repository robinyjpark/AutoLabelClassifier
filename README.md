# Automated Spinal MRI Labelling from Reports Using a Large Language Model

![Labelling diagram](https://github.com/robinyjpark/AutoLabelClassifier/blob/main/label_process_simple.png)

See the [Project Page](https://www.robots.ox.ac.uk/~vgg/research/auto-report-labeller/).

This repository contains code for experiments in "Automated Spinal MRI Labelling from Reports Using a Large Language Model" (**MICCAI 2024 Spotlight**; see paper [here](https://www.robots.ox.ac.uk/~vgg/publications/2024/Park24/park24.pdf)). The data used are not publicly available and thus are not provided in this repository.

The code has been tested on python3.10 on ubuntu linux. If you use or adapt this software for your own work, please cite:

Park, R. Y., Windsor, R., Jamaludin, A., Zisserman, A. "Automated Spinal MRI Labelling from Reports Using a Large Language Model" In: Proceedings of 27th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2024.

bibtex:

```
@inproceedings{Park24,
   author    = {Robin Y. Park and Rhydian Windsor and Amir Jamaludin and Andrew Zisserman},
   booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
   title     = {Automated Spinal MRI Labelling from Reports Using a Large Language Model},
   year      = {2024},
   pages     = {101--111}
}
```

## Installation/Quick Start 

1. Build + activate python virtual environment to run the code in this repository:
`make venv; source venv/bin/activate`
Alternatively you can try to install the repo on top of an existing venv with `pip install .`

2. Try to run the example by `make test`. This will attempt to label sample reports in the `example_data` directory for spinal stenosis. Sample reports were sourced from [USARAD](https://usarad.com/sample-reports/sample-mri.html).

3. The sample reports table contains two columns: `pat_id` (patient ID) and `report`. The output file will add the following columns: `summary` (the LLM's summary of the report focused on the condition of interest), `probability` (normalised score for presence of condition) and `prediction` (yes or no, based on probability and threshold).

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
