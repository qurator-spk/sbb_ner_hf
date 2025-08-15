# sbb_ner_hf

[![HuggingFace](https://img.shields.io/badge/-HuggingFace-FDEE21?logo=HuggingFace&logoColor=white&text=white)](https://huggingface.co/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](https://pandas.pydata.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](https://numpy.org/)

## Description
This tool aims at finetuning and evaluating Named Entity Recognition (NER) models for German historical newspaper contents with the help of the [HuggingFace Transformers](https://github.com/huggingface/transformers) library. It is implemented in such a way that different pretrained models from HuggingFace can be trained and tested with a variation of preprocessed and optionally combined datasets.

In its current state, this repository mostly serves the purpose of training and evaluating NER models; however, including inference and automatic annotation of input datasets would be feasible with the help of further developments as well. Also have a look at our previously developed, BERT-based solutions for entity recognition, disambiguation and linking [sbb_ner](https://github.com/qurator-spk/sbb_ner) and [sbb_ned](https://github.com/qurator-spk/sbb_ned).

* License: [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
* Related resources: ZEFYS2025 dataset on [Github](https://github.com/qurator-spk/ZEFYS2025) and Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15771823.svg)](https://doi.org/10.5281/zenodo.15771823) 

## Installation
* Install Python version 3.10.14, best to use a virtual environment for that (e.g. with [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)).
* Install the requirements from `requirements.txt` file.
* Clone this repo.

Overall, it would be beneficial to have GPU access (depending on the choice of models and parameters this may not be necessary though). 

## Usage

### Preprocessing

Most of the datasets that were used in this process needed additional preprocessing to be of the same dataset format. See below for more information, how the datasets were acquired and processed in this first step.

| dataset | preprocessing steps |
| --- |  --- |  
| [zefys2025](https://github.com/qurator-spk/sbb_ner_data) | run `preprocess_zefys2025.py` |
| [hisGermaNER](https://huggingface.co/datasets/stefan-it/HisGermaNER) | run `preprocess_hisgermaner.py` |
| [hipe2020](https://github.com/hipe-eval/HIPE-2022-data/tree/main/data/v2.1/hipe2020/de) | run `preprocess_hipe_hipe2020.py` |
| [neiss](https://github.com/NEISSproject/NERDatasets) (arendt, sturm) | run `preprocess_neiss.py` for each subset |
| [europeana](https://github.com/cneud/ner-corpora) (lft, onb) | transform into zefys2025 data format, run `preprocess_zefys2025.py` for each subset |
| [conll2003](eriktks/conll2003) | / (loaded directly via HF) |
| [germeval2014](GermanEval/germeval_14) | / (loaded directly via HF) | 

Each preprocessing script uses the [`datasets.Dataset.save_to_disk()`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.save_to_disk) function to save the `.hf` dataset including train/test/validation splits in Apache [Arrow](https://huggingface.co/docs/datasets/about_arrow) format for simple reloading ([`datasets.Dataset.load_from_disk()`](https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.load_from_disk)). There are three columns in a dataset: 
* `id`: identifier for each sentence in the dataset
* `tokens`: nested list of all the tokens per sentence
* `ner_tags`: nested list of all the NER tags per sentence

### Introduction
For a first/broad understanding of different functionalities included in [`config.py`](config.py), 
[`train.py`](train.py), [`merge_datasets.py`](merge_datasets.py) and [`eval_opt.py`](eval_opt.py), see [`main.ipynb`](). 
The [`token_classification.ipynb`](https://github.com/huggingface/notebooks/blob/main/examples/token_classification.ipynb) notebook from HuggingFace served as a starting point to the developments which 
can be found in these files. 
To run the code cells from `main.ipynb`, [Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/stable/) needs to be installed.

### Experiments
To be able to experiment with multiple training configurations at once, [`experiments.py`](experiments.py) and 
[`Makefile`](Makefile) were developed. Experimental results are saved as `.pkl` files and can then be accessed similar 
to [`experiments.ipynb`](experiments.ipynb).

[`experiment.py`](experiment.py) provides the following command line interface. How it has been used to obtain the
results published in the paper can be seen from the [`Makefile`](Makefile). The tables in the paper have been generated
by [`experiments.ipynb`](experiments.ipynb).
```
python experiment.py --help
Usage: experiment.py [OPTIONS] RESULT_FILE

Options:
  --max-epochs INTEGER            Maximum number of epochs to train. Default
                                  30.
  --exp-type [single|merged|historical|contemporary]
  --batch-size INTEGER            Can be supplied multiple times. Batch size
                                  to try.
  --learning-rate FLOAT           Can be supplied multiple times. Learning
                                  rate to try.
  --weight-decay FLOAT            Can be supplied multiple times. Weight decay
                                  to try.
  --warmup-step INTEGER           Can be supplied multiple times. Warmup steps
                                  to try.
  --use-data-config TEXT          Can be supplied multiple times. Run only on
                                  these training configs.
  --use-test-config TEXT          Can be supplied multiple times. Test each
                                  trained model on these configs.
  --pretrain-config-file PATH     Train on pretrained models defined in this
                                  result file (from a previous experiment.py
                                  run).
  --pretrain-path PATH            Load the pretrained models checkpoints
                                  (EXP_... directories) from this directory.
                                  Default './'
  --model-storage-path PATH       Store the models checkpoints (EXP_...
                                  directories) in this directory.
  --dry-run                       Dry run only. Do not train or test but just
                                  check if everything runs through.
  --help                          Show this message and exit.
```

## How to cite

### Dataset

```bibtex
@dataset{schneider_2025_15771823,
  author       = {Schneider, Sophie and
                  Förstel, Ulrike and
                  Labusch, Kai and
                  Lehmann, Jörg and
                  Neudecker, Clemens},
  title        = {ZEFYS2025: A German Dataset for Named Entity
                   Recognition and Entity Linking for Historical
                   Newspapers
                  },
  month        = jul,
  year         = 2025,
  publisher    = {Staatsbibliothek zu Berlin - Berlin State Library},
  version      = 1,
  doi          = {10.5281/zenodo.15771823},
  url          = {https://doi.org/10.5281/zenodo.15771823},
}

```

### Publication
[will be added soon]