# import pandas as pd
import torch

import config
import train
import eval_opt
from merge_datasets import drop_ner_labels,merge_ds, get_label_list as get_merged_label_list
from datasets import Sequence, ClassLabel

import click
import os
import pandas as pd
import itertools
import hashlib

models = [
# #  {"path": "flair/ner-german", "add_prefix_space": False},
#     {"path": "dbmdz/electra-base-german-europeana-cased-discriminator",
#      "add_prefix_space": True, "ignore_mismatched_sizes": True},
    {"path": "dbmdz/bert-tiny-historic-multilingual-cased", "add_prefix_space": False},
    # {"path": "dbmdz/bert-mini-historic-multilingual-cased", "add_prefix_space": False},
    # {"path": "dbmdz/bert-base-german-cased", "add_prefix_space": False},
    # {"path": "FacebookAI/roberta-base", "add_prefix_space": True},
    # {"path": "FacebookAI/xlm-roberta-base", "add_prefix_space": True},
    # {"path": "deepset/gbert-base", "add_prefix_space": True},
    # {"path": "dbmdz/bert-base-historic-multilingual-cased", "add_prefix_space": False},
    # {"path": "distilbert/distilbert-base-uncased", "add_prefix_space": False}
]

dataset_defs = [
    {"name": "hipe2020", "path": "data/hipe2020_20250415.hf", "source": "local"},
    {"name": "hipe2020-nc", "path": "data/hipe2020_not-casted.hf", "source": "local"},

    {"name": "zefys2025", "path": "data/zefys2025_20250404.hf", "source": "local"},
    {"name": "zefys2025-nc", "path": "data/zefys2025_not-casted.hf", "source": "local"},
    {"name": "zefys2025-nc-wls", "path": "data/zefys2025_with_last_sents.hf", "source": "local"},

    {"name": "newseye", "path": "data/newseye_20250404.hf", "source": "local"},

    {"name": "hisgerman", "path": "data/hisgermaner_20250404.hf", "source": "local"},
    {"name": "hisgerman-nc", "path": "data/hisgermaner_not-casted.hf", "source": "local"},

    {"name": "conll2003", "path": "eriktks/conll2003", "source": "hf"},

    {"name": "GermEval",  "path": "GermanEval/germeval_14", "source": "hf"},

    {"name": "europeana-lft", "path": "data/europeana_lft.hf", "source": "local"},
    {"name": "europeana-onb", "path": "data/europeana_onb.hf", "source": "local"},
    {"name": "neiss-arendt", "path": "data/neiss_arendt.hf", "source": "local"},
    {"name": "neiss-sturm", "path": "data/neiss_sturm.hf", "source": "local"}
]

data_configs_single = [
    {"train": {"name": "hipe2020", "def": ["hipe2020-nc"]},
     "test": [{"name": "hipe2020", "def": ["hipe2020-nc"]}]},

    {"train": {"name": "zefys2025", "def": ["zefys2025-nc-wls"]},
     "test": [{"name": "zefys2025", "def": ["zefys2025-nc-wls"]}]},

    # {"train": {"name": "newseye", "def": ["newseye"]}, "test": [{"name": "newseye", "def": ["newseye"]}]},

    {"train": {"name": "hisgerman", "def": ["hisgerman-nc"]}, "test": [{"name": "hisgerman", "def": ["hisgerman-nc"]}]},

    # {"train": {"name": "conll2003", "def": ["conll2003"]}, "test": [{"name": "conll2003", "def": ["conll2003"]}]},

    # {"train": {"name": "GermEval", "def": ["GermEval"]}, "test": [{"name": "GermEval", "def": ["GermEval"]}]},

    {"train": {"name": "europeana-lft", "def": ["europeana-lft"]},
     "test": [{"name": "europeana-lft", "def": ["europeana-lft"]}]},

    {"train": {"name": "europeana-onb", "def": ["europeana-onb"]},
     "test": [{"name": "europeana-onb", "def": ["europeana-onb"]}]},

    {"train": {"name": "neiss-arendt", "def": ["neiss-arendt"]},
     "test": [{"name": "neiss-arendt", "def": ["neiss-arendt"]}]},

    {"train": {"name": "neiss-sturm", "def": ["neiss-sturm"]},
     "test": [{"name": "neiss-sturm", "def": ["neiss-sturm"]}]},
]

data_configs_merged = [
    # {
    #     "train": {"name": "hipe2020+zefys2025", "def": ["hipe2020-nc", "zefys2025-nc"]},
    #     "test": [{"name": "hipe2020", "def": ["hipe2020-nc"]},
    #              {"name": "zefys2025", "def": ["zefys2025-nc-wls"]}]
    # },
    # {
    #     "train": {"name": "hisgerman+zefys2025", "def": ["hisgerman-nc", "zefys2025-nc-wls"]},
    #     "test": [{"name": "hisgerman", "def": ["hisgerman-nc"]},
    #              {"name": "zefys2025", "def": ["zefys2025-nc-wls"]}]
    # },
    # # {
    # #     "train": {"name": "hisgerman+hipe2020", "def": ["hisgerman-nc", "hipe2020-nc"]},
    # #     "test": [{"name": "hisgerman", "def": ["hisgerman-nc"]},
    # #              {"name": "hipe2020", "def": ["hipe2020-nc"]}]
    # # },
    # {
    #     "train": {"name": "europeana-lft+zefys2025", "def": ["europeana-lft", "zefys2025-nc-wls"]},
    #     "test": [{"name": "europeana-lft", "def": ["europeana-lft"]},
    #              {"name": "zefys2025", "def": ["zefys2025-nc-wls"]}]
    # },
    # {
    #     "train": {"name": "europeana-onb+zefys2025", "def": ["europeana-onb", "zefys2025-nc-wls"]},
    #     "test": [{"name": "europeana-onb", "def": ["europeana-onb"]},
    #              {"name": "zefys2025", "def": ["zefys2025-nc-wls"]}]
    # },
    # {
    #     "train": {"name": "neiss-arendt+zefys2025", "def": ["neiss-arendt", "zefys2025-nc-wls"]},
    #     "test": [{"name": "neiss-arendt", "def": ["neiss-arendt"]},
    #              {"name": "zefys2025", "def": ["zefys2025-nc-wls"]}]
    # },
    # {
    #     "train": {"name": "neiss-sturm+zefys2025", "def": ["neiss-sturm", "zefys2025-nc-wls"]},
    #     "test": [{"name": "neiss-sturm", "def": ["neiss-sturm"]},
    #              {"name": "zefys2025", "def": ["zefys2025-nc-wls"]}]
    # },
    {
        "train": {"name": "all-historic",
                  "def": ["neiss-sturm", "neiss-arendt", "europeana-onb", "europeana-lft", "hisgerman-nc",
                          "hipe2020-nc", "zefys2025-nc-wls"]},
        "test": [{"name": "europeana-lft", "def": ["europeana-lft"]},
                 {"name": "europeana-onb", "def": ["europeana-onb"]},
                 {"name": "neiss-arendt", "def": ["neiss-arendt"]},
                 {"name": "neiss-sturm", "def": ["neiss-sturm"]},
                 {"name": "hipe2020", "def": ["hipe2020-nc"]},
                 {"name": "hisgerman", "def": ["hisgerman-nc"]},
                 {"name": "zefys2025", "def": ["zefys2025-nc-wls"]}]
    }
]

data_configs = data_configs_merged

batch_sizes = [32]
learning_rates = [2e-5]
# batch_sizes = [32, 64, 96]
# learning_rates = [2e-5, 1e-4, 1e-3]
weight_decays = [0.01]
warmup_steps = [100]


@click.command()
@click.argument('result-file', type=click.Path())
@click.option('--max-epochs', type=int, default=1)
def main(result_file, max_epochs):

    results = None
    if os.path.exists(result_file):
        results = pd.read_pickle(result_file)

    train.set_torch_device()

    for model_def, data_config, batch_size, learning_rate, weight_decay, warmup_step in (
            itertools.product(models, data_configs, batch_sizes, learning_rates, weight_decays, warmup_steps)):

        try:
            train_params = config.TrainingParams()

            train_params.batch_size = batch_size
            train_params.learning_rate = learning_rate
            train_params.weight_decay = weight_decay
            train_params.warmup_steps = warmup_step
            train_params.num_train_epochs = max_epochs

            exp_ID = "EXP_" + hashlib.sha256((str(model_def) + str(data_config) + str(train_params)).encode()).hexdigest()

            # noinspection PyTypeChecker
            if results is not None and sum(results.exp_ID == exp_ID) > 0:
                print("Skipping {} - experiment already exists.".format(exp_ID))
                continue

            model_config = config.Resource(path=model_def["path"], source="hf")
            model_config.set_name()
            print(model_config.info())

            tokenizer = train.get_tokenizer(model_def["path"], model_def["add_prefix_space"],
                                            ignore_mismatched_sizes=model_def["ignore_mismatched_sizes"]
                                            if "ignore_mismatched_sizes" in model_def else False)

            train_config = data_config["train"]
            test_configs = data_config["test"]

            train_tokenized_data, train_label_list = load_dataset_config(train_config, tokenizer)

            trained_ner_model, model_out_path, best_result = (
                train.train_model(model_config, train_config["name"], train_label_list, train_params,
                                  train_tokenized_data, tokenizer, save_strategy="epoch", exp_model_path=exp_ID))

            for test_config in test_configs:

                test_tokenized_data, _ = load_dataset_config(test_config, tokenizer, train_label_list)

                class_report, errors = eval_opt.compute_metrics_per_tag(trained_ner_model, test_tokenized_data,
                                                                        train_label_list,
                                                                        output_dict=True)

                result = process_report(best_result.copy(), class_report)

                result['test'] = test_config["name"]
                result["exp_ID"] = exp_ID

                if results is None:
                    results = pd.DataFrame([result])
                else:
                    results = pd.concat([results, pd.DataFrame([result])])

            results.to_pickle(result_file)
        except torch.OutOfMemoryError as e:
            print("Out of memory.")

    results.to_pickle(result_file)


def load_dataset_config(data_config, tokenizer, label_list=None):
    datasets = []
    for dataset_def in data_config["def"]:

        dataset_def = get_dataset_def(dataset_def)

        dataset_config = config.Resource(path=dataset_def["path"], source=dataset_def["source"])
        dataset_config.set_name()

        print(dataset_config.info())

        datasets.append(train.load_ner_dataset(dataset_config.path, dataset_config.source))

    merged_dataset = merge_ds(datasets)
    merged_label_list = get_merged_label_list(merged_dataset["train"]["ner_tags"])

    merged_dataset = (merged_dataset.
                      cast_column("ner_tags",
                                  Sequence(ClassLabel(names=merged_label_list))))

    merged_dataset, merged_label_list = drop_ner_labels(merged_label_list, merged_dataset)

    # import ipdb;ipdb.set_trace()

    tokenized_dataset = train.prepare_dataset(merged_dataset, tokenizer)

    return tokenized_dataset, merged_label_list if label_list is None else label_list


def get_dataset_def(dataset_def):

    for d in dataset_defs:
        if d["name"] == dataset_def:
            return d

    raise RuntimeError("Unknown dataset definition: {}.".format(dataset_def))


def process_report(best_result, class_report):
    total_support = 0.0
    best_result['f1_test'] = 0.0
    best_result['precision_test'] = 0.0
    best_result['recall_test'] = 0.0

    if 'PER' in class_report:
        total_support += class_report['PER']['support']
    if 'LOC' in class_report:
        total_support += class_report['LOC']['support']
    if 'ORG' in class_report:
        total_support += class_report['ORG']['support']

    if 'PER' in class_report:
        weight = class_report['PER']['support'] / total_support
        best_result['f1_test'] += weight * class_report['PER']['f1-score']
        best_result['precision_test'] += weight * class_report['PER']['precision']
        best_result['recall_test'] += weight * class_report['PER']['recall']

        best_result['PER_f1_test'] = class_report['PER']['f1-score']
        best_result['PER_precision_test'] = class_report['PER']['precision']
        best_result['PER_recall_test'] = class_report['PER']['recall']

    if 'LOC' in class_report:
        weight = class_report['LOC']['support'] / total_support
        best_result['f1_test'] += weight * class_report['LOC']['f1-score']
        best_result['precision_test'] += weight * class_report['LOC']['precision']
        best_result['recall_test'] += weight * class_report['LOC']['recall']

        best_result['LOC_f1_test'] = class_report['LOC']['f1-score']
        best_result['LOC_precision_test'] = class_report['LOC']['precision']
        best_result['LOC_recall_test'] = class_report['LOC']['recall']

    if 'ORG' in class_report:
        weight = class_report['ORG']['support'] / total_support
        best_result['f1_test'] += weight * class_report['ORG']['f1-score']
        best_result['precision_test'] += weight * class_report['ORG']['precision']
        best_result['recall_test'] += weight * class_report['ORG']['recall']

        best_result['ORG_f1_test'] = class_report['ORG']['f1-score']
        best_result['ORG_precision_test'] = class_report['ORG']['precision']
        best_result['ORG_recall_test'] = class_report['ORG']['recall']

    return best_result


if __name__ == '__main__':
    main()
