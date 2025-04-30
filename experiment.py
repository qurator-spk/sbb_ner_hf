# import pandas as pd
import torch
# from pygments.lexer import default

import config
import train
import eval_opt
from merge_datasets import map_ner_tags_to_zefys, merge_ds, zefys_label_list  # get_label_list as get_merged_label_list
# from datasets import Sequence, ClassLabel

import click
import os
import pandas as pd
import itertools
import hashlib

published_models = [
    {"path": "dbmdz/electra-base-german-europeana-cased-discriminator",
     "add_prefix_space": True, "ignore_mismatched_sizes": True},
    {"path": "dbmdz/bert-tiny-historic-multilingual-cased", "add_prefix_space": False},
    {"path": "dbmdz/bert-mini-historic-multilingual-cased", "add_prefix_space": False},
    {"path": "dbmdz/bert-base-german-cased", "add_prefix_space": False},
    {"path": "FacebookAI/roberta-base", "add_prefix_space": True},
    {"path": "FacebookAI/xlm-roberta-base", "add_prefix_space": True},
    {"path": "deepset/gbert-base", "add_prefix_space": True},
    {"path": "dbmdz/bert-base-historic-multilingual-cased", "add_prefix_space": False},
    {"path": "distilbert/distilbert-base-uncased", "add_prefix_space": False}
]

named_published_models = {
    "electra-base-german-europeana-cased-discriminator" :
        {"path": "dbmdz/electra-base-german-europeana-cased-discriminator",
         "add_prefix_space": True, "ignore_mismatched_sizes": True},
    "bert-tiny-historic-multilingual-cased":
        {"path": "dbmdz/bert-tiny-historic-multilingual-cased", "add_prefix_space": False},
    "bert-mini-historic-multilingual-cased":
        {"path": "dbmdz/bert-mini-historic-multilingual-cased", "add_prefix_space": False},
    "bert-base-german-cased":
        {"path": "dbmdz/bert-base-german-cased", "add_prefix_space": False},
    "roberta-base":
        {"path": "FacebookAI/roberta-base", "add_prefix_space": True},
    "xlm-roberta-base":
        {"path": "FacebookAI/xlm-roberta-base", "add_prefix_space": True},
    "gbert-base": {"path": "deepset/gbert-base", "add_prefix_space": True},
    "bert-base-historic-multilingual-cased":
        {"path": "dbmdz/bert-base-historic-multilingual-cased", "add_prefix_space": False},
    "distilbert-base-uncased":
        {"path": "distilbert/distilbert-base-uncased", "add_prefix_space": False}
}

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

    {"train": {"name": "newseye", "def": ["newseye"]}, "test": [{"name": "newseye", "def": ["newseye"]}]},

    {"train": {"name": "hisgerman", "def": ["hisgerman-nc"]}, "test": [{"name": "hisgerman", "def": ["hisgerman-nc"]}]},

    {"train": {"name": "conll2003", "def": ["conll2003"]}, "test": [{"name": "conll2003", "def": ["conll2003"]}]},

    {"train": {"name": "GermEval", "def": ["GermEval"]}, "test": [{"name": "GermEval", "def": ["GermEval"]}]},

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
    {
        "train": {"name": "hipe2020+zefys2025", "def": ["hipe2020-nc", "zefys2025-nc"]},
        "test": [{"name": "hipe2020", "def": ["hipe2020-nc"]},
                 {"name": "zefys2025", "def": ["zefys2025-nc-wls"]}]
    },
    {
        "train": {"name": "hisgerman+zefys2025", "def": ["hisgerman-nc", "zefys2025-nc-wls"]},
        "test": [{"name": "hisgerman", "def": ["hisgerman-nc"]},
                 {"name": "zefys2025", "def": ["zefys2025-nc-wls"]}]
    },
    # {
    #     "train": {"name": "hisgerman+hipe2020", "def": ["hisgerman-nc", "hipe2020-nc"]},
    #     "test": [{"name": "hisgerman", "def": ["hisgerman-nc"]},
    #              {"name": "hipe2020", "def": ["hipe2020-nc"]}]
    # },
    {
        "train": {"name": "europeana-lft+zefys2025", "def": ["europeana-lft", "zefys2025-nc-wls"]},
        "test": [{"name": "europeana-lft", "def": ["europeana-lft"]},
                 {"name": "zefys2025", "def": ["zefys2025-nc-wls"]}]
    },
    {
        "train": {"name": "europeana-onb+zefys2025", "def": ["europeana-onb", "zefys2025-nc-wls"]},
        "test": [{"name": "europeana-onb", "def": ["europeana-onb"]},
                 {"name": "zefys2025", "def": ["zefys2025-nc-wls"]}]
    },
    {
        "train": {"name": "neiss-arendt+zefys2025", "def": ["neiss-arendt", "zefys2025-nc-wls"]},
        "test": [{"name": "neiss-arendt", "def": ["neiss-arendt"]},
                 {"name": "zefys2025", "def": ["zefys2025-nc-wls"]}]
    },
    {
        "train": {"name": "neiss-sturm+zefys2025", "def": ["neiss-sturm", "zefys2025-nc-wls"]},
        "test": [{"name": "neiss-sturm", "def": ["neiss-sturm"]},
                 {"name": "zefys2025", "def": ["zefys2025-nc-wls"]}]
    },
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
    },
    {
        "train": {"name": "all-historic-woz",
                  "def": ["neiss-sturm", "neiss-arendt", "europeana-onb", "europeana-lft", "hisgerman-nc",
                          "hipe2020-nc"]},
        "test": [{"name": "europeana-lft", "def": ["europeana-lft"]},
                 {"name": "europeana-onb", "def": ["europeana-onb"]},
                 {"name": "neiss-arendt", "def": ["neiss-arendt"]},
                 {"name": "neiss-sturm", "def": ["neiss-sturm"]},
                 {"name": "hipe2020", "def": ["hipe2020-nc"]},
                 {"name": "hisgerman", "def": ["hisgerman-nc"]}]
    }
]

# batch_sizes = [32]
# learning_rates = [2e-5]
# batch_sizes = [32, 64, 96]
# learning_rates = [2e-5, 1e-4, 1e-3]
# weight_decays = [0.01]
# warmup_steps = [100]


# noinspection SpellCheckingInspection
@click.command()
@click.argument('result-file', type=click.Path())
@click.option('--max-epochs', type=int, default=30, help="Maximum number of epochs to train. Default 30.")
@click.option('--exp-type', type=click.Choice(['single', 'merged', 'historical', 'contemporary'],
                                              case_sensitive=False),
              default="single")
@click.option('--batch-size', type=int, multiple=True, default=[32],
              help="Can be supplied multiple times. Batch size to try.")
@click.option('--learning-rate', type=float, multiple=True, default=[2e-5],
              help="Can be supplied multiple times. Learning rate to try.")
@click.option('--weight-decay', type=float, multiple=True, default=[0.01],
              help="Can be supplied multiple times. Weight decay to try.")
@click.option('--warmup-step', type=int, multiple=True, default=[100],
              help="Can be supplied multiple times. Warmup steps to try.")
@click.option('--use-data-config', type=str, multiple=True, default=[],
              help="Can be supplied multiple times. Run only on these training configs.")
@click.option('--pretrain-config-file', type=click.Path(exists=True), default=None,
              help="Train on pretrained models defined in this result file (from a previous experiment.py run).")
@click.option('--pretrain-path', type=click.Path(exists=True), default="./",
              help="Load the pretrained models checkpoints (EXP_... directories) from this directory. Default './'")
@click.option('--model-storage-path', type=click.Path(exists=True), default="./",
              help="Store the models checkpoints (EXP_... directories) in this directory.")
@click.option('--dry-run', type=bool, is_flag=True, help='Dry run only. Do not train or test but just '
                                                         'check if everything runs through.')
def main(result_file, max_epochs, exp_type, batch_size, learning_rate, weight_decay, warmup_step, use_data_config,
         pretrain_config_file, pretrain_path, model_storage_path, dry_run):

    if not pretrain_path.endswith("/"):
        pretrain_path += "/"

    if not model_storage_path.endswith("/"):
        model_storage_path += "/"

    if exp_type == "single":
        data_configs = data_configs_single
    elif exp_type == "merged":
        data_configs = data_configs_single
    elif exp_type == "historical":
        use_data_config = ["europeana-lft", "europeana-onb", "hipe2020", "hisgerman",
                           "neiss-arendt", "neiss-sturm", "zefys2025"]
    elif exp_type == "contemporary":
        use_data_config = ["conll2003", "GermEval"]
    else:
        raise RuntimeError("Unknown type of experiment.")

    if len(use_data_config) > 0:
        data_configs = ([dc for dc in data_configs_single if dc["train"]["name"] in use_data_config] +
                        [dc for dc in data_configs_merged if dc["train"]["name"] in use_data_config])

    results = None
    if os.path.exists(result_file):
        results = pd.read_pickle(result_file)

    train.set_torch_device()

    if pretrain_config_file is None:
        model_defs = published_models
    else:
        pretrain_configs = pd.read_pickle(pretrain_config_file)

        model_defs = []
        for _, (model, exp_ID, pretrain_data) in pretrain_configs[["model", "exp_ID", "train"]].iterrows():
            model_tmp = named_published_models[model].copy()
            model_tmp["pretrained_model"] = exp_ID
            model_tmp["pretrain"] = pretrain_data

            model_defs.append(model_tmp)

    # noinspection PyUnboundLocalVariable
    for model_def, data_config, bs, lr, wd, ws in (
            itertools.product(model_defs, data_configs, batch_size, learning_rate, weight_decay, warmup_step)):

        try:
            train_params = config.TrainingParams()

            train_params.batch_size = bs
            train_params.learning_rate = lr
            train_params.weight_decay = wd
            train_params.warmup_steps = ws
            train_params.num_train_epochs = max_epochs

            exp_ID = "EXP_" + hashlib.sha256((str(model_def) + str(data_config) +
                                              str(train_params)).encode()).hexdigest()
            # noinspection PyTypeChecker
            if results is not None and sum(results.exp_ID == exp_ID) > 0:
                print("Skipping {} - experiment already exists.".format(exp_ID))
                continue

            model_config = config.Resource(path=model_def["path"], source="hf")
            model_config.set_name()
            print(model_config.info())

            train_config = data_config["train"]
            test_configs = data_config["test"]

            pretrained_model_path = None if "pretrained_model" not in model_def \
                else pretrain_path + model_def["pretrained_model"]

            if pretrained_model_path is not None:
                print("Loading pretrained model from: {}".format(pretrained_model_path))

            if dry_run:
                best_result = {"model": model_config.name, "train": train_config["name"],
                               "train_params": str(train_params), "epoch": -1, "f1_test": 0.0, "f1_early": 0.0}
            else:

                tokenizer = train.get_tokenizer(model_def["path"], model_def["add_prefix_space"],
                                                ignore_mismatched_sizes=model_def["ignore_mismatched_sizes"]
                                                if "ignore_mismatched_sizes" in model_def else False)

                train_tokenized_data = load_dataset_config(train_config, tokenizer)

                trained_ner_model, model_out_path, best_result = (
                    train.train_model(model_config, train_config["name"], zefys_label_list, train_params,
                                      train_tokenized_data, tokenizer, save_strategy="epoch",
                                      exp_model_path=model_storage_path + exp_ID,
                                      pretrained_model_path=pretrained_model_path))

            for test_config in test_configs:

                if dry_run:
                    result = best_result.copy()
                else:
                    # noinspection PyUnboundLocalVariable
                    test_tokenized_data = load_dataset_config(test_config, tokenizer)

                    # noinspection PyUnboundLocalVariable
                    class_report, errors = eval_opt.compute_metrics_per_tag(trained_ner_model, test_tokenized_data,
                                                                            zefys_label_list, output_dict=True)
                    result = process_report(best_result.copy(), class_report)

                result['test'] = test_config["name"]

                if "pretrain" in model_def:
                    result["pretrain"] = model_def["pretrain"]
                    result["pretrained_model"] = model_def["pretrained_model"]

                result["exp_ID"] = exp_ID

                if results is None:
                    results = pd.DataFrame([result])
                else:
                    results = pd.concat([results, pd.DataFrame([result])])

            results.to_pickle(result_file)
        except torch.OutOfMemoryError as e:
            print("Out of memory.")

    results.to_pickle(result_file)


def load_dataset_config(data_config, tokenizer):
    datasets = []
    for dataset_def in data_config["def"]:

        dataset_def = get_dataset_def(dataset_def)

        dataset_config = config.Resource(path=dataset_def["path"], source=dataset_def["source"])
        dataset_config.set_name()

        print(dataset_config.info())

        datasets.append(train.load_ner_dataset(dataset_config.path, dataset_config.source))

    merged_dataset = merge_ds(datasets)

    merged_dataset = map_ner_tags_to_zefys(merged_dataset)

    tokenized_dataset = train.prepare_dataset(merged_dataset, tokenizer)

    return tokenized_dataset


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
