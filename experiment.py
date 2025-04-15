import pandas as pd

import config
import train
import eval_opt
import click
import os
import pandas as pd
import itertools
import hashlib

models = [
#    {"path": "flair/ner-german", "add_prefix_space": False},
    {"path": "dbmdz/bert-tiny-historic-multilingual-cased", "add_prefix_space": False},
    {"path": "dbmdz/bert-mini-historic-multilingual-cased", "add_prefix_space": False},
    #{"path": "dbmdz/bert-base-german-cased", "add_prefix_space": False},
    #{"path": "FacebookAI/roberta-base", "add_prefix_space": True},
    #{"path": "dbmdz/bert-base-historic-multilingual-cased", "add_prefix_space": False},
    #{"path": "distilbert/distilbert-base-uncased", "add_prefix_space": False}
]

datasets = [
    {"name": "zefys2025", "path": "data/zefys2025_20250404.hf", "source": "local"},
    {"name": "newseye", "path": "data/newseye_20250404.hf", "source": "local"},
    {"name": "hisgerman", "path": "data/hisgermaner_20250404.hf", "source": "local"},
    {"name": "conll2003", "path": "eriktks/conll2003", "source": "hf"},
    {"name": "GermEval",  "path": "GermanEval/germeval_14", "source": "hf"}
]


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


@click.command()
@click.argument('result-file', type=click.Path())
def main(result_file):

    max_epochs = 1

    results = None
    if os.path.exists(result_file):
        results = pd.read_pickle(result_file)

    print(models)
    print(datasets)

    for model_def, dataset_def in itertools.product(models, datasets):

        train_params = config.TrainingParams()

        train_params.batch_size = 32
        train_params.num_train_epochs = max_epochs

        exp_ID = "EXP_" + hashlib.sha256((str(model_def) + str(dataset_def) + str(train_params)).encode()).hexdigest()

        if results is not None and sum(results.exp_ID == exp_ID) > 0:
            print("Skipping {} - experiment already exists.".format(exp_ID))
            continue

        model_config = config.Resource(path=model_def["path"], source="hf")
        model_config.set_name()
        print(model_config.info())

        dataset_config = config.Resource(path=dataset_def["path"], source=dataset_def["source"])
        dataset_config.set_name()

        print(dataset_config.info())

        train.set_torch_device()

        ner_dataset = train.load_ner_dataset(dataset_config.path, dataset_config.source)

        tokenizer = train.get_tokenizer(model_def["path"], model_def["add_prefix_space"])

        tokenized_dataset = train.prepare_dataset(ner_dataset, tokenizer)

        label_list = train.get_label_list(ner_dataset)

        trained_ner_model, model_out_path, best_result = (
            train.train_model(model_config, dataset_config, label_list, train_params,
                              tokenized_dataset, tokenizer, save_strategy="epoch", exp_model_path=exp_ID))

        class_report, errors = eval_opt.compute_metrics_per_tag(trained_ner_model, tokenized_dataset, label_list,
                                                                output_dict=True)

        result = process_report(best_result, class_report)

        result["exp_ID"] = exp_ID

        if results is None:
            results = pd.DataFrame([result])
        else:
            results = pd.concat([results, pd.DataFrame([result])])

        # import ipdb;ipdb.set_trace()

        #if len(results) > 2:
        #    break

    results.to_pickle(result_file)


if __name__ == '__main__':
    main()
