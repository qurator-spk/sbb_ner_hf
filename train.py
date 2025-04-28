import torch
from datetime import datetime
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, \
    EarlyStoppingCallback  # , AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
# import eval_opt
import evaluate
import numpy as np
from glob import glob

task = "ner"


def set_torch_device():
    # set GPU else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def set_model_path(model_path, dataset_path):
    # set path for saving model later
    now = datetime.now().strftime('%Y-%m-%d_%H-%M')
    out_path = f"{model_path}_{dataset_path}_{now}"
    return out_path


def load_ner_dataset(dataset_path, dataset_source):
    if dataset_source == "hf":
        # load a dataset (HF)
        dataset = load_dataset(dataset_path, cache_dir="./hf-dataset-cache/" + dataset_path, trust_remote_code=True)
    elif dataset_source == "local":
        dataset = load_from_disk(dataset_path)
    else:
        raise RuntimeError("Unknown type of source.")
    return dataset


def get_tokenizer(model_path, add_prefix_space=False, ignore_mismatched_sizes=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=add_prefix_space,
                                              ignore_mismatched_sizes=ignore_mismatched_sizes)
    return tokenizer


def get_label_list(dataset):
    label_list = dataset["train"].features[f"{task}_tags"].feature.names
    return label_list


def prepare_dataset(dataset, tokenizer):
    label_all_tokens = True
    label_list = get_label_list(dataset)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"{task}_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    return tokenized_dataset


# dropout as described here:
# https://discuss.huggingface.co/t/changing-dropout-during-disltilbert-fine-tuning/88290/3
# raises TypeError: XLMRobertaForSequenceClassification.__init__() got an unexpected keyword argument 'dropout'
# dropout=train_params.dropout, attention_dropout=train_params.dropout
def load_model(model_path, label_list):
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(label_list))
                                                            # cache_dir=".hf-model-cache/" + model_path)
    return model


def train_model(model_config, data_config_name, label_list, train_params, tokenized_dataset, tokenizer,
                save_strategy="steps", exp_model_path=None, pretrained_model_path=None):

    epoch = 0
    best_f1 = -1.0
    best_result = None

    def compute_metrics(p):
        nonlocal best_f1
        nonlocal best_result
        nonlocal epoch

        epoch += 1

        metric = evaluate.load(
            "seqeval")  # load_metric has been removed,
        # see https://discuss.huggingface.co/t/cant-import-load-metric-from-datasets/107524/2

        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)

        result = {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            "f1_early": round(results["overall_f1"], 2)
        }

        if best_f1 < result["f1"]:
            best_f1 = result["f1"]
            best_result = result
            best_result["epoch"] = epoch
            best_result["model"] = model_config.name
            best_result["train"] = data_config_name
            best_result["train_params"] = str(train_params)

        return result

    model_out_path = set_model_path(model_config.name, data_config_name)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    model_load_path = model_config.path

    if pretrained_model_path is not None:
        model_load_path = glob(pretrained_model_path + "/checkpoint*")[0]

    model = load_model(model_load_path, label_list)

    train_args = TrainingArguments(
        model_out_path if exp_model_path is None else exp_model_path,
        eval_strategy=train_params.eval_strategy,
        save_strategy=save_strategy,
        learning_rate=train_params.learning_rate,
        per_device_train_batch_size=train_params.batch_size,
        per_device_eval_batch_size=train_params.batch_size,
        num_train_epochs=train_params.num_train_epochs,
        weight_decay=train_params.weight_decay,
        metric_for_best_model="f1_early",
        greater_is_better=True,
        save_total_limit=1,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    return trainer, model_out_path, best_result
