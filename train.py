import torch
from datetime import datetime
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import eval_opt
import evaluate
import numpy as np

task = "ner"

def set_torch_device():
    #set GPU else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def set_model_path(model_path):
    #set path for saving model later
    now = datetime.now().strftime('%Y-%m-%d_%H-%M')
    model_name = model_path.split("/")[-1]
    out_path = f"{model_name}-finetuned-{task}-{now}"
    return out_path

def load_ner_dataset(dataset_path, dataset_source):
    if dataset_source == "hf":
        # load a dataset (HF)
        dataset = load_dataset(dataset_path)
    elif dataset_source == "local":
        dataset = load_from_disk(dataset_path)
    return dataset

def get_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
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

def load_model(model_path, label_list):
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(label_list))
    return model

def train_model(model_path, label_list, train_params, tokenized_dataset, tokenizer):
    
    def compute_metrics(p):
        metric = evaluate.load("seqeval") #load_metric has been removed, see https://discuss.huggingface.co/t/cant-import-load-metric-from-datasets/107524/2
        
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

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    model_out_path = set_model_path(model_path)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    model = load_model(model_path, label_list)

    args = TrainingArguments(
    model_out_path,
    eval_strategy = train_params.eval_strategy,
    learning_rate=train_params.learning_rate,
    per_device_train_batch_size=train_params.batch_size,
    per_device_eval_batch_size=train_params.batch_size,
    num_train_epochs=train_params.num_train_epochs,
    weight_decay=train_params.weight_decay,
    )

    trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer
    