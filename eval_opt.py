import evaluate
import numpy as np
import optuna
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
import train
from seqeval.metrics import classification_report
import itertools

metric = evaluate.load("seqeval")

def compute_metrics_per_tag(trainer, tokenized_dataset, label_list):
    # eval metrics for each category
    
    predictions, labels, _ = trainer.predict(tokenized_dataset["test"])
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

    pred_flattened = list(itertools.chain(*true_predictions))
    labels_flattened = list(itertools.chain(*true_labels))
    class_report = classification_report([labels_flattened], [pred_flattened])

    #print(len(true_predictions), len(true_labels), len(tokenized_dataset["test"]))
    """
    print("example from test dataset: " + str(tokenized_dataset["test"][0]["tokens"]))
    print("true classification in test dataset: " + str(true_labels[0]))
    print("predicted classification based on finetuned model: " + str(true_predictions[0]))
    """

    errors = []
    for i, (pred, lab) in enumerate(zip(true_predictions, true_labels)):
        if pred != lab:
            errors.append([true_predictions[i], true_labels[i]])
            
    #print(errors[0])
    print(classification_report([labels_flattened], [pred_flattened]))
    
    return class_report, errors

def optimize(optimize_params, train_params, model_path, model_out_path, label_list, tokenizer, tokenized_dataset):

    def optuna_hp_space(trial):
        hp_space_dict = dict()
        for k, v in optimize_params.hp_space.items():
            if v["param_type"] == "float":
                hp_space_dict[k] = trial.suggest_float(k, v["borders"][0], v["borders"][1], log=True)
            elif v["param_type"] == "categorical": 
                hp_space_dict[k] = trial.suggest_categorical(k, v["borders"])
        print("optimizing within the following hyperparameter space:")
        print(hp_space_dict)
        return hp_space_dict

    def model_init(trial):
        model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(label_list))
        return model

    def compute_metrics(p):
        #load_metric has been removed, see https://discuss.huggingface.co/t/cant-import-load-metric-from-datasets/107524/2
        
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
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    #model_out_path = train.set_model_path(model_path)
    data_collator = DataCollatorForTokenClassification(tokenizer)

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
    model=None,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    model_init=model_init,
    data_collator=data_collator)

    best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=optimize_params.n_trials,
    #compute_objective=compute_objective,
    )

    return best_trial

def save_class_report(class_report, out_format, model_out_path):
    #html
    if out_format == "html":
        class_report_str = class_report.lstrip()
        class_report_str = class_report_str[:-2]
        class_report_str = class_report_str.replace("micro avg", "micro_avg")
        class_report_str = class_report_str.replace("macro avg", "macro_avg")
        class_report_str = class_report_str.replace("weighted avg", "weighted_avg")
        class_report_str = class_report_str.replace('\n\n', '</td></tr><tr><td>')
        class_report_str = class_report_str.replace('\n', '</td></tr><tr><td>')
        class_report_str = ' '.join(class_report_str.split())
        class_report_str = class_report_str.replace('<td> ', '<td>')
        class_report_str = class_report_str.replace(' ', '</td><td>')
        class_report_str = "<html><body><div><span><table><tr><td></td><td>" + class_report_str + "</td></tr></table></span></div></body></html>"
        with open(model_out_path + "/classification_report.html", "w") as file:
            file.write(class_report_str)
    #md
    elif out_format == "md":
        class_report_str = class_report.lstrip()
        class_report_str = class_report_str[:-2]
        class_report_str = class_report_str.replace("micro avg", "micro_avg")
        class_report_str = class_report_str.replace("macro avg", "macro_avg")
        class_report_str = class_report_str.replace("weighted avg", "weighted_avg")
        class_report_strs = class_report_str.split("\n\n")
        
        for i, each_str in enumerate(class_report_strs):
            lines = each_str.split("\n")
            for idx, line in enumerate(lines):
                line =  ' '.join(line.split())
                line = line.replace(" ", " | ")
                line = "| " + line + " |\n"
                lines[idx] = line
            class_report_strs[i] = " ".join(lines)
        
        class_report_header = "| | precision | recall | f1-score | support |\n | --- | --- | --- | --- | --- |\n "
        class_report_md = class_report_header + class_report_strs[1] + class_report_strs[2][:-1]
        class_report_md
        with open(model_out_path + "/classification_report.md", "w") as file:
            file.write(class_report_md)