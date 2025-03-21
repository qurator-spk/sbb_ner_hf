import evaluate
import numpy as np
import optuna
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
import train

metric = evaluate.load("seqeval")

def compute_metrics_per_tag(trainer, tokenized_dataset, label_list):
    # eval metrics for each category
    
    predictions, labels, _ = trainer.predict(tokenized_dataset["validation"])
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
    
    results_per_tag = metric.compute(predictions=true_predictions, references=true_labels)
    return results_per_tag

def optimize(optimize_params, train_params, model_path, label_list, tokenizer, tokenized_dataset):

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

    model_out_path = train.set_model_path(model_path)
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