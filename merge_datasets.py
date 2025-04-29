from datasets import Dataset, DatasetDict, ClassLabel, Sequence, concatenate_datasets
import pandas as pd

zefys_label_list = ["B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG", "O"]


def get_label_list(labels):
    # copied from https://github.com/huggingface/transformers/blob/66fd3a8d626a32989f4569260db32785c6cbf42a/
    # examples/pytorch/token-classification/run_ner.py#L320
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def merge_ds(list_of_datasets):

    merged_dataset = None

    for i, ds in enumerate(list_of_datasets):
        if i == 0:  # at the beginning just take the first one

            merged_dataset = ds

        else:  # merge all other together with the previously merged dataset
            merged_dataset = DatasetDict(
                {"train": concatenate_datasets([merged_dataset["train"], list_of_datasets[i]["train"]]),
                 "test": concatenate_datasets([merged_dataset["test"], list_of_datasets[i]["test"]]),
                 "validation": concatenate_datasets(
                     [merged_dataset["validation"], list_of_datasets[i]["validation"]])})

    if merged_dataset is None:
        raise RuntimeError("No dataset have been merged.")

    return merged_dataset


def map_split_ner_tags_to_zefys(datasplit):

    ner_tags = datasplit["ner_tags"]

    for i_sent, split_sent in enumerate(ner_tags):
        ner_tags[i_sent] = ["O" if x not in zefys_label_list else x for x in split_sent]

    split_updated = Dataset.from_pandas(
        pd.DataFrame({'id': datasplit['id'], 'tokens': datasplit['tokens'], 'ner_tags': ner_tags}))

    split_updated = split_updated.cast_column("ner_tags", Sequence(ClassLabel(names=zefys_label_list)))
    
    return split_updated


def map_ner_tags_to_zefys(dataset):

    dataset_updated = DatasetDict({
        "train": map_split_ner_tags_to_zefys(dataset["train"]),
        "validation": map_split_ner_tags_to_zefys(dataset["validation"]),
        "test": map_split_ner_tags_to_zefys(dataset["test"])
    })
                
    return dataset_updated
