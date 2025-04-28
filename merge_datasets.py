from datasets import Dataset, DatasetDict, ClassLabel, Sequence, concatenate_datasets
import pandas as pd


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


# create mapping dict: num > label
def create_mapping_dict(label_list):
    mapping_dict = dict()
    for i, label in enumerate(label_list):
        mapping_dict[i] = label
    return mapping_dict


# replace all labels, with either "O" or other label from label_list
def replace_all_labels(datasplit, mapping_dict, labels_idxs_to_delete):
    ner_tags = datasplit["ner_tags"]
    for key, value in mapping_dict.items():
        if key not in labels_idxs_to_delete:
            for i_sent, split_sent in enumerate(ner_tags): 
                if key in split_sent:
                    ner_tags[i_sent] = [value if x==key else x for x in split_sent]
        else:
            for i_sent, split_sent in enumerate(ner_tags): 
                if key in split_sent:
                    ner_tags[i_sent] = ["O" if x==key else x for x in split_sent]
    return ner_tags


def combine_label_functions(datasplit, mapping_dict, label_idxs_to_delete, label_list, zefys_label_list): 
    split_labels_updated = replace_all_labels(datasplit, mapping_dict, label_idxs_to_delete)
    split_df_updated = pd.DataFrame(
    {'id': datasplit['id'],
     'tokens': datasplit['tokens'],
     'ner_tags': split_labels_updated,

    })
    label_list_updated = [x for i, x in enumerate(label_list) if i not in label_idxs_to_delete]
    split_updated = Dataset.from_pandas(split_df_updated)
    split_updated = split_updated.cast_column("ner_tags", Sequence(ClassLabel(names=zefys_label_list)))
    
    return split_updated

    
def drop_ner_labels(label_list, dataset):
    zefys_label_list = ["B-LOC", "I-LOC", "B-PER", "I-PER", "B-ORG", "I-ORG", "O"]

    label_idxs_to_delete = []
    for i, label in enumerate(label_list):
        if label not in zefys_label_list:
            label_idxs_to_delete.append(i)
    
    train_split = dataset["train"]
    test_split = dataset["test"]
    val_split = dataset["validation"]

    mapping_dict = create_mapping_dict(label_list)

    dataset_train = combine_label_functions(train_split, mapping_dict, label_idxs_to_delete, label_list, zefys_label_list)
    dataset_test = combine_label_functions(test_split, mapping_dict, label_idxs_to_delete, label_list, zefys_label_list)
    dataset_val = combine_label_functions(val_split, mapping_dict, label_idxs_to_delete, label_list, zefys_label_list)

    dataset_updated = DatasetDict({
        "train": dataset_train,
        "validation": dataset_val,
        "test": dataset_test
    })
                
    return dataset_updated
