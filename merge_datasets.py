from datasets import Dataset, DatasetDict, Features, ClassLabel, Sequence
import pandas as pd

#create mapping dict: num > label
def create_mapping_dict(label_list):
    mapping_dict = dict()
    for i, label in enumerate(label_list):
        mapping_dict[i] = label
    return mapping_dict

#replace all labels, with either "O" or other label from label_list
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

def combine_label_functions(datasplit, mapping_dict, label_idxs_to_delete, label_list): 
    split_labels_updated = replace_all_labels(datasplit, mapping_dict, label_idxs_to_delete)
    split_df_updated = pd.DataFrame(
    {'id': datasplit['id'],
     'tokens': datasplit['tokens'],
     'ner_tags': split_labels_updated,

    })
    label_list_updated = [x for i, x in enumerate(label_list) if i not in label_idxs_to_delete]
    split_updated = Dataset.from_pandas(split_df_updated)
    split_updated = split_updated.cast_column("ner_tags", Sequence(ClassLabel(names=label_list_updated)))
    
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

    dataset_train = combine_label_functions(train_split, mapping_dict, label_idxs_to_delete, label_list)
    dataset_test = combine_label_functions(test_split, mapping_dict, label_idxs_to_delete, label_list)
    dataset_val = combine_label_functions(val_split, mapping_dict, label_idxs_to_delete, label_list)

    dataset_updated = DatasetDict({
    "train":dataset_train,
    "validation":dataset_val,
    "test":dataset_test
    })
                
    return dataset_updated