import csv
import pandas as pd
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from datasets import Features, Sequence, Value, ClassLabel

dataset_name = "stefan-it/HisGermaNER"

dataset_train = load_dataset(dataset_name, delimiter="\t", quoting=csv.QUOTE_NONE, split='train')  
dataset_test = load_dataset(dataset_name, delimiter="\t", quoting=csv.QUOTE_NONE, split='test') 
dataset_val = load_dataset(dataset_name, delimiter="\t", quoting=csv.QUOTE_NONE, split='validation') 

label_list = set([x for x in dataset_train["NE-COARSE-LIT"] if x is not None])
label_list = [x for x in label_list]
label_list

# function to preprocess and clean all dataset splits in the same way
def clean_dataset_split(dataset_split):
    #read as df
    df_dataset = pd.DataFrame(dataset_split)
    
    #identify sentence splits using EndOfSentence
    df_ends = df_dataset.loc[df_dataset["MISC"] == "EndOfSentence"]
    df_ends_idx = df_ends.index.tolist()

    #read into lists of tokens / tags per sentence while cleaning from structural information / comments
    sent_tokens = []
    sent_ner_tags = []
    
    for i, end_idx in enumerate(df_ends_idx):
        sent_start = df_ends_idx[i-1]
        sent_end = df_ends_idx[i]
        sent_tokens_check = df_dataset["TOKEN"][sent_start+1:sent_end+1].tolist()
        sent_ner_tags_check = df_dataset["NE-COARSE-LIT"][sent_start+1:sent_end+1].tolist()
        if sent_tokens_check != []:
            if "-DOCSTART-" in sent_tokens_check:
                docstart_indices = [i for i, x in enumerate(sent_tokens_check) if x == "-DOCSTART-"]
                comment_indices = [i for i, x in enumerate(sent_tokens_check) if x.startswith("# onb:")]
                indices_to_delete = docstart_indices + comment_indices
                sent_tokens_check = [i for j, i in enumerate(sent_tokens_check) if j not in indices_to_delete]
                sent_ner_tags_check = [i for j, i in enumerate(sent_ner_tags_check) if j not in indices_to_delete]
        sent_tokens.append(sent_tokens_check)
        sent_ner_tags.append(sent_ner_tags_check)
    
    sent_tokens = [x for x in sent_tokens if x != []]
    sent_ner_tags = [x for x in sent_ner_tags if x != []]

    if len(sent_tokens) == len(sent_ner_tags):
        pass
    else:
        print("the length of sent_tokens and sent_ner_tags lists are not the same - please check again!")

    #create new, minimal df based on lists
    df_dataset_updated = pd.DataFrame(
    {'id': range(len(sent_tokens)),
     'tokens': sent_tokens,
     'ner_tags': sent_ner_tags
    })

    #transform data into Datasets format and set labels for ClassLabel based on label_list
    dataset_updated = Dataset.from_pandas(df_dataset_updated)
    dataset_updated = dataset_updated.cast_column("ner_tags", Sequence(ClassLabel(names=label_list)))

    return dataset_updated

#clean all splits per function
dataset_train_cleaned = clean_dataset_split(dataset_train)
dataset_test_cleaned = clean_dataset_split(dataset_test)
dataset_val_cleaned = clean_dataset_split(dataset_val)

hisgermaner_dataset_cleaned = DatasetDict({
    "train":dataset_train_cleaned,
    "validation":dataset_val_cleaned,
    "test":dataset_test_cleaned
})

hisgermaner_dataset_cleaned.save_to_disk("data/hisgermaner.hf")