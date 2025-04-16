import pandas as pd
import csv
import numpy as np 
from datasets import Dataset, DatasetDict
from datasets import Sequence, ClassLabel
from datasets import concatenate_datasets

input_path = "../data/hipe2022_not-preprocessed/newseye/"
dev_dataset = pd.read_csv(input_path + "hipe-newseye-dev-de.tsv", sep="\t", comment="#", quoting=csv.QUOTE_NONE)
dev2_dataset = pd.read_csv(input_path + "hipe-newseye-dev2-de.tsv", sep="\t", comment="#", quoting=csv.QUOTE_NONE)
test_dataset = pd.read_csv(input_path + "hipe-newseye-test-de.tsv", sep="\t", comment="#", quoting=csv.QUOTE_NONE)
train_dataset = pd.read_csv(input_path + "hipe-newseye-train-de.tsv", sep="\t", comment="#", quoting=csv.QUOTE_NONE)

label_list = set([x for x in train_dataset["NE-COARSE-LIT"] if x not in [None, np.nan]]) #to do: remove nan - also in labels!
label_list = [x for x in label_list]
print(label_list)

# read TOKEN and NE-TAG over all files as one list, split into sentences by using No. col

# define lists to add all sentences to
tokens=[]
tags=[]

# function to preprocess and clean all dataset splits in the same way
def clean_dataset_split(input):

    input["MISC"] = input["MISC"].str.split("|")

    #check if "EndOfSentence" in MISC col
    input_sent_ends = input.loc[input['MISC'].explode().eq('EndOfSentence').loc[lambda x: x].index]
    df_ends_idx = input_sent_ends.index.tolist()

    #read into lists of tokens / tags per sentence while cleaning from structural information / comments
    sent_tokens = []
    sent_ner_tags = []
    
    for i, end_idx in enumerate(df_ends_idx):
        if i == 0:
            sent_start = -1
        else:
            sent_start = df_ends_idx[i-1]
        sent_end = df_ends_idx[i]
        sent_tokens_check = input["TOKEN"][sent_start+1:sent_end+1].tolist()
        sent_ner_tags_check = input["NE-COARSE-LIT"][sent_start+1:sent_end+1].tolist()

        if np.nan in sent_ner_tags_check:
            indexes = [i for i, x in enumerate(sent_ner_tags_check) if str(x) == "nan"]
            sent_tokens_check = [i for j, i in enumerate(sent_tokens_check) if j not in indexes]
            sent_ner_tags_check = [i for j, i in enumerate(sent_ner_tags_check) if j not in indexes]

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

    #df_dataset_updated.to_csv("data/newseye_train.csv", quoting=csv.QUOTE_ALL)
    
    #transform data into Datasets format and set labels for ClassLabel based on label_list
    dataset_updated = Dataset.from_pandas(df_dataset_updated)
    #dataset_updated = dataset_updated.cast_column("ner_tags", Sequence(ClassLabel(names=label_list)))

    return dataset_updated

#clean all splits per function
dataset_test_cleaned = clean_dataset_split(test_dataset)
dataset_val_cleaned = clean_dataset_split(dev_dataset)
dataset_val2_cleaned = clean_dataset_split(dev2_dataset)
dataset_train_cleaned = clean_dataset_split(train_dataset)



newseye_dataset_cleaned = DatasetDict({
    "train":dataset_train_cleaned,
    "validation": concatenate_datasets([dataset_val_cleaned, dataset_val2_cleaned]),
    "test":dataset_test_cleaned
})

newseye_dataset_cleaned.save_to_disk("data/newseye_not-casted.hf")