import pandas as pd
import csv
import numpy as np 
from datasets import Dataset, DatasetDict
from datasets import Sequence, ClassLabel

input_path = "data/hipe_hipe2020_not_preprocessed/"
dev_dataset = pd.read_csv(input_path + "hipe2020_dev.tsv", sep="\t", comment="#", quoting=csv.QUOTE_NONE)
test_dataset = pd.read_csv(input_path + "hipe2020_test.tsv", sep="\t", comment="#", quoting=csv.QUOTE_NONE)
train_dataset = pd.read_csv(input_path + "hipe2020_train.tsv", sep="\t", comment="#", quoting=csv.QUOTE_NONE)

label_list = set([x for x in train_dataset["NE-COARSE-LIT"] if x not in [None, np.nan]]) #to do: remove nan - also in labels!
label_list = [x.upper() for x in label_list]
label_list = ["B-PER" if x=="B-PERS" else x for x in label_list]
label_list = ["I-PER" if x=="I-PERS" else x for x in label_list]
print(label_list)

# read TOKEN and NE-TAG over all files as one list, split into sentences by using No. col

# define lists to add all sentences to
tokens=[]
tags=[]

# function to preprocess and clean all dataset splits in the same way
def clean_dataset_split(input):

    input["MISC"] = input["MISC"].str.split("|")
    input["NE-COARSE-LIT"] = input["NE-COARSE-LIT"].str.upper()
    input["NE-COARSE-LIT"] = input["NE-COARSE-LIT"].replace("B-PERS", "B-PER")
    input["NE-COARSE-LIT"] = input["NE-COARSE-LIT"].replace("I-PERS", "I-PER")

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
        if np.nan in sent_tokens_check:
            indexes = [i for i, x in enumerate(sent_tokens_check) if str(x) == "nan"]
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
    
    df_dataset_updated = df_dataset_updated.dropna()

    #save last split as csv for result checking
    #df_dataset_updated.to_csv("data/hipe2020_test.csv", quoting=csv.QUOTE_ALL)
    
    #transform data into Datasets format and set labels for ClassLabel based on label_list
    dataset_updated = Dataset.from_pandas(df_dataset_updated)
    #dataset_updated = dataset_updated.cast_column("ner_tags", Sequence(ClassLabel(names=label_list)))

    return dataset_updated

#clean all splits per function

dataset_val_cleaned = clean_dataset_split(dev_dataset)
dataset_train_cleaned = clean_dataset_split(train_dataset)
dataset_test_cleaned = clean_dataset_split(test_dataset)

newseye_dataset_cleaned = DatasetDict({
    "train":dataset_train_cleaned,
    "validation": dataset_val_cleaned,
    "test":dataset_test_cleaned
})

newseye_dataset_cleaned.save_to_disk("data/hipe2020_not-casted.hf")