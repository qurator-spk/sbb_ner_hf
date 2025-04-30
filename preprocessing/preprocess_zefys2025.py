# imports
import pandas as pd
import csv
import os
from datasets import Dataset, DatasetDict
from datasets import Features, Sequence, Value, ClassLabel
import itertools
import numpy as np

# read all files into one large dataframe
cwd = os.getcwd()

input_path = "../../sbb_ner_data/zefys/"
files = [file for file in os.listdir(input_path) if file.endswith('.tsv')]

# read TOKEN and NE-TAG over all files as one list, split into sentences by using No. col

# define lists to add all sentences to
tokens=[]
tags=[]

for file in files:
    #read input as pd df
    input = pd.read_csv(input_path + file, sep="\t", comment="#", quoting=csv.QUOTE_NONE)

    #find sentence beginnings
    sent_starts = input.loc[input["No."] == 0]
    sent_starts_idx = sent_starts.index.tolist()
    end_of_file_pos = len(input.index) #add pos of last line to be able to include last sent in for loop
    sent_starts_idx.append(end_of_file_pos)

    #read into nested lists based on sentence structure
    for i, index in enumerate(sent_starts_idx):
        if i != 0:
        
            sent_tokens = input["TOKEN"][sent_starts_idx[i-1]:index].tolist()
            sent_tags = input["NE-TAG"][sent_starts_idx[i-1]:index].tolist()

            #remove nan tokens
            if np.nan in sent_tokens:
                indexes = [i for i, x in enumerate(sent_tokens) if str(x) == "nan"]
                sent_tokens = [i for j, i in enumerate(sent_tokens) if j not in indexes]
                sent_tags = [i for j, i in enumerate(sent_tags) if j not in indexes]

            #remove nan tags
            if np.nan in sent_tags:
                indexes = [i for i, x in enumerate(sent_tags) if str(x) == "nan"]
                print(file, sent_tokens, sent_tags, indexes)
                sent_tokens = [i for j, i in enumerate(sent_tokens) if j not in indexes]
                sent_tags = [i for j, i in enumerate(sent_tags) if j not in indexes]

            tokens.append(sent_tokens)
            tags.append(sent_tags)

#this should equal the number of sentences in the dataset:
print(len(tokens), len(tags))

# create HF dataset from lists

# create new, minimal df based on lists
df_dataset = pd.DataFrame(
{'id': range(len(tokens)),
'tokens': tokens,
'ner_tags': tags
})

#df_dataset.to_csv("data/zefys.csv")

# define label_lists
tags_flattened = list(itertools.chain(*tags))
label_list = list(set(tags_flattened))

# transform data into Datasets format and set labels for ClassLabel based on label_list
zefys_dataset = Dataset.from_pandas(df_dataset)
#zefys_dataset = zefys_dataset.cast_column("ner_tags", Sequence(ClassLabel(names=label_list)))

# add train/test/val split to dataset
# https://discuss.huggingface.co/t/how-to-split-main-dataset-into-train-dev-test-as-datasetdict/1090/2

# 80% train, 20% test + validation
train_testvalid = zefys_dataset.train_test_split(test_size=0.2)
# Split the 20% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
# gather everyone if you want to have a single DatasetDict
zefys_dataset_splits = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']})

print(zefys_dataset_splits)

#https://stackoverflow.com/a/72021864
zefys_dataset_splits.save_to_disk("data/zefys2025.hf")

"""
#how to (re-)load?
from datasets import load_from_disk
ds = load_from_disk('data/zefys2025.hf')
"""