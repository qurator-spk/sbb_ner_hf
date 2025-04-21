import pandas as pd
from datasets import Dataset, DatasetDict
import csv

#simply switch file names here and output name in the end to change NEISS editions ["sturm"/"arendt"]

train_file = "data/neiss_not_preprocessed/train_sturm.conll"
test_file = "data/neiss_not_preprocessed/test_sturm.conll"
val_file = "data/neiss_not_preprocessed/dev_sturm.conll"

def read_conll_files(file):
    tokens = [[]]
    ner_tags = [[]]
    file_opened = open(file, "r", encoding="utf-8")
    for line in file_opened:
        if line != "\n":
            line = line.split()
            #tokens
            token = line[0]
            tokens[-1].append(token)
            #tags
            ner_tag = line[1]
            #rename PER/LOC/ORG to fit our label definitions
            ner_tag = ner_tag.upper()
            if ner_tag == "B-PERSON" or ner_tag=="B-PERS":
                ner_tag = "B-PER"
            elif ner_tag == "B-PLACE":
                ner_tag = "B-LOC"
            elif ner_tag == "B-ORGANIZATION":
                ner_tag = "B-ORG"
            elif ner_tag == "I-PERSON" or ner_tag=="I-PERS":
                ner_tag = "I-PER"
            elif ner_tag == "I-PLACE":
                ner_tag = "I-LOC"
            elif ner_tag == "I-ORGANIZATION":
                ner_tag = "I-ORG"
            ner_tags[-1].append(ner_tag)
        else:
            tokens.append([])
            ner_tags.append([])

    dataset_df = pd.DataFrame(
    {'id': range(len(tokens)),
     'tokens': tokens,
     'ner_tags': ner_tags
    })
    
    #dataset_df.to_csv("data/sturm_train.csv", quoting=csv.QUOTE_ALL)
    dataset_hf = Dataset.from_pandas(dataset_df)

    file_opened.close()

    return dataset_hf

val_dataset = read_conll_files(val_file)
test_dataset = read_conll_files(test_file)
train_dataset = read_conll_files(train_file)

neiss_dataset = DatasetDict({
    "train":train_dataset,
    "validation": val_dataset,
    "test":test_dataset
})

neiss_dataset.save_to_disk("data/neiss_sturm.hf")