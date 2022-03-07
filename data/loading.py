import os
from Bio import SeqIO
import numpy as np
import pandas as pd


DATA_DIR = r"D:\Documents\datasets\AIST4010\arg sequences\data"
train_fp = os.path.join(DATA_DIR, "train.fasta")
val_fp = os.path.join(DATA_DIR, "val.fasta")
test_fp = os.path.join(DATA_DIR, "test.fasta")


def extract_class_and_seq(records):
    for idx in range(len(records)):
        desc, seq = records[idx].description.split('|'), str(records[idx].seq)
        if desc[0] == "sp":
            records[idx] = ("sp", seq)
        else:
            records[idx] = (desc[3], seq)
    return records


def load_data_as_df(phase, fasta_fp=None):
    if fasta_fp:
        fp = fasta_fp
    elif phase == "train":
        fp = train_fp
    elif phase == "valid":
        fp = val_fp
    elif phase == "test":
        fp = test_fp
    else:
        raise ValueError("Unknown phase")
    with open(fp) as handle:
        records = extract_class_and_seq(list(SeqIO.parse(handle, 'fasta')))
        df = pd.DataFrame(records, columns=["arg_class", "sequence"])
    return df
