from Bio import SeqIO
from data.preprocess import TARGET_DICTIONARY, seq2ohe
from data.sampler import get_weighted_sampler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import torch

DATA_DIR = r"D:\Documents\datasets\AIST4010\arg sequences\data"
train_fp = os.path.join(DATA_DIR, "train.fasta")
val_fp = os.path.join(DATA_DIR, "val.fasta")
test_fp = os.path.join(DATA_DIR, "test.fasta")


# data loading
def extract_class_and_seq(records, is_test=False):
    if is_test:
        for idx in range(len(records)):
            desc, seq = records[idx].description, str(records[idx].seq)
            records[idx] = (desc, seq)
    else:
        for idx in range(len(records)):
            desc, seq = records[idx].description.split('|'), str(records[idx].seq)
            if desc[0] == "sp":
                records[idx] = ("sp", TARGET_DICTIONARY["sp"], seq)
            else:
                records[idx] = (desc[3], TARGET_DICTIONARY[desc[3]], seq)
    return records


def load_data_as_df(phase, fasta_fp=None):
    cols = ["arg_class", "target", "sequence"]
    if fasta_fp:
        fp = fasta_fp
    elif phase == "train":
        fp = train_fp
    elif phase == "valid":
        fp = val_fp
    elif phase == "test":
        fp = test_fp
        cols = ["id", "sequence"]
    else:
        raise ValueError("Unknown phase")
    with open(fp) as handle:
        records = extract_class_and_seq(list(SeqIO.parse(handle, 'fasta')))
        df = pd.DataFrame(records, columns=cols)
    return df


def get_loader(phase, fasta_fp=None, pad_len=600, batch_size=64):
    df = load_data_as_df(phase, fasta_fp)
    ohe_enc = seq2ohe(df.sequence, pad_len=pad_len)
    tar_enc = df.target
    sampler = get_weighted_sampler(tar_enc)
    tensor_seq = torch.tensor(ohe_enc)
    tensor_tar = torch.tensor(tar_enc).reshape(-1, 1)
    ds = TensorDataset(tensor_seq, tensor_tar)
    loader = DataLoader(ds, sampler=sampler, batch_size=batch_size)
    return loader

