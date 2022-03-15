from Bio import SeqIO
from data.preprocess import TARGET_DICTIONARY, seq2ohe, label_encode, prottrans_preprocess
from data.sampler import get_weighted_sampler
from torch.utils.data import DataLoader, TensorDataset
from model.prottrans_fe import get_prottrans_tokenizer
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
    cols = ["arg_class", "target", "sequence"] if phase != "test" else ["id", "sequence"]
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
        records = extract_class_and_seq(list(SeqIO.parse(handle, 'fasta')),
                                        is_test=(phase == "test"))
        df = pd.DataFrame(records, columns=cols)
    return df


def get_loader(seqs, tar_enc, pad_len=600, batch_size=256,
               label_enc=False, add_sampler=True, shuffle=True):
    if label_enc:
        seq_enc = label_encode(seqs, pad_len=pad_len)
        tensor_seq = torch.tensor(seq_enc).permute(0, 1)
    else:
        seq_enc = seq2ohe(seqs, pad_len=pad_len)
        tensor_seq = torch.tensor(seq_enc).permute(0, 2, 1)
    tensor_tar = torch.tensor(tar_enc, dtype=torch.int64).reshape(-1, 1)
    if add_sampler:
        sampler = get_weighted_sampler(tar_enc)
        shuffle = False
    else:
        sampler = None
    ds = TensorDataset(tensor_seq, tensor_tar)
    loader = DataLoader(ds, sampler=sampler, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_loader_prottrans(seqs, tar_enc, pad_len=100, batch_size=256,
                         add_sampler=True, shuffle=True):
    padded_seq = seqs.apply(lambda x: prottrans_preprocess(x, pad_len=pad_len))
    tokenizer = get_prottrans_tokenizer()
    ids = tokenizer.batch_encode_plus(padded_seq, add_special_tokens=False,
                                      pad_to_max_length=True)
    input_ids = torch.tensor(ids['input_ids'])
    attention_masks = torch.tensor(ids['attention_mask'])
    tensor_tar = torch.tensor(tar_enc, dtype=torch.int64).reshape(-1, 1)
    ds = TensorDataset(input_ids, attention_masks, tensor_tar)
    if add_sampler:
        sampler = get_weighted_sampler(tar_enc)
        shuffle = False
    else:
        sampler = None
    loader = DataLoader(ds, sampler=sampler, batch_size=batch_size, shuffle=shuffle)
    return loader






