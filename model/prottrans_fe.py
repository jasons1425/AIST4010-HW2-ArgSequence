import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from model.protcnn import ResidualBlock
import re
import numpy as np
import os
import requests
from tqdm.auto import tqdm


ProtTrans_Model = None
ProtTrans_Tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_prottrans_model():
    global ProtTrans_Model, device
    if ProtTrans_Model is None:
        ProtTrans_Model = BertModel.from_pretrained("Rostlab/prot_bert")
        for params in ProtTrans_Model.parameters():
            params.required_grad = False
        ProtTrans_Model = ProtTrans_Model.to(device).eval()
    return ProtTrans_Model


def get_prottrans_tokenizer():
    global ProtTrans_Tokenizer
    if ProtTrans_Tokenizer is None:
        ProtTrans_Tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert",
                                                            do_lower_case=False)
    return ProtTrans_Tokenizer


class ProtTransCNN(nn.Module):
    def __init__(self, enc_dim, out_dim, in_ksize, fc_blks=(None, 1024), pool=nn.MaxPool1d, act=nn.ReLU,
                 norm=nn.BatchNorm1d, dropout=0, init=nn.init.kaiming_normal_):
        super(ProtTransCNN, self).__init__()
        conv1 = nn.Conv1d(1024, enc_dim, in_ksize, padding="same")
        assert fc_blks[0] is not None, "Please specify the in_dim of first FC layer"
        fc_start = nn.Linear(fc_blks[0], fc_blks[1])
        fc_last = nn.Linear(fc_blks[-1], out_dim)
        if init:
            init(conv1.weight)
            init(fc_start.weight)
            init(fc_last.weight)
        fc_layers = []
        for idx in range(2,  len(fc_blks)):
            fc = nn.Linear(fc_blks[idx-1], fc_blks[idx])
            if init:
                init(fc.weight)
            fc_layers.extend([fc, act()])
        self.dropout = nn.Dropout(dropout)
        self.stack = nn.Sequential(
            norm(1024),
            conv1,
            act(),
            pool(3),
            nn.Dropout(dropout),
            nn.Flatten(),
            fc_start,
            act(),
            *fc_layers,
            nn.Dropout(dropout),
            fc_last
        )

    def forward(self, embedding):
        embedding = self.dropout(embedding)  # randomly 'mask' the sequences at some positions
        embedding = embedding.permute(0, 2, 1)
        return self.stack(embedding)


class ProtTransCNNv2(nn.Module):
    def __init__(self, out_dim, fc_blks=(None, 1024), act=nn.ReLU,
                 norm=nn.BatchNorm1d, dropout=0, init=nn.init.kaiming_normal_):
        super(ProtTransCNNv2, self).__init__()
        assert fc_blks[0] is not None, "Please specify the in_dim of first FC layer"
        fc_start = nn.Linear(fc_blks[0], fc_blks[1])
        fc_last = nn.Linear(fc_blks[-1], out_dim)
        if init:
            init(fc_start.weight)
            init(fc_last.weight)
        fc_layers = []
        for idx in range(2,  len(fc_blks)):
            fc = nn.Linear(fc_blks[idx-1], fc_blks[idx])
            if init:
                init(fc.weight)
            fc_layers.extend([fc, act()])
        self.dropout = nn.Dropout(dropout)
        self.stack = nn.Sequential(
            # norm(2048),
            nn.Flatten(),
            fc_start,
            act(),
            *fc_layers,
            nn.Dropout(dropout),
            fc_last
        )

    def forward(self, embedding):
        seq_len = embedding.size(1)
        cls_tokens = embedding[:, 0, :]
        word_tokens_avg = torch.mean(embedding[:, 1:seq_len-1, :], dim=1)
        embedding = torch.cat((cls_tokens, word_tokens_avg), dim=1)
        embedding = self.dropout(embedding)  # randomly 'mask' the sequences at some positions
        # embedding = embedding.permute(0, 2, 1)
        return self.stack(embedding)


