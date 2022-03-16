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
    def __init__(self, out_dim, in_ksize, res_dim, res_ksize, resblk_size,
                 res_dil=1, fc_blks=(None, 1024), pool=nn.MaxPool1d, act=nn.ReLU,
                 norm=nn.BatchNorm1d, dropout=0, init=nn.init.kaiming_normal_):
        super(ProtTransCNN, self).__init__()
        self.encoder = get_prottrans_model()
        conv1 = nn.Conv1d(1024, res_dim, in_ksize, padding="same")
        assert fc_blks[0] is not None, "Please specify the in_dim of first FC layer"
        fc_start = nn.Linear(fc_blks[0], fc_blks[1])
        fc_last = nn.Linear(fc_blks[-1], out_dim)
        if init:
            init(conv1.weight)
            init(fc_start.weight)
            init(fc_last.weight)
        res_layers = [ResidualBlock(res_dim, res_ksize, res_dil, act, norm, init=init)
                      for _ in range(resblk_size)]
        fc_layers = []
        for idx in range(2,  len(fc_blks)):
            fc = nn.Linear(fc_blks[idx-1], fc_blks[idx])
            if init:
                init(fc.weight)
            fc_layers.extend([fc, act()])
        self.dropout = nn.Dropout(dropout)
        self.stack = nn.Sequential(
            conv1,
            act(),
            *res_layers,
            pool(3),
            nn.Dropout(dropout),
            nn.Flatten(),
            fc_start,
            act(),
            *fc_layers,
            nn.Dropout(dropout),
            fc_last
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            embedding = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        embedding = self.dropout(embedding)  # randomly 'mask' the sequences at some positions
        embedding = embedding.permute(0, 2, 1)
        return self.stack(embedding)


