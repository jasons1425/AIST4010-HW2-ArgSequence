from torch.autograd import Variable
import torch.nn as nn
import torch
import math


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, ksize, dilation=1,
                 act=nn.ReLU, norm=nn.BatchNorm1d, init=nn.init.kaiming_normal_):
        super(ResidualBlock, self).__init__()
        conv1 = nn.Conv1d(input_dim, input_dim, ksize,
                          dilation=dilation, padding=dilation*(ksize-1)//2)
        conv2 = nn.Conv1d(input_dim, input_dim, ksize, padding="same")
        if init:
            init(conv1.weight)
            init(conv2.weight)
        self.stack = nn.Sequential(
            norm(input_dim),
            act(),
            conv1,
            norm(input_dim),
            act(),
            conv2
        )

    def forward(self, x):
        output = self.stack(x)
        return output + x


class ProtCNN(nn.Module):
    def __init__(self, in_dim, out_dim, in_ksize, res_dim, res_ksize, resblk_size,
                 res_dil=1, fc_blks=(None, 1024,), pool=nn.MaxPool1d, act=nn.ReLU,
                 norm=nn.BatchNorm1d, dropout=0, init=nn.init.kaiming_normal_):
        super(ProtCNN, self).__init__()
        conv1 = nn.Conv1d(in_dim, res_dim, in_ksize, padding="same")
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

    def forward(self, x):
        output = self.stack(x)
        return output


class ProtCNNftEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, in_ksize, res_dim, res_ksize, resblk_size, res_dil=1,
                 fc_blks=(None, 1024,), enc_dim=512, seq_len=400, pool=nn.MaxPool1d, act=nn.ReLU,
                 norm=nn.BatchNorm1d, dropout=0, init=nn.init.kaiming_normal_, add_init_dropout = True):
        super(ProtCNNftEmbedding, self).__init__()
        self.embed = Embedder(in_dim, enc_dim)
        self.position_enc = PositionalEncoder(enc_dim, seq_len)
        conv1 = nn.Conv1d(enc_dim, res_dim, in_ksize, padding="same")
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
        stack = [
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
        ]
        if add_init_dropout:
            stack.insert(0, nn.Dropout(dropout))
        self.stack = nn.Sequential(*stack)

    def forward(self, x):
        x = self.embed(x)
        x = self.position_enc(x).permute(0, 2, 1)
        output = self.stack(x)
        return output
