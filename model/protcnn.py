import torch.nn as nn


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
            pool(2),
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
