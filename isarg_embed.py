import torch
from data.loading import load_data_as_df, get_loader
from helper.process import train_model
from model.protcnn import ProtCNNftEmbedding


# only select the data belonging to arg class
PAD_LEN = 100
BATCH_SIZE = 64
df_train, df_valid = load_data_as_df("train"), load_data_as_df("valid")
df_train["isarg"], df_valid["isarg"] = (df_train.target != 0), (df_valid.target != 0)
train_loader_cls = get_loader(df_train.sequence, df_train.isarg,
                              pad_len=PAD_LEN, batch_size=BATCH_SIZE, label_enc=True, add_sampler=False)
valid_loader_cls = get_loader(df_valid.sequence, df_valid.isarg,
                              pad_len=PAD_LEN, batch_size=BATCH_SIZE, label_enc=True, add_sampler=False)
dataloaders = {"train": train_loader_cls, "valid": valid_loader_cls}


# setting up objects for training
ENC_DIM = 1024
IN_DIM, OUT_DIM = 23+1, 2
IN_KSIZE, RES_KSIZE = 3, 3
RES_DIM, RES_BLKSIZE, RES_DIL = 128, 5, 2
FC_BLKS = [4224, 256]
ACT, DROPOUT = torch.nn.ReLU, 0.5
LR, MOMENTUM, DECAY = 1e-3, 0.9, 0.01
HALF = True
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ProtCNNftEmbedding(IN_DIM, OUT_DIM, IN_KSIZE, RES_DIM, RES_KSIZE,
                           RES_BLKSIZE, RES_DIL, FC_BLKS, ENC_DIM, PAD_LEN,
                           act=ACT, dropout=DROPOUT, init=torch.nn.init.xavier_normal_)
# model.load_state_dict(torch.load(r"trials/isarg_embed9599.pth"))
if HALF:
    model = model.half()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                              lr_lambda=lambda epoch: 1 if epoch < 200 else 0.1)


# train the model
EPOCHS = 200
best_model, losses, accs = train_model(model, dataloaders, criterion, optimizer,
                                       EPOCHS, device, half=HALF, to_long=True, scheduler=None)

torch.save(model.state_dict(),  f"isarg_embed.pth")
