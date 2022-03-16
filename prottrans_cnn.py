import torch
from data.loading import load_data_as_df, get_loader, get_loader_prottrans
from helper.process import train_model_prottrans
from model.prottrans_fe import ProtTransCNN


# only select the data belonging to arg class
PAD_LEN = 100
BATCH_SIZE = 128
df_train, df_valid = load_data_as_df("train"), load_data_as_df("valid")
df_train["isarg"], df_valid["isarg"] = (df_train.target != 0), (df_valid.target != 0)
train_loader_cls = get_loader_prottrans(df_train.sequence, df_train.isarg, pad_len=PAD_LEN,
                                        batch_size=BATCH_SIZE, add_sampler=False)
valid_loader_cls = get_loader_prottrans(df_valid.sequence, df_valid.isarg, pad_len=PAD_LEN,
                                        batch_size=BATCH_SIZE, add_sampler=False)
dataloaders = {"train": train_loader_cls, "valid": valid_loader_cls}


# setting up objects for training
ENC_DIM = 512
IN_DIM, OUT_DIM = 23+1, 2
IN_KSIZE, RES_KSIZE = 3, 3
RES_DIM, RES_BLKSIZE, RES_DIL = 64, 1, 2
FC_BLKS = [2112, 200, 400]
ACT, DROPOUT = torch.nn.ReLU, 0.5
LR, MOMENTUM, DECAY = 1e-3, 0.9, 0.01
HALF = True
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ProtTransCNN(OUT_DIM, IN_KSIZE, RES_DIM, RES_KSIZE,
                     RES_BLKSIZE, RES_DIL, FC_BLKS,
                     act=ACT, dropout=DROPOUT)
if HALF:
    model = model.half()
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                              lr_lambda=lambda epoch: 1 if epoch < 200 else 0.1)


# train the model
EPOCHS = 30
best_model, losses, accs = train_model_prottrans(model, dataloaders, criterion, optimizer, EPOCHS,
                                                 device, half=HALF, to_long=True, scheduler=None)

torch.save(model.state_dict(),  f"isarg_prottrans.pth")