import torch
from data.loading import load_data_as_df, get_loader, get_loader_prottrans
from helper.process import train_model_prottrans
from model.prottrans_fe import ProtTransCNNv2, get_prottrans_model


# only select the data belonging to arg class
PAD_LEN = 200
BATCH_SIZE = 64
df_train, df_valid = load_data_as_df("train"), load_data_as_df("valid")
df_train["isarg"], df_valid["isarg"] = (df_train.target != 0), (df_valid.target != 0)
train_loader_cls = get_loader_prottrans(df_train.sequence, df_train.isarg, pad_len=PAD_LEN,
                                        batch_size=BATCH_SIZE, add_sampler=False, special_tokens=True)
valid_loader_cls = get_loader_prottrans(df_valid.sequence, df_valid.isarg, pad_len=PAD_LEN,
                                        batch_size=BATCH_SIZE, add_sampler=False, special_tokens=True)
dataloaders = {"train": train_loader_cls, "valid": valid_loader_cls}


# setting up objects for training
ENC_DIM = 256
IN_DIM, OUT_DIM = 23+1, 2
IN_KSIZE = 3
FC_BLKS = [2048, 512]
ACT, DROPOUT = torch.nn.ReLU, 0.5
LR, MOMENTUM, DECAY = 1e-3, 0.9, 0.01
HALF = True
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = get_prottrans_model()
model = ProtTransCNNv2(OUT_DIM, FC_BLKS,
                       act=ACT, dropout=DROPOUT, init=torch.nn.init.xavier_normal_)
if HALF:
    model = model.half()
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                              lr_lambda=lambda epoch: 1 if epoch < 40 else (0.1 if epoch < 80 else 0.01))


# train the model
EPOCHS = 30
best_model, losses, accs = train_model_prottrans(model, dataloaders, criterion, optimizer, EPOCHS,
                                                 device, embedder, half=HALF, to_long=True, scheduler=None)

torch.save(model.state_dict(),  f"argclass_prottransv2.pth")