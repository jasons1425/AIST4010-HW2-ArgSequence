import torch
from data.loading import load_data_as_df, get_loader
from helper.process import train_model
from model.protcnn import ProtCNN


# only select the data belonging to arg class
PAD_LEN = 800
BATCH_SIZE = 512
df_train, df_valid = load_data_as_df("train"), load_data_as_df("valid")
df_train["isarg"], df_valid["isarg"] = (df_train.target != 0), (df_valid.target != 0)
train_cls, valid_cls = df_train.loc[df_train.isarg], df_valid.loc[df_valid.isarg]
train_loader_cls = get_loader(train_cls.sequence, train_cls.target - 1, pad_len=PAD_LEN, batch_size=BATCH_SIZE)
valid_loader_cls = get_loader(valid_cls.sequence, valid_cls.target - 1, pad_len=PAD_LEN, batch_size=BATCH_SIZE)
dataloaders = {"train": train_loader_cls, "valid": valid_loader_cls}


# setting up objects for training
IN_DIM, OUT_DIM = 23, 14
IN_KSIZE, RES_KSIZE = 5, 3
RES_DIM, RES_BLKSIZE, RES_DIL = 128, 3, 2
FC_BLKS = [34048, 800]
ACT, DROPOUT = torch.nn.SiLU, 0.7
LR, MOMENTUM, DECAY = 1e-3, 0.9, 0.01
HALF = True
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ProtCNN(IN_DIM, OUT_DIM, IN_KSIZE,
                RES_DIM, RES_KSIZE, RES_BLKSIZE, RES_DIL,
                FC_BLKS, act=ACT, dropout=DROPOUT)
if HALF:
    model = model.half()
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LR, momentum=MOMENTUM, weight_decay=DECAY)

# train the model
EPOCHS = 100
best_model, losses, accs = train_model(model, dataloaders, criterion,
                                       optimizer, EPOCHS, device, half=HALF)

torch.save(model.state_dict(),  f"argclass.pth")



