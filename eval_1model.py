import torch
import pandas as pd
from data.loading import load_data_as_df, get_loader
from model.protcnn import ProtCNNftEmbedding, ProtCNN


# define the test data and the data path to export the results
test_fp = r""
store_fp = r"submission.csv"


# only select the data belonging to arg class
PAD_LEN = 100
BATCH_SIZE = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
df_test = load_data_as_df("test", fasta_fp=test_fp)
test_loader = get_loader(df_test.sequence, df_test.index, pad_len=PAD_LEN, batch_size=BATCH_SIZE,
                         label_enc=True, add_sampler=False, shuffle=False)


# argclass model
argcls_model_weights = r"trials/argclass_embed9699.pth"
argcls_model_config = {
    "in_dim": 24,
    "out_dim": 15,
    "in_ksize": 3,
    "res_dim": 128,
    "res_ksize": 3,
    "resblk_size": 2,
    "res_dil": 2,
    "fc_blks": [4224, 1000],
    "enc_dim": 512,
    "seq_len": 100,
    "act": torch.nn.ReLU,
    "dropout": 0.6,
}
argcls_model = ProtCNNftEmbedding(**argcls_model_config).to(device)
argcls_model.load_state_dict(torch.load(argcls_model_weights))
argcls_model.eval()

results = pd.DataFrame({'id':[], 'label':[]})
for inputs, ids in test_loader:
    inputs = inputs.to(device)
    outputs_argcls = argcls_model(inputs)
    (__, preds_argcls) = torch.topk(outputs_argcls, 1, dim=1)
    final_preds = (preds_argcls - 1) % 15
    ids_to_write, preds_to_write = df_test.id.loc[ids.numpy().flatten()], final_preds.cpu().numpy().flatten()
    results = pd.concat([results, pd.DataFrame({'id': ids_to_write,
                                                'label': preds_to_write})], ignore_index=True)

results = results.astype({'label': 'int32'})
results.to_csv(store_fp, index=False)
