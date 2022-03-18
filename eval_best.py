import torch
import torch.nn.functional as F
import pandas as pd
from helper.evaluation import evaluate_f1score
from data.loading import load_data_as_df, get_loader
from model.protcnn import ProtCNNftEmbedding, ProtCNN


# define the test data and the data path to export the results
test_fp = r"D:\Documents\datasets\AIST4010\arg sequences\data\test.fasta"
store_fp = r"submission.csv"
assert test_fp is not None


# only select the data belonging to arg class
PAD_LEN = 100
BATCH_SIZE = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
df_valid = load_data_as_df("valid")
df_valid['isarg'] = (df_valid.target != 0)
df_test = load_data_as_df("test", fasta_fp=test_fp)
valid_loader_cls = get_loader(df_valid.sequence, df_valid.target,
                              pad_len=PAD_LEN, batch_size=BATCH_SIZE, label_enc=True, add_sampler=False)
valid_loader_isarg = get_loader(df_valid.sequence, df_valid.isarg,
                                pad_len=PAD_LEN, batch_size=BATCH_SIZE, label_enc=True, add_sampler=False)
test_loader = get_loader(df_test.sequence, df_test.index, pad_len=PAD_LEN, batch_size=BATCH_SIZE,
                         label_enc=True, add_sampler=False, shuffle=False)

# isarg model
isarg_model_weights = r"trials/isarg_embed9628.pth"
isarg_model_config = {
    "in_dim": 24,
    "out_dim": 2,
    "in_ksize": 3,
    "res_dim": 128,
    "res_ksize": 3,
    "resblk_size": 5,
    "res_dil": 2,
    "fc_blks": [4224, 400],
    "enc_dim": 1024,
    "seq_len": 100,
    "act": torch.nn.ReLU,
    "dropout": 0.5,
    "add_init_dropout": False,
}
isarg_model = ProtCNNftEmbedding(**isarg_model_config).to(device)
isarg_model.load_state_dict(torch.load(isarg_model_weights))
precision, recall, f1 = evaluate_f1score(isarg_model, valid_loader_isarg, device)
print(f"isarg validation: precision is {precision:.5f}, recall is {recall:.5f}, f1-score is {f1:.5f}")
use_isarg = False


# argclass model
argcls_model_weights = r"trials/argclass_embed9711.pth"
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
precision, recall, f1 = evaluate_f1score(argcls_model, valid_loader_cls, device, use_isarg=None)
print(f"argcls1 validation: precision is {precision:.5f}, recall is {recall:.5f}, f1-score is {f1:.5f}")


# argclass model
argcls_model_weights2 = r"trials/argclass_embed9699.pth"
argcls_model_config2 = {
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
argcls_model2 = ProtCNNftEmbedding(**argcls_model_config2).to(device)
argcls_model2.load_state_dict(torch.load(argcls_model_weights2))
precision, recall, f1 = evaluate_f1score(argcls_model2, valid_loader_cls, device, use_isarg=None)
print(f"argcls2 validation: precision is {precision:.5f}, recall is {recall:.5f}, f1-score is {f1:.5f}")
use_argcls_model2 = True


isarg_model.eval()
argcls_model.eval()
get_prob_isarg = lambda x: F.softmax(x, dim=1)[:, 1]
results = pd.DataFrame({'id':[], 'label':[]})
for inputs, ids in test_loader:
    inputs = inputs.to(device)
    outputs_isarg, outputs_argcls, outputs_argcls2 = isarg_model(inputs), argcls_model(inputs), argcls_model2(inputs)
    if use_argcls_model2:
        combined_outputs = (outputs_argcls + outputs_argcls2)/2
        (__, preds_argcls) = torch.topk(combined_outputs, 1, dim=1)
    else:
        (__, preds_argcls) = torch.topk(outputs_argcls, 1, dim=1)
    if use_isarg:
        preds_isarg = (get_prob_isarg(outputs_isarg) >= 0.05).reshape(-1, 1)
        final_preds = ((preds_argcls * preds_isarg) - 1) % 15
    else:
        final_preds = (preds_argcls - 1) % 15
    ids_to_write, preds_to_write = df_test.id.loc[ids.numpy().flatten()], final_preds.cpu().numpy().flatten()
    results = pd.concat([results, pd.DataFrame({'id': ids_to_write,
                                                'label': preds_to_write})], ignore_index=True)

results = results.astype({'label': 'int32'})
results.to_csv(store_fp, index=False)
