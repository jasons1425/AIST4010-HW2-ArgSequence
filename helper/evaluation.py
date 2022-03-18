import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score


def evaluate_f1score(model, dataloader, device, half=True, to_long=True, pre_input_process=None, use_isarg=None):
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            if half:
                inputs = inputs.half()
            inputs = inputs.to(device)
            if to_long:
                inputs = inputs.long()
            if pre_input_process:
                inputs = pre_input_process(inputs)
            outputs = model(inputs)
            _, pred = torch.topk(outputs, 1, dim=1)
            if use_isarg:
                get_prob_isarg = lambda x: F.softmax(x, dim=1)[:, 1]
                outputs_isarg = use_isarg(inputs)
                preds_isarg = (get_prob_isarg(outputs_isarg) >= 0.05).reshape(-1, 1)
                pred = preds_isarg * pred
            preds.extend(pred.flatten().tolist())
            targets.extend(labels.flatten().tolist())
    precision = precision_score(targets, preds, average="macro")
    recall = recall_score(targets, preds, average="macro")
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1
