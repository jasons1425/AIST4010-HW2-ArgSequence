from torch.utils.data import WeightedRandomSampler
import numpy as np
import torch


# expect the target to be of pd.Series type
# ref: https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
def get_weighted_sampler(targets):
    class_weights = {target_label: 1. / cnt
                     for target_label, cnt in targets.value_counts().items()}
    samples_weight = np.array([class_weights[t] for t in targets])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler
