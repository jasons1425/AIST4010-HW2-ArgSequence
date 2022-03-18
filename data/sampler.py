from torch.utils.data import WeightedRandomSampler
import numpy as np
import pandas as pd
import torch
from typing import Callable


# expect the target to be of pd.Series type
# ref: https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
def get_weighted_sampler(targets):
    class_weights = {target_label: 1. / cnt
                     for target_label, cnt in targets.value_counts().items()}
    samples_weight = np.array([class_weights[t] for t in targets])
    samples_weight = torch.from_numpy(samples_weight).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler


# modified the _get_labels() method so that it fits the tensor dataset
# ref: https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, labels, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        self.labels = pd.Series(labels)

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self.labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        return self.labels

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
