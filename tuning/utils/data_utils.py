import torch
from torch.utils.data import SequentialSampler

from datasets import Dataset

def _get_train_sampler(
    self, train_dataset: Dataset | None = None
) -> torch.utils.data.Sampler | None:
    return SequentialSampler(train_dataset)
