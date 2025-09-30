from __future__ import annotations
from collections import Counter
from typing import Sequence, Optional, List
import torch
from torch.utils.data import Sampler, Subset

def _infer_all_labels(dataset) -> List[int]:

    if isinstance(dataset, Subset):
        parent_labels = _infer_all_labels(dataset.dataset)
        return [int(parent_labels[i]) for i in dataset.indices]


    if hasattr(dataset, "get_labels") and callable(dataset.get_labels):
        return list(map(int, dataset.get_labels()))

    for attr in ("targets", "labels", "y"):
        if hasattr(dataset, attr):
            val = getattr(dataset, attr)
            if callable(val):
                val = val()
            try:
                return list(map(int, list(val)))
            except Exception:
                pass

    raise ValueError(
        "Cannot infer labels from dataset. "
        "Provide dataset.get_labels(), or targets/labels/y attribute."
    )

class ImbalancedDatasetSampler(Sampler[int]):
    
    def __init__(
        self,
        dataset,
        indices: Optional[Sequence[int]] = None,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        smoothing: float = 0.0,
    ):
        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else list(indices)
        self.num_samples = len(self.indices) if num_samples is None else int(num_samples)
        self.replacement = bool(replacement)

        all_labels = _infer_all_labels(dataset)
        labels = [int(all_labels[i]) for i in self.indices]

        counts = Counter(labels)
        self.weights = torch.tensor(
            [1.0 / (counts[l] + float(smoothing)) for l in labels],
            dtype=torch.double,
        )

    def __iter__(self):
        chosen = torch.multinomial(self.weights, self.num_samples, replacement=self.replacement)
        return (self.indices[i] for i in chosen.tolist())

    def __len__(self) -> int:
        return self.num_samples


def make_class_weights_from_labels(labels: Sequence[int], smoothing: float = 0.0) -> torch.Tensor:

    counts = Counter(map(int, labels))
    max_cnt = max(counts.values())

    class_ids = sorted(counts.keys())
    weights = [max_cnt / (counts[c] + float(smoothing)) for c in class_ids]
    return torch.tensor(weights, dtype=torch.float)
