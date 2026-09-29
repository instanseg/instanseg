import numpy as np
import torch

from instanseg.utils.metrics import _robust_average_precision


def _square(offset: int) -> torch.Tensor:
    img = torch.zeros((1, 32, 32), dtype=torch.int32)
    img[0, offset:offset + 8, offset:offset + 8] = 1
    return img


def test_filtered_labels_stay_paired_with_predictions() -> None:
    # The first image is empty and gets dropped; the remaining pairs must stay aligned.
    empty = torch.zeros((1, 32, 32), dtype=torch.int32)
    labels = torch.stack([empty, _square(2), _square(20)])
    predicted = torch.stack([_square(10), _square(2), _square(20)])
    assert np.allclose(_robust_average_precision(labels, predicted, threshold=[0.5]), 1.0)


def test_sparse_labels_mask_predictions() -> None:
    # Predictions inside the unlabelled (-1) region should be ignored, not counted as false positives.
    label = _square(2)
    label[0, 16:, 16:] = -1
    predicted = _square(2)
    predicted[0, 20:28, 20:28] = 2
    result = _robust_average_precision(label[None].clone(), predicted[None].clone(), threshold=[0.5])
    assert np.allclose(result, 1.0)
