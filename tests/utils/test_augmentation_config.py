import collections

import pytest
import torch

from instanseg.utils.augmentation_config import get_augmentation_dict
from instanseg.utils.augmentations import Augmentations


DROP_PROBABILITIES = {"CPDMI_2023": 0.3, "TissueNet": 0.2}


def _minimal_config(mapping=None):
    return get_augmentation_dict(
        dim_in=0,
        nuclei_channel=None,
        amount=0.5,
        augmentation_type="minimal",
        dataset_channel_drop_probabilities=mapping,
    )


def test_minimal_channel_suppression_is_opt_in_and_train_only():
    default_config = _minimal_config()
    mapped_config = _minimal_config(DROP_PROBABILITIES)

    assert "channel_suppress" not in default_config["train"]["Fluorescence"]
    assert mapped_config["train"]["Fluorescence"]["channel_suppress"] == [
        1,
        DROP_PROBABILITIES,
    ]
    assert "channel_suppress" not in mapped_config["test"]["Fluorescence"]


def test_mapping_overrides_heavy_channel_suppression():
    default_config = get_augmentation_dict(
        dim_in=0,
        nuclei_channel=None,
        amount=0.5,
        augmentation_type="heavy",
    )
    mapped_config = get_augmentation_dict(
        dim_in=0,
        nuclei_channel=None,
        amount=0.5,
        augmentation_type="heavy",
        dataset_channel_drop_probabilities=DROP_PROBABILITIES,
    )

    assert default_config["train"]["Fluorescence"]["channel_suppress"] == [1, 0.3]
    assert mapped_config["train"]["Fluorescence"]["channel_suppress"] == [
        1,
        DROP_PROBABILITIES,
    ]


@pytest.mark.parametrize(
    "mapping, message",
    [
        ([], "must be a dictionary"),
        ({1: 0.2}, "keys must be strings"),
        ({"TissueNet": "0.2"}, "must be numeric"),
        ({"TissueNet": -0.1}, "between 0 and 1"),
        ({"TissueNet": 1.1}, "between 0 and 1"),
    ],
)
def test_invalid_dataset_channel_drop_probabilities_are_rejected(mapping, message):
    with pytest.raises(ValueError, match=message):
        _minimal_config(mapping)


def test_channel_suppression_probability_endpoints_and_unknown_dataset():
    augmenter = Augmentations(channel_invariant=True)
    image = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(4, 4, 4)
    mapping = {"drop_all": 1.0, "drop_none": 0.0}

    dropped, _ = augmenter.channel_suppress(
        image, amount=mapping, metadata={"parent_dataset": "drop_all"}
    )
    retained, _ = augmenter.channel_suppress(
        image, amount=mapping, metadata={"parent_dataset": "drop_none"}
    )
    unknown, _ = augmenter.channel_suppress(
        image, amount=mapping, metadata={"parent_dataset": "unlisted"}
    )

    assert dropped.shape[0] == 1
    assert torch.equal(retained, image)
    assert torch.equal(unknown, image)


def test_parent_dataset_metadata_selects_the_probability():
    augmentation_dict = {
        "Fluorescence": collections.OrderedDict(
            [
                ("to_tensor", [1]),
                ("channel_suppress", [1, {"CPDMI_2023": 1.0, "TissueNet": 0.0}]),
            ]
        )
    }
    augmenter = Augmentations(
        augmentation_dict=augmentation_dict,
        channel_invariant=True,
    )
    image = torch.ones(4, 8, 8)
    labels = torch.zeros(8, 8)
    base_metadata = {
        "image_modality": "Fluorescence",
        "nuclei_channels": [0],
        "pixel_size": 0.5,
    }

    cpdmi, _ = augmenter(
        image, labels, {**base_metadata, "parent_dataset": "CPDMI_2023"}
    )
    tissuenet, _ = augmenter(
        image, labels, {**base_metadata, "parent_dataset": "TissueNet"}
    )

    assert cpdmi.shape[0] == 1
    assert tissuenet.shape[0] == 4
