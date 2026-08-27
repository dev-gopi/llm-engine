from training.data import _mixture_groups


def test_dataset_mixture_weights_are_dataset_level_probabilities() -> None:
    groups = _mixture_groups(
        ["data/small/train.jsonl", "data/large/train.jsonl"],
        [2, 8],
        {"dataset_weights": {"small": 0.25, "large": 0.75}},
    )
    assert groups == [(0, 2, 0.25), (2, 10, 0.75)]
