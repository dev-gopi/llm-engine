import pytest
import torch

from scripts.train_vision import loader_options


def test_in_process_loader_omits_worker_only_options():
    assert loader_options({"num_workers": 0}, torch.device("cpu")) == {
        "num_workers": 0, "pin_memory": False,
    }


def test_worker_prefetch_and_persistence_are_configurable():
    options = loader_options({"num_workers": 2, "prefetch_factor": 1,
                              "persistent_workers": True}, torch.device("cuda"))
    assert options == {"num_workers": 2, "pin_memory": True,
                       "prefetch_factor": 1, "persistent_workers": True}


@pytest.mark.parametrize("config", [{"num_workers": -1},
                                    {"num_workers": 1, "prefetch_factor": 0}])
def test_invalid_worker_resources(config):
    with pytest.raises(ValueError):
        loader_options(config, torch.device("cpu"))
