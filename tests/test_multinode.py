import pytest

from training.distributed import DistributedTrainer
from training.multinode import topology_from_environment, validate_fsdp_topology


def test_multinode_topology_validation() -> None:
    topology = topology_from_environment({
        "WORLD_SIZE": "8", "RANK": "5", "LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "4",
        "MASTER_ADDR": "node0", "MASTER_PORT": "29500",
    })
    assert topology.nodes == 2
    assert topology.node_rank == 1


def test_multinode_topology_rejects_partial_nodes() -> None:
    with pytest.raises(ValueError, match="divisible"):
        topology_from_environment({
            "WORLD_SIZE": "3", "RANK": "0", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "2",
            "MASTER_ADDR": "node0", "MASTER_PORT": "29500",
        })


def test_fsdp_topology_preflight_requires_declared_rank_and_node_count() -> None:
    topology = topology_from_environment({
        "WORLD_SIZE": "4", "RANK": "3", "LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "2",
        "MASTER_ADDR": "node0", "MASTER_PORT": "29500",
    })
    validate_fsdp_topology(topology, minimum_world_size=4, required_nodes=2)
    with pytest.raises(ValueError, match="at least 8 ranks"):
        validate_fsdp_topology(topology, minimum_world_size=8)
    with pytest.raises(ValueError, match="requires 1 nodes"):
        validate_fsdp_topology(topology, required_nodes=1)


def test_distributed_trainer_exposes_fsdp_preflight(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("MASTER_ADDR", "node0")
    monkeypatch.setenv("MASTER_PORT", "29500")

    DistributedTrainer.preflight_fsdp(minimum_world_size=4, required_nodes=2)
