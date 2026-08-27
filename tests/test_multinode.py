import pytest

from training.multinode import topology_from_environment


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
