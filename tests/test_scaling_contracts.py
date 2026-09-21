import pytest
from inference.tensor_parallel import ParallelTopologyContract
from training.multinode import DistributedTopology, validate_3d_parallel_contract

def test_parallel_contract():
    c=ParallelTopologyContract(2,2,1); c.validate(attention_heads=8,num_experts=1); assert c.world_size==4

def test_multinode_contract():
    t=DistributedTopology(4,0,0,2,0,2,'node0',29500)
    assert validate_3d_parallel_contract(t,tensor_parallel=2,pipeline_parallel=2,expert_parallel=1)['world_size']==4
