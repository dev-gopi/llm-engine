from model.gpt import MiniGPT
from training.pipeline_minigpt import MiniGPTPipelinePartition, balanced_layer_range


def test_balanced_pipeline_ranges_cover_all_layers():
    ranges = [balanced_layer_range(10, 3, i) for i in range(3)]
    assert ranges == [(0, 4), (4, 7), (7, 10)]


def test_pipeline_partition_owns_only_local_blocks():
    model = MiniGPT(vocab_size=32, dim=16, layers=4, heads=4, max_pos=16)
    first = MiniGPTPipelinePartition(model, stage=0, stages=2)
    last = MiniGPTPipelinePartition(model, stage=1, stages=2)
    assert len(first.blocks) == 2 and len(last.blocks) == 2
    assert first.tok is not None and first.head is None
    assert last.tok is None and last.head is not None
