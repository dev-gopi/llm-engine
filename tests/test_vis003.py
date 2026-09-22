from multimodal.model import validate_modality_contracts
from vision.encoder import ModalityContract


def test_audio_video_contracts_are_explicit():
    contracts=[ModalityContract('audio','audio-v1',512),ModalityContract('video','video-v1',768)]
    validate_modality_contracts(contracts)
