import hashlib
from inference.quantization import build_quantized_manifest, validate_quantized_manifest, architecture_fingerprint

def test_quantized_manifest_provenance_and_compatibility():
    config={'architecture':'MiniGPT','vocab_size':16,'hidden_size':8,'layers':1,'heads':2,'kv_heads':2,'ffn_hidden_size':16,'position_type':'rotary','max_position':32}
    manifest=build_quantized_manifest(fmt='gguf',config=config,calibration_sha256=hashlib.sha256(b'x').hexdigest(),calibration_tokens=10,bits=4)
    validate_quantized_manifest(manifest,config=config)
    assert manifest['model_config_fingerprint']==architecture_fingerprint(config)
