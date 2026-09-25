import json
from pathlib import Path
import torch
from datasets.collator import Collator
from datasets.loader import LazyJSONLDataset
from model.gpt import MiniGPT
from model.loss import CausalLanguageModelLoss
from model.vocabulary import adapt_config_to_tokenizer, checkpoint_tokenizer_options
from optim.adamw import build_adamw
from training.checkpoint import load_checkpoint, save_checkpoint
from training.trainer import Trainer
from tokenizer.encoder import Tokenizer
from utils.config import load_yaml

torch.set_num_threads(4)
torch.manual_seed(42)
tok = Tokenizer.load('data/tokenizer')
cfg = adapt_config_to_tokenizer(load_yaml('configs/model.gpu.yaml'), tok)
model = MiniGPT.from_config(cfg, device='cpu')
payload = load_checkpoint('checkpoints/pretraining/latest.pt', model, use_ema=False,
                          restore_rng=False, **checkpoint_tokenizer_options(tok))
source_step = payload['step']
metadata = dict(payload.get('metadata', {}))
metadata.update(model_config=cfg, tokenizer_fingerprint=tok.fingerprint,
                source_checkpoint='checkpoints/pretraining/latest.pt', source_step=source_step)
del payload
report = {'source_step': source_step, 'parameter_count': model.num_parameters(),
          'device': 'cpu', 'cuda_available': torch.cuda.is_available(), 'checks': {}, 'steps': [],
          'note': 'Four optimizer updates validate execution only, not model quality improvement.'}
report['checks']['all_parameters_finite'] = all(bool(torch.isfinite(p).all()) for p in model.parameters())
assert report['checks']['all_parameters_finite']
model.eval()
ids = torch.tensor([tok.encode('The little girl found a puppy.', add_bos=True)])
with torch.inference_mode():
    full = model(ids)
    _, cache = model(ids[:, :-1], use_cache=True)
    incremental = model(ids[:, -1:], past_key_values=cache)
    torch.testing.assert_close(full[:, -1:], incremental, rtol=1e-4, atol=1e-4)
    changed = ids.clone()
    changed[:, -1] = tok.token_to_id('<|eos|>')
    torch.testing.assert_close(full[:, :-1], model(changed)[:, :-1], rtol=1e-4, atol=1e-4)
report['checks'].update(cache_matches_full_forward=True, causal_attention=True)
base = Path('checkpoints/pretraining-pilot/base.pt')
if base.exists():
    raise ValueError('baseline exists; refusing to overwrite')
save_checkpoint(base, model, step=source_step, metadata=metadata)
report['frozen_base_checkpoint'] = str(base)
loss_fn = CausalLanguageModelLoss.from_config(load_yaml('configs/pretraining.improved.gpu.yaml'))
optimizer = build_adamw(model, learning_rate=1e-5)
trainer = Trainer(model, optimizer, loss_fn=loss_fn, device='cpu')
collator = Collator(tok.token_to_id('<|pad|>'))
for name, path in [
 ('pretraining_educational', 'data/cleaned/pretraining-pilot/fineweb_edu/train.packed.jsonl'),
 ('pretraining_code', 'data/cleaned/pretraining-pilot/code_pretraining/train.packed.jsonl'),
 ('sft_math', 'data/cleaned/instruction-pilot/math/train.jsonl'),
 ('sft_code', 'data/cleaned/instruction-pilot/coding/train.jsonl'),
]:
    dataset = LazyJSONLDataset(path, tok, max_length=512)
    batch = collator([dataset[0]])
    assert int(batch['loss_mask'][:, 1:].sum()) > 0
    loss = trainer.train_step(batch)
    row = {'stage': name, 'loss': loss, 'gradient_norm': trainer.last_gradient_norm,
           'optimizer_step': trainer.global_step, 'target_tokens': int(batch['loss_mask'][:, 1:].sum())}
    report['steps'].append(row)
    print(json.dumps(row), flush=True)
assert trainer.global_step == 4 and trainer.nonfinite_updates == 0
report['checks'].update(four_optimizer_steps=True, nonfinite_updates=trainer.nonfinite_updates)
report['smoke_checkpoint'] = 'checkpoints/improvement-smoke/latest.pt'
save_checkpoint(report['smoke_checkpoint'], model, step=trainer.global_step,
                metadata={**metadata, 'smoke_only': True})
Path('reports/minigpt_improvements.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report['checks']), flush=True)
