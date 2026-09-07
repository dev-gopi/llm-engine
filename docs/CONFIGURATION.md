# Configuration defaults and overrides

YAML files loaded through `utils.config.load_yaml` can share defaults:

```yaml
extends: defaults/training-runtime.yaml
runtime:
  tokenizer: data/tokenizer-finetuning
  output: checkpoints/finetuning/latest.pt
  best_output: checkpoints/finetuning/best.pt
```

`extends` accepts one filename or a list. Parent paths are relative to the YAML
file containing them. Parents merge left to right, then the child overrides them.
Nested mappings merge; lists and scalar values replace. Cycles and invalid parent
specifications raise errors. Ordinary dataset/checkpoint paths remain relative
to the process working directory, as before.

The training CLI resolves the following options from `runtime` when their CLI
flags are omitted: `tokenizer`, `output`, `best_output`, `log_file`, `report_json`,
`report_refresh_seconds`, `report_telemetry_seconds`, and
`report_telemetry_points`. The fine-tuning GPU profile inherits shared reporting
defaults and specifies its own tokenizer and output paths.

Chat and generation resolve `model_config`, `tokenizer_path`, `checkpoint_path`,
and `device` from the `serving` section of the selected inference YAML. Their CLI
flags override those values. Generation no longer scans checkpoint directories;
a missing configured checkpoint fails instead of silently selecting other weights.

Precedence is explicit CLI argument, then YAML, then a code fallback default.
Zero CLI values remain explicit and reach the relevant validation. Architecture
and training hyperparameters continue to use their existing configuration keys.
This does not make algorithmic constants into settings or remove every default
from the codebase. Other task-specific scripts retain their existing options.

Configuration is read at process startup, not hot-reloaded. Restart the command
after editing settings. Existing explicit training commands remain supported.
