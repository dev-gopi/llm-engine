# Fine-tuned model response test

Checkpoint: `checkpoints/finetuning/best.pt`, step 16,000 (standard checkpoint
weights; no EMA shadow was present).

Verdict: Fine-tuning produced a large behavioral improvement over the recorded
pretraining probes. The model now recognizes the assistant/chat format, gives
short answers, attempts requested stories and Python, and generates text in
Bengali and Hindi. It is still not reliable enough to serve as a factual or
task-completing assistant: none of the seven correctness-sensitive probes fully
passed.

## Probe results

| Probe | Result | Evidence |
|---|---|---|
| Greeting/identity | Pass | `I'm Gopi. How can I help you today?` |
| Factual QA | Fail | Said `The capital of Spain is Paris.` for France |
| Arithmetic reasoning | Fail | Did not calculate or answer `4` |
| Exact instruction | Fail | Returned `A. A. A.` instead of `READY` |
| Story writing | Partial | Produced a related girl/dog narrative, but changed the girl to `Jack`, contradicted details, and did not complete the requested helping story |
| Python coding | Fail | Produced `def add(x)` while referencing undefined `a`, `b`, `find`, and `n` |
| Bengali QA | Fail | Generated Bengali script but did not answer Dhaka |
| Hindi QA | Fail | Generated Hindi script but did not answer New Delhi |

These are qualitative development probes using one seed, not a standardized
accuracy benchmark. They show behavior and failure modes but do not estimate
general benchmark accuracy.

## Held-out training metrics

At step 16,000, weighted validation loss improved from 3.397708 to 3.206663
(5.62%), and perplexity improved from 29.4391 to 24.3158 (17.40%). The strongest
relative perplexity improvements were Hindi (39.52%), Bengali (28.06%), chat
(14.87%), and GSM8K-style data (13.59%). Coding improved only 4.20%, while
English perplexity regressed 2.04%.

The loss improvements mean the model predicts held-out domain text better; they
do not establish answer correctness. The fixed prompts demonstrate that the
remaining correctness gap is substantial.

The training run was still active when this report was produced (59.2% complete
at the latest generated training report), so this is an interim assessment.
