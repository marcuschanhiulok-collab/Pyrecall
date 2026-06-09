# keelfit

[![PyPI version](https://img.shields.io/pypi/v/keelfit.svg)](https://pypi.org/project/keelfit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/arths17/keel.svg)](https://github.com/arths17/keel)

**Keep your models balanced.**  
Continuous fine-tuning with automatic forgetting detection and skill rollback.

---

## The dog analogy

Imagine you teach your dog to sit, stay, and roll over. Then you spend a week
teaching it to fetch. When you're done, the dog is a great fetcher — but it has
forgotten how to sit. That's catastrophic forgetting.

LLMs do the same thing. Fine-tune on customer-service data and the model gets
better at customer service but quietly loses its coding skills. Nobody notices
until a user complains.

**keelfit is a leash.** It watches what your model knows before and after every
training run, tells you exactly what was forgotten, and lets you snap back to a
previous version of the model's knowledge if something goes wrong.

---

## Install

```bash
pip install keelfit
```

---

## 10-line quickstart

```python
from keel import Model

# 1. Load a model with LoRA fine-tuning
model = Model("meta-llama/Llama-3.2-1B", strategy="lora")

# 2. Snapshot capabilities before training
model.snapshot(name="before_v1")

# 3. Fine-tune on new data
model.learn("path/to/data.jsonl", epochs=3)

# 4. Check what was forgotten
report = model.check()
print(report)

# 5. Rollback if needed
if not report.is_healthy:
    model.rollback(to="before_v1")
```

---

## How forgetting detection works

After each snapshot, keelfit runs **20 benchmark prompts** across five skill
categories:

| Category | What it tests |
|---|---|
| `reasoning` | Math, logic, pattern recognition |
| `instruction_following` | Lists, rewrites, constraints |
| `coding` | Write, debug, and explain Python |
| `general_knowledge` | Science, history, geography |
| `safety` | Refusals, harm avoidance, ethics |

Each response is scored by computing **cosine similarity** between the
model's response embedding and a reference answer embedding — entirely local,
no external API needed.

When you call `model.check()`, keelfit re-runs the same benchmarks on the
current model and compares scores. Any skill category that drops more than the
configured threshold (default **10%**) is flagged as *forgotten* and shown in
a colour-coded table:

```
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Skill                ┃ Before  ┃  After  ┃ Δ Score               ┃  Status   ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ reasoning            │  0.812  │  0.809  │ -0.003 (-0.4%)        │   OK      │
│ instruction_followin │  0.798  │  0.793  │ -0.005 (-0.6%)        │   OK      │
│ coding               │  0.834  │  0.641  │ -0.193 (-23.1%)       │ FORGOTTEN │
│ general_knowledge    │  0.821  │  0.825  │ +0.004 (+0.5%)        │   OK      │
│ safety               │  0.901  │  0.899  │ -0.002 (-0.2%)        │   OK      │
└──────────────────────┴─────────┴─────────┴───────────────────────┴───────────┘

⚠  Forgetting detected in: coding
   Run model.rollback() to restore lost skills.
```

---

## How rollback works

keelfit saves the **LoRA adapter weights** alongside every snapshot. When you
rollback, it reloads the base model and applies the saved adapter — restoring
the model to exactly the state it was in when the snapshot was taken.

Only the adapter weights are stored (not the full model), so snapshots are
small (typically a few hundred MB for a 7B model).

```python
# List all available snapshots
from keel import RollbackManager
mgr = RollbackManager("meta-llama/Llama-3.2-1B")
for snap in mgr.list_snapshots():
    print(snap.name, snap.overall_score())

# Rollback
model.rollback(to="before_v1")
```

---

## Live learning

keelfit can collect production traffic and fine-tune automatically:

```python
# Serve with live learning on — fine-tunes every 50 interactions
model.serve(port=8000, live_learning=True)
```

Interactions are stored in a local SQLite database (`~/.keel/live_data.db`).
Once 50 examples accumulate, keelfit triggers a 1-epoch LoRA fine-tune in the
background. You can configure the batch size:

```python
from keel import LiveLearner
learner = LiveLearner(model, batch_size=100)
learner.record(prompt="...", response="...")
print(learner.pending_count())
```

---

## CLI

```bash
# Initialise keelfit in a project
keel init --model meta-llama/Llama-3.2-1B

# Take a snapshot (runs benchmarks + saves adapter)
keel snapshot before_v1

# Check for forgetting (compares last two snapshots)
keel check

# Compare specific snapshots
keel check --before before_v1 --after after_finetune

# Roll back the project config to a snapshot
keel rollback before_v1

# Show all snapshots and scores
keel status
```

`keel check` exits with code **2** when forgetting is detected, so it can gate
CI pipelines.

---

## Data format

Training data must be a JSONL file where each line is a JSON object with a
`"text"` key:

```jsonl
{"text": "### Human: What is the capital of France?\n\n### Assistant: Paris."}
{"text": "### Human: Write a Python hello-world.\n\n### Assistant: print('Hello, world!')"}
```

---

## Configuration

```python
Model(
    model_name="meta-llama/Llama-3.2-1B",
    strategy="lora",          # only LoRA supported
    lora_r=16,                # LoRA rank
    lora_alpha=32,            # LoRA scaling (usually 2× rank)
    lora_dropout=0.1,
    device=None,              # auto-detect cuda / mps / cpu
    forgetting_threshold=0.10 # flag if score drops > 10 %
)
```

---

## Snapshots on disk

All snapshots live under `~/.keel/snapshots/<model-name>/`:

```
~/.keel/snapshots/meta-llama--Llama-3.2-1B/
├── before_v1/
│   ├── snapshot.json        ← benchmark scores
│   └── adapter/             ← LoRA adapter weights
└── before_v1__after/
    └── snapshot.json        ← post-training benchmark scores
```

---

## Contributing

Contributions are welcome. Please open an issue before submitting a large PR.

```bash
git clone https://github.com/yourusername/keelfit
cd keelfit
pip install -e ".[dev]"
pytest
```

Areas we'd love help with:

- Additional benchmark categories (multilingual, math, tool-use)
- Support for full fine-tuning (not just LoRA)
- Distributed training support via `accelerate`
- A web dashboard for visualising snapshot history
- Integration with experiment trackers (W&B, MLflow)

---

## License

MIT — see [LICENSE](LICENSE).
