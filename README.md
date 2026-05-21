# Landmark Classification

This repository holds the project to classify landmarks. The aim is to get a 60% or above accucary. For me the objective of this project was to apply previous learnings, but also explore more modern architectures, specifically I-JEPA. 

## How to run this project

This project requirements are documented in the `requirements.txt`. I leveraged `uv` to get them installed and create a `.venv`. 

### Requirements
- `uv` cli
- `python` 3.12 was used. Others my work, but this is what used for me.
- A GPU. While it would be possible to run with a CPU, a GPU greatly decreases the training time.

## Optional: Matryoshka Representation Learning

The from-scratch models (`scratch_cnn`, `scratch_resnet`) can be trained with a
[Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) head as
a drop-in alternative to the default classifier. The MRL head emits logits at
several nested prefix widths of the 512-d feature vector and is trained with
the sum of cross-entropy losses across those widths, so after training the
first `m` dimensions of the embedding are usable as a standalone classifier
for several `m` -- with no inference overhead.

Two variants are supported:

- `mrl-e` (default): one shared `Linear(512, num_classes)`, per-prefix logits
  come from slicing its weight matrix. Identical parameter count to the
  vanilla head.
- `mrl`: one independent `Linear(m, num_classes)` per granularity. More
  parameters, matches the original paper.

CLI:

```bash
# default schedule [8, 16, 32, 64, 128, 256, 512], shared (mrl-e) head
python -m src.experiment --models scratch_cnn --mrl-granularities auto

# explicit schedule + independent heads variant
python -m src.experiment --models scratch_cnn \
    --mrl-granularities 8,32,128,512 --mrl-mode mrl

# leave it off (default) -- behavior is unchanged
python -m src.experiment --models scratch_cnn
```

Run names get a `__mrl-{mode}` suffix so results files stay distinguishable.
The MRL config is persisted under a top-level `mrl` key in the run's config,
and per-granularity test accuracies are written under
`evaluation.per_granularity_accuracy` in the JSON.

MRL is currently wired only for the from-scratch backbones; if requested for a
transfer-learning model the runner prints a warning and falls back to the
vanilla head.