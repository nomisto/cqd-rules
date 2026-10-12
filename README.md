# Training-free Score Calibration for Complex Query Decomposition

Code for *Training-free score calibration for complex query decomposition* (Ott, Chekol, Meilicke, Stuckenschmidt; ESWC 2025).

## Contents

| Folder | What it contains |
|---|---|
| `cqdr/` | CQD<sup>Rule</sup>: complex query decomposition with AnyBURL rules as the link predictor (applied with [PyClause](https://github.com/symbolic-kg/PyClause)) |
| `cqd_kgesn/` | CQD<sup>KGE+SN</sup>: CQD with ComplEx and standard score normalisation (sigmoid, min-max, softmax), based on [adaptive-cqd](https://github.com/EdinburghNLP/adaptive-cqd) |
| `cqdr+a/` | Experiment testing whether a learned adaptation function (as in CQD<sup>A</sup>) improves rule confidences |
| `models/` | Place for the pretrained ComplEx checkpoints (see `models/README.md`) |

Each folder has its own README with the exact commands.

## Data and rules

The `data/` and `rules/` folders are not part of this repository (about 5 GB unpacked). Download them from figshare:

https://figshare.com/s/4f1fbd5f5d2c4aca7c2e?file=51314348

Unpack the archive so that `data/` and `rules/` sit next to the code folders:

```
.
├── cqd_kgesn/
├── cqdr/
├── cqdr+a/
├── data/
│   └── betae/            # FB15k, FB15k-237 and NELL in BetaE and Query2Box format
├── models/
└── rules/
    ├── fb15krules-1000
    ├── fb15k237rules-1000
    └── nellrules-1000
```

The scripts use relative paths such as `../data/betae/...` and `../rules/...`, so run them from inside their own folder.
