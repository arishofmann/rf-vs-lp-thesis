# BA-Thesis: Linear Probing vs. Random Forests under Geographic Domain Shift

Code and configs for the bachelor thesis *"Linear Probing vs. Random Forests: Decoding Frozen Geospatial Foundation Model Embeddings under Domain Shift"* (Aris Hofmann, DHBW; supervised by Prof. Dr. Kai Holzweißig and Romeo Kienzler, IBM Research Zurich).

## Overview

Configs and scripts for evaluating LP, RF and kNN decoders on frozen TerraMind v1 embeddings. The experiments cover geographic domain shift between Finland and Portugal on BigEarthNet v2.0.

The experiments cover:
- **Pre-study (Exp 0):** All 19 classes, all four scenarios (FI→FI, PT→PT, FI→PT, PT→FI).
- **Dimension A:** Forest-class subset (4 balanced classes, all decoders).
- **Dimension B:** Sample efficiency curves (1%–100% of training data).
- **Dimension C:** RF optimization (Standard / Balanced / High-mtry / PCA).
- **Dimension D:** Per-class domain shift analysis.
- **Ablation E1:** Native multi-output RF (Binary Relevance confound test).
- **Ablation E3:** Linear Probing on PCA-reduced embeddings.
- **Ablation E5:** Random Projection at the same target dimensionality (PCA control).

## Reproduction

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install git+https://github.com/arishofmann/terratorch.git@c35a193
```

### 2. Generate embeddings

```bash
python scripts/generate_embeddings.py --config configs/<scenario>.yaml
```

Embeddings are extracted once from the frozen TerraMind v1 Base encoder and cached on disk.

### 3. Run experiments

```bash
python scripts/run_all_experiments_v2.py
```

This produces `results/results_experiment_{0,A,B,C,D,E1,E3}.{csv,json}` with five seeds per configuration.

### 4. Run ablations

```bash
# E5: Random Projection ablation
python scripts/run_random_projection_ablation.py
```

## Repository structure

```
scripts/    Experiment runners + train/eval pipeline
configs/    YAML configs per scenario and decoder
```

## TerraTorch fork

The experiments use a TerraTorch fork that adds a sklearn-in-Lightning pipeline for RF/LP decoders with checkpoint persistence: <https://github.com/arishofmann/terratorch> (commit `c35a193`).

## License

Apache License 2.0 — see `LICENSE`.
