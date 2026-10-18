# ECGR

Implementation of electrical-connectivity-constrained PV crack risk from electroluminescence signals.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data generation and preparation

The generator renders EL textures and crack/collection geometry for independent modules. A separate full-sheet resistor circuit produces the synthetic I–V endpoints; these are generated electrical quantities. `source.json` records the process and seed.

```bash
python -m ecgr.synthetic --output data/generated --groups 36 --seed 17
ecgr prepare --config configs/reproduce.json --input data/generated/manifest.json --data data/reproduce
```

For external data: Use [ELPV](https://github.com/zae-bayern/elpv-dataset) for EL classification/structure context and paired EL–I–V measurements for electrical validation, such as [Sandia PV-IV-EL](https://data.openei.org/submissions/8378). ELPV class labels alone are not independent electrical degradation targets. Export NPZ files with normalized grayscale `image`, integer generating-cell `regions` (0 background), structure masks `crack,grid,busbar` when annotated, and `delta_pmpp`/`delta_rs` when independently measured. Use consistently oriented endpoints: positive values denote degradation. Automatic probability maps can be supplied as `pred_crack,pred_grid,pred_busbar`.

## Training and inference

`configs/reproduce.json` fixes seed `17`. Use the same config and prepared inputs throughout; change `seed` there for another run and keep its output directories separate.

```bash
ecgr train --config configs/reproduce.json --data data/reproduce --output runs/reproduce_structure
```

## Evaluation

```bash
ecgr evaluate --config configs/reproduce.json --data data/reproduce --checkpoint runs/reproduce_structure/structure.pt --output runs/reproduce
ecgr evaluate --config configs/reproduce.json --data data/reproduce --reviewed --output runs/reproduce_reviewed
```

## Comparisons and ablations

Evaluate automatic and reviewed structures separately, then pass their evaluated run directories to `compare --inputs`.

```bash
ecgr compare --inputs runs/reproduce runs/reproduce_reviewed --output outputs/comparison
ecgr plot --output outputs/comparison
```

## Outputs

The selected commands write their generated inputs under `data/generated/`, prepared data under `data/reproduce/`, and checkpoints/results in the explicit output directories. Corresponding steps produce `history.csv`, `structure.pt`, `samples.csv`, `structure.csv`, `explanations.json`, `metrics.json`, `source.json`. Comparisons write CSV/JSON and plots under their output directory.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
