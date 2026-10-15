# ECGR

Implementation of electrical-connectivity-constrained PV crack risk from electroluminescence signals.

## Installation


Use Python 3.11 or newer. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
``` For GPU execution, install matching PyTorch and torchvision builds using the [PyTorch selector](https://pytorch.org/get-started/locally/).


## EL and independent electrical records

Use [ELPV](https://github.com/zae-bayern/elpv-dataset) for EL classification/structure context and paired EL–I–V measurements for electrical validation, such as [Sandia PV-IV-EL](https://data.openei.org/submissions/8378). ELPV class labels alone are not independent electrical degradation targets. Export NPZ files with normalized grayscale `image`, integer generating-cell `regions` (0 background), structure masks `crack,grid,busbar` when annotated, and `delta_pmpp`/`delta_rs` when independently measured. Use consistently oriented endpoints: positive values denote degradation. Automatic probability maps can be supplied as `pred_crack,pred_grid,pred_busbar`.

A JSON manifest lists `{id,path,module,split,subset}`. Keep repeated inspections, cells and crops from one physical module in one split. `subset` can encode crack orientation or non-crack confounds and is used only for reporting.

```bash
ecgr prepare --input data/manifest.json --data data/el
ecgr train --data data/el --output runs/structure
ecgr evaluate --data data/el --checkpoint runs/structure/structure.pt --output runs/full
ecgr evaluate --data data/el --reviewed --output runs/reviewed
ecgr compare --inputs runs/full runs/reviewed --output outputs/comparison
ecgr plot --output outputs/comparison
```

Omit the structure checkpoint to use precomputed automatic maps. `train` is a compact three-output convolutional structure estimator with BCE+Dice, interchangeable with a stronger segmentation model. Reviewed-mask evaluation keeps the same electrical functional and endpoints.

## Graph risk and explanations

Skeletonized visible fingers and busbars form an eight-neighbor conductance graph. Nominal edge conductance depends on type and pixel length; crack probability attenuates it by `exp(-kappa*mean_probability)`. Each generating region is represented at its centroid's nearest network node, weighted by region area. Grounded sparse Laplacian solves yield effective conductance to visible busbars. Risk compares actual/crack-free conductance on the **same** visible graph and clips each collection ratio at one. Regions with no crack-free terminal path are excluded and their area fraction is reported as lost observability. Busbar source nodes have zero collection loss.

Edge criticality removes a crack-attenuated edge and recomputes the same risk. Explanations contain pixel endpoints and risk increments. `maximum_explanations` limits the candidate edges to the largest nominal conductance losses. Baselines hold probability masks fixed: crack area/length, a global orientation proxy, per-region crack-convex-hull inactive-area proxy and thresholded binary reachability. Geometry proxies are explicit controls, not independent electrical ground truth.

## Metrics and outputs

`configs/default.json` controls segmentation training, thresholds, kappa, busbar conductance, explanation count, seed, device and grid occlusion. Occlusion independently removes visible finger pixels before graph construction while retaining busbars; use separate configs/output paths for the stress curve.

`structure.csv` contains per-structure Dice/IoU/mIoU/precision/recall/clDice/Conn-F1. `samples.csv` stores every functional and independent endpoint; `explanations.json` stores critical edges. `metrics.json` reports Spearman correlation with delta-Pmpp and delta-Rs overall and by subset, plus the contributing count. Undefined correlations are null. Records, rather than graph-derived proxies, define electrical outcomes. Aggregate repeated cell/crop observations to module endpoints before export to avoid treating correlated crops as independent electrical samples. Comparison CSV/JSON and PNG/SVG plots use computed measurements.

## Multiple seeds

Run matched configurations with different seeds in separate directories, then aggregate the generated metric files:

```bash
ecgr-report --inputs runs/seed0/metrics.json runs/seed1/metrics.json --output outputs/seeds
```

Use actual metric paths from your runs, including method subdirectories where applicable. Keep dataset, split and experimental condition fixed within a report. Summary CSV/JSON gives per-method arithmetic means and sample standard deviations; a single run has no standard-deviation estimate.

Conn-F1 compares connectedness of up to 64 deterministic ground-truth skeleton anchors across all pairs. Predicted anchors are matched within two pixels; missed anchors count as disconnected. mIoU averages foreground and background IoU. These definitions fix the otherwise unspecified topology-metric sampling convention.
