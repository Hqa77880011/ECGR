"""Render EL geometry with separate dense-sheet circuit I-V targets."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def sheet_resistance(damage, grain):
    """Solve a full four-neighbor sheet between opposite Dirichlet contacts."""
    size = len(damage)
    edges, weights = [], []
    for y in range(size):
        for x in range(size):
            for dy, dx in ((0, 1), (1, 0)):
                yy, xx = y + dy, x + dx
                if yy >= size or xx >= size:
                    continue
                edges.append((y * size + x, yy * size + xx))
                attenuation = np.exp(-5 * max(damage[y, x], damage[yy, xx]))
                weights.append(np.sqrt(grain[y, x] * grain[yy, xx]) * attenuation)
    edges, weights = np.asarray(edges), np.asarray(weights)
    i, j = edges.T
    laplacian = coo_matrix((np.r_[weights, weights, -weights, -weights],
                           (np.r_[i, j, i, j], np.r_[i, j, j, i])),
                          shape=(size * size, size * size)).tocsr()
    voltage = np.zeros(size * size)
    voltage[np.arange(size) * size] = 1
    free = np.flatnonzero((np.arange(size * size) % size != 0) &
                          (np.arange(size * size) % size != size - 1))
    voltage[free] = spsolve(laplacian[free][:, free], -(laplacian @ voltage)[free])
    current = (laplacian @ voltage)[np.arange(size) * size].sum()
    return float(1 / current)


def module(rng, size=48):
    yy, xx = np.mgrid[:size, :size]
    busbar = ((xx >= 3) & (xx <= 4) & (yy >= 3) & (yy < size - 3))
    grid = (yy % 8 == 4) & (xx >= 3) & (xx < size - 3)
    regions = np.zeros((size, size), np.int64)
    inside = (xx > 5) & (xx < size - 3) & (yy > 3) & (yy < size - 3)
    regions[inside] = 1 + (xx[inside] >= size // 2) + 2 * (yy[inside] >= size // 2)
    orientation = rng.choice([0.0, np.pi / 4, np.pi / 2])
    center = rng.uniform(size * 0.3, size * 0.7, 2)
    distance = np.abs((xx - center[0]) * np.cos(orientation) + (yy - center[1]) * np.sin(orientation))
    crack = (distance < rng.uniform(0.65, 1.1)) & (yy > 3) & (yy < size - 3) & (xx > 5)
    severity = rng.uniform(0.3, 1.0)
    texture = gaussian_filter(rng.normal(size=(size, size)), 2)
    image = 0.55 + 0.08 * texture + 0.15 * grid + 0.3 * busbar - (0.3 + 0.2 * severity) * crack
    spot = (xx - rng.uniform(8, size - 8)) ** 2 + (yy - rng.uniform(8, size - 8)) ** 2 < 9
    image -= 0.12 * spot  # Non-crack dark-area confound.
    image += rng.normal(0, 0.025, image.shape)
    # This circuit uses all sheet nodes, hidden grain disorder and a different law;
    # it never calls the visible-network risk estimator under evaluation.
    damage = zoom(gaussian_filter(crack.astype(float), 0.7), 12 / size, order=1) * severity
    grain = rng.lognormal(0, 0.2, (12, 12))
    nominal = sheet_resistance(np.zeros_like(damage), grain)
    damaged = sheet_resistance(damage, grain)
    voltage = np.linspace(0, 1, 33)
    nominal_current, damaged_current = (1 - voltage) / nominal, (1 - voltage) / damaged
    p0 = np.max(voltage * nominal_current)
    pd = np.max(voltage * damaged_current)
    return dict(image=np.clip(image, 0, 1).astype(np.float32), regions=regions,
                crack=crack.astype(np.float32), grid=grid.astype(np.float32),
                busbar=busbar.astype(np.float32), delta_pmpp=np.float32(1 - pd / p0),
                delta_rs=np.float32(damaged - nominal), iv_voltage=voltage,
                iv_nominal=nominal_current, iv_damaged=damaged_current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int, default=36)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.groups < 10:
        parser.error("Use at least ten independent modules")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(str(output))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for group in range(args.groups):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, group]))
        split = "train" if group < 0.6 * args.groups else "validation" if group < 0.8 * args.groups else "test"
        name = f"module_{group:03d}.npz"
        np.savez_compressed(output / name, **module(rng))
        records.append(dict(id=str(group), path=name, module=f"module-{group}", split=split,
                            subset="synthetic_dense_sheet", endpoint_source="generated circuit I-V"))
    (output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "source.json").write_text(json.dumps(dict(
        source="rendered EL and independently solved full-sheet resistor circuit",
        truth="Thevenin I-V/power curves, not measured PV endpoints or ECGR scores",
        seed=args.seed, groups=args.groups,
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
