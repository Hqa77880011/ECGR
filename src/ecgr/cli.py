import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr
from skimage.morphology import skeletonize
from .common import load, save, table, plot
from .graph import graph, risk, explain, proxies, conductance


class Structure(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 3, 1),
        )

    def forward(self, x):
        return self.net(x)


def prepare(a, c):
    records = load(a.input)
    modules = {}
    result = []
    for record in records:
        if record["module"] in modules and modules[record["module"]] != record["split"]:
            raise ValueError("Physical module crosses splits")
        modules[record["module"]] = record["split"]
        path = Path(record["path"])
        path = path if path.is_absolute() else Path(a.input).parent / path
        raw = np.load(path)
        if "regions" not in raw or "image" not in raw:
            raise ValueError("image and generating regions are required")
        result.append(dict(record, path=str(path.resolve())))
    save(Path(a.data) / "manifest.json", result)


def train(a, c):
    torch.manual_seed(c["seed"])
    records = [r for r in load(Path(a.data) / "manifest.json") if r["split"] == "train"]
    model = Structure(c["width"]).to(c["device"])
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    rows = []
    if not records:
        raise ValueError("No training modules")
    for epoch in range(c["epochs"]):
        total = 0.0
        for record in records:
            raw = np.load(record["path"])
            x = torch.tensor(raw["image"], device=c["device"], dtype=torch.float32)[
                None, None
            ]
            target = torch.tensor(
                np.stack([raw["crack"], raw["grid"], raw["busbar"]]),
                device=c["device"],
                dtype=torch.float32,
            )[None]
            logits = model(x)
            prob = logits.sigmoid()
            dice = 1 - (2 * (prob * target).sum((2, 3)) + 1) / (
                prob.sum((2, 3)) + target.sum((2, 3)) + 1
            )
            loss = (
                nn.functional.binary_cross_entropy_with_logits(logits, target)
                + dice.mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        rows.append(dict(epoch=epoch + 1, loss=total / len(records)))
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    table(output / "history.csv", rows)
    torch.save(dict(model=model.cpu().state_dict(), config=c), output / "structure.pt")


def segmentation(prediction, truth):
    from scipy.ndimage import label
    from scipy.spatial import cKDTree

    p = prediction >= 0.5
    y = truth >= 0.5
    intersection = np.sum(p & y)
    skeleton_p = skeletonize(p)
    skeleton_y = skeletonize(y)
    precision = np.sum(skeleton_p & y) / max(skeleton_p.sum(), 1)
    recall = np.sum(skeleton_y & p) / max(skeleton_y.sum(), 1)
    foreground_iou = intersection / max(np.sum(p | y), 1)
    background_iou = np.sum(~p & ~y) / max(np.sum(~p | ~y), 1)
    anchors = np.argwhere(skeleton_y)
    connection_f1 = None
    if len(anchors) >= 2:
        anchors = anchors[
            np.linspace(0, len(anchors) - 1, min(64, len(anchors))).astype(int)
        ]
        predicted_labels, _ = label(p, structure=np.ones((3, 3)))
        true_labels, _ = label(y, structure=np.ones((3, 3)))
        truth_ids = true_labels[anchors[:, 0], anchors[:, 1]]
        foreground = np.argwhere(p)
        predicted_ids = np.zeros(len(anchors), int)
        if len(foreground):
            distances, indices = cKDTree(foreground).query(anchors)
            matched = foreground[indices]
            predicted_ids = predicted_labels[matched[:, 0], matched[:, 1]]
            predicted_ids[distances > 2] = 0
        i, j = np.triu_indices(len(anchors), 1)
        positive = (truth_ids[i] == truth_ids[j]) & (truth_ids[i] > 0)
        predicted = (predicted_ids[i] == predicted_ids[j]) & (predicted_ids[i] > 0)
        tp = np.sum(positive & predicted)
        connection_f1 = float(2 * tp / max(positive.sum() + predicted.sum(), 1))
    return dict(
        dice=2 * intersection / max(p.sum() + y.sum(), 1),
        iou=foreground_iou,
        miou=(foreground_iou + background_iou) / 2,
        precision=intersection / max(p.sum(), 1),
        recall=intersection / max(y.sum(), 1),
        cldice=2 * precision * recall / max(precision + recall, 1e-12),
        connection_f1=connection_f1,
    )


def evaluate(a, c):
    model = None
    if a.checkpoint:
        state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        model = Structure(state["config"]["width"])
        model.load_state_dict(state["model"])
        model.eval()
    rng = np.random.default_rng(c["seed"])
    rows = []
    extraction = []
    explanations = []
    for record in load(Path(a.data) / "manifest.json"):
        if record["split"] != "test":
            continue
        raw = np.load(record["path"])
        regions = raw["regions"]
        if a.reviewed:
            masks = np.stack([raw["crack"], raw["grid"], raw["busbar"]])
        elif model:
            with torch.no_grad():
                masks = (
                    model(torch.tensor(raw["image"], dtype=torch.float32)[None, None])
                    .sigmoid()[0]
                    .numpy()
                )
        else:
            masks = np.stack([raw["pred_crack"], raw["pred_grid"], raw["pred_busbar"]])
        for i, key in enumerate(["crack", "grid", "busbar"]):
            if key in raw:
                extraction.append(
                    dict(
                        id=record["id"],
                        structure=key,
                        **segmentation(masks[i], raw[key]),
                    )
                )
        crack, grid, busbar = masks
        if c["occlusion"] > 0:
            grid = grid.copy()
            missing = rng.random(grid.shape) < c["occlusion"]
            grid[missing] = 0
        g = graph(crack, grid, busbar, regions, c)
        score, observable = risk(g)
        binary = g["damaged"].copy()
        binary[
            (g["nominal"] - g["damaged"]) / np.maximum(g["nominal"], 1e-12) > 0.5
        ] = 0
        nominal = conductance(g, g["nominal"])
        actual = conductance(g, binary)
        valid = nominal > 0
        binary_risk = 1 - np.sum(g["areas"][valid] * (actual[valid] > 0)) / np.sum(
            g["areas"][valid]
        )
        row = dict(
            id=record["id"],
            module=record["module"],
            subset=record.get("subset", "all"),
            ecgr=score,
            binary_reachability=float(binary_risk),
            observable_fraction=observable,
            **proxies(crack, grid, regions, c),
        )
        for key in ["delta_pmpp", "delta_rs"]:
            row[key] = float(raw[key]) if key in raw else None
        rows.append(row)
        explanations.append(dict(id=record["id"], edges=explain(g, c)))
    summaries = []
    for endpoint in ["delta_pmpp", "delta_rs"]:
        for subset in sorted({"all"} | {r["subset"] for r in rows}):
            available = [
                r
                for r in rows
                if r[endpoint] is not None
                and (subset == "all" or r["subset"] == subset)
            ]
            for method in [
                "crack_area",
                "crack_length",
                "orientation_proxy",
                "inactive_area_proxy",
                "binary_reachability",
                "ecgr",
            ]:
                rho = (
                    float(
                        spearmanr(
                            [r[method] for r in available],
                            [r[endpoint] for r in available],
                        ).statistic
                    )
                    if len(available) >= 3
                    else None
                )
                summaries.append(
                    dict(
                        method=method,
                        endpoint=endpoint,
                        subset=subset,
                        spearman=rho,
                        records=len(available),
                        reviewed_masks=a.reviewed,
                        occlusion=c["occlusion"],
                    )
                )
    if not any(r["records"] for r in summaries):
        raise ValueError("No independent electrical endpoints available")
    table(Path(a.output) / "samples.csv", rows)
    table(Path(a.output) / "structure.csv", extraction)
    save(Path(a.output) / "explanations.json", explanations)
    save(Path(a.output) / "metrics.json", summaries)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command", choices=["prepare", "train", "evaluate", "compare", "plot"]
    )
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--input")
    p.add_argument("--data", default="data/el")
    p.add_argument("--output", default="runs/full")
    p.add_argument("--checkpoint")
    p.add_argument("--reviewed", action="store_true")
    p.add_argument("--inputs", nargs="*")
    a = p.parse_args()
    c = load(a.config)
    if a.command == "prepare":
        prepare(a, c)
    elif a.command == "train":
        train(a, c)
    elif a.command == "evaluate":
        evaluate(a, c)
    elif a.command == "compare":
        rows = [r for d in a.inputs for r in load(Path(d) / "metrics.json")]
        save(Path(a.output) / "comparison.json", rows)
        table(Path(a.output) / "comparison.csv", rows)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")
