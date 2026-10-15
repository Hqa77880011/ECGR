import numpy as np
from scipy.spatial import cKDTree, ConvexHull, QhullError
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from skimage.morphology import skeletonize


def graph(crack, grid, busbar, regions, c):
    visible = skeletonize(
        (grid >= c["grid_threshold"]) | (busbar >= c["grid_threshold"])
    )
    pixels = np.argwhere(visible)
    if not len(pixels):
        raise ValueError("No visible collection network")
    lookup = {tuple(p): i for i, p in enumerate(pixels)}
    edges = []
    nominal = []
    damaged = []
    for i, (y, x) in enumerate(pixels):
        for dy, dx in [(0, 1), (1, -1), (1, 0), (1, 1)]:
            j = lookup.get((y + dy, x + dx))
            if j is None:
                continue
            length = np.hypot(dy, dx)
            kind = (
                c["busbar_conductance"]
                if (busbar[y, x] + busbar[y + dy, x + dx]) / 2 >= c["grid_threshold"]
                else 1.0
            )
            g0 = kind / length
            pc = (crack[y, x] + crack[y + dy, x + dx]) / 2
            edges.append((i, j))
            nominal.append(g0)
            damaged.append(g0 * np.exp(-c["kappa"] * pc))
    terminals = np.array([busbar[y, x] >= c["grid_threshold"] for y, x in pixels])
    if not terminals.any():
        raise ValueError("No visible busbar terminal")
    sources = []
    areas = []
    identities = []
    tree = cKDTree(pixels)
    for identity in np.unique(regions):
        if identity == 0:
            continue
        cells = np.argwhere(regions == identity)
        if not len(cells):
            continue
        sources.append(int(tree.query(cells.mean(0))[1]))
        areas.append(len(cells))
        identities.append(int(identity))
    if not sources:
        raise ValueError("No generating cell regions")
    return dict(
        pixels=pixels,
        edges=np.asarray(edges, dtype=int),
        nominal=np.asarray(nominal),
        damaged=np.asarray(damaged),
        terminals=terminals,
        sources=np.asarray(sources),
        areas=np.asarray(areas),
        region_ids=identities,
    )


def conductance(g, weights):
    n = len(g["pixels"])
    edges = g["edges"]
    valid = weights > 1e-15
    if not valid.any():
        return np.zeros(len(g["sources"]))
    ii, jj = edges[valid].T
    ww = weights[valid]
    adjacency = coo_matrix(
        (np.r_[ww, ww], (np.r_[ii, jj], np.r_[jj, ii])), shape=(n, n)
    ).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    grounded_components = set(labels[g["terminals"]])
    reachable = np.array([l in grounded_components for l in labels])
    interior = np.flatnonzero(reachable & ~g["terminals"])
    result = np.zeros(len(g["sources"]))
    result[g["terminals"][g["sources"]]] = np.inf
    if not len(interior):
        return result
    laplacian = diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency
    factor = splu(laplacian[interior][:, interior].tocsc())
    location = {node: i for i, node in enumerate(interior)}
    unique = {s for s in g["sources"] if s in location}
    cache = {}
    for node in unique:
        impulse = np.zeros(len(interior))
        impulse[location[node]] = 1
        resistance = factor.solve(impulse)[location[node]]
        cache[node] = 1 / max(resistance, 1e-15)
    for i, node in enumerate(g["sources"]):
        if node in cache:
            result[i] = cache[node]
    return result


def risk(g, weights=None):
    nominal = conductance(g, g["nominal"])
    actual = conductance(g, g["damaged"] if weights is None else weights)
    valid = nominal > 0
    if not valid.any():
        raise ValueError("No generating region is observable from a visible busbar")
    ratio = np.zeros_like(actual)
    finite = valid & np.isfinite(nominal)
    ratio[finite] = np.minimum(actual[finite] / nominal[finite], 1)
    ratio[np.isinf(nominal)] = 1
    score = 1 - float(
        np.sum(g["areas"][valid] * ratio[valid]) / np.sum(g["areas"][valid])
    )
    return score, float(np.sum(g["areas"][valid]) / np.sum(g["areas"]))


def explain(g, c):
    base, _ = risk(g)
    ranking = np.argsort(g["nominal"] - g["damaged"])[::-1][: c["maximum_explanations"]]
    rows = []
    for index in ranking:
        if g["nominal"][index] - g["damaged"][index] <= 1e-12:
            continue
        weights = g["damaged"].copy()
        weights[index] = 0
        changed, _ = risk(g, weights)
        rows.append(
            dict(
                edge=int(index),
                pixels=g["pixels"][g["edges"][index]].tolist(),
                criticality=changed - base,
            )
        )
    return sorted(rows, key=lambda r: -r["criticality"])


def proxies(crack, grid, regions, c):
    binary = crack >= c["crack_threshold"]
    skeleton = skeletonize(binary)
    area = float(np.mean(binary[regions > 0]))
    length = float(skeleton.sum() / max((regions > 0).sum(), 1))

    def direction(mask):
        points = np.argwhere(mask)
        if len(points) < 2:
            return np.array([1.0, 0.0])
        return np.linalg.svd(points - points.mean(0), full_matrices=False)[2][0]

    orientation = area * (
        1 - abs(float(direction(skeleton) @ direction(grid >= c["grid_threshold"])))
    )
    hull_area = 0
    for identity in np.unique(regions):
        if not identity:
            continue
        points = np.argwhere(binary & (regions == identity))
        if len(points) < 3:
            hull_area += len(points)
            continue
        try:
            hull_area += ConvexHull(points).volume
        except QhullError:
            hull_area += len(points)
    return dict(
        crack_area=area,
        crack_length=length,
        orientation_proxy=orientation,
        inactive_area_proxy=float(hull_area / max((regions > 0).sum(), 1)),
    )
