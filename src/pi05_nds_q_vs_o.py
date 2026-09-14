"""Normalized decoding scores for H→Q vs H→O on a finished occupancy run.

NDS_Q = 1 - MSE_{H→Q} / MSE_{base,Q}  (R^2 of train-standardized Q)
NDS_O = 1 - BCE_{H→O} / BCE_{base,O}  (unweighted Bernoulli NLL vs train voxel rates)
D_{O-Q} = NDS_O - NDS_Q

Baselines use train-split statistics only. Negative scores are kept as-is.
Script-23 occupancy checkpoints (Linear(D,64)-GELU-Linear(64,4096)) can be
reused for H→O evaluation at bottleneck 64; metrics.csv test_loss cannot,
because it is pos-weighted BCE + Dice rather than unweighted BCE.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.pi05_occupancy_cmi import parse_condition, zscore_train
from src.pi05_occupancy_full import probe_dir

OUTPUT_DIM_O = 4096
EPS = 1e-6


def nds(model_loss: float, baseline_loss: float) -> float:
    baseline_loss = float(baseline_loss)
    if not np.isfinite(baseline_loss) or abs(baseline_loss) < EPS:
        return float("nan")
    return 1.0 - float(model_loss) / baseline_loss


def d_o_minus_q(nds_o: float, nds_q: float) -> float:
    return float(nds_o) - float(nds_q)


def occupancy_frequency_baseline(occupancy: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    targets = occupancy.reshape(len(occupancy), -1).astype(np.float64)
    p = targets[train_idx].mean(axis=0)
    return np.clip(p, EPS, 1.0 - EPS).astype(np.float32)


def per_sample_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return np.mean((pred - target) ** 2, axis=1)


def per_sample_bernoulli_nll(target: np.ndarray, prob: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=np.float64)
    p = np.clip(np.asarray(prob, dtype=np.float64), EPS, 1.0 - EPS)
    return np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)), axis=1)


def per_sample_bce_with_logits(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    # stable softplus-style BCE: max(z,0) - z*y + log(1+exp(-|z|))
    nll = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return nll.mean(axis=1)


def shuffle_rows_by_split(
    features: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> np.ndarray:
    shuffled = np.array(features, copy=True)
    rng = np.random.default_rng(int(seed))
    shuffled[train_idx] = shuffled[train_idx][rng.permutation(len(train_idx))]
    shuffled[test_idx] = shuffled[test_idx][rng.permutation(len(test_idx))]
    return shuffled


def bottleneck_decoder(input_dim: int, hidden: int, output_dim: int, device):
    from torch import nn

    return nn.Sequential(
        nn.Linear(int(input_dim), int(hidden)),
        nn.GELU(),
        nn.Linear(int(hidden), int(output_dim)),
    ).to(device)


def occupancy_ckpt_is_reusable(checkpoint: dict[str, Any], bottleneck: int) -> bool:
    if int(checkpoint.get("output_dim", OUTPUT_DIM_O)) != OUTPUT_DIM_O:
        return False
    state = checkpoint.get("state_dict") or {}
    weight = state.get("0.weight")
    if weight is None:
        return False
    return int(weight.shape[0]) == int(bottleneck)


def load_occupancy_checkpoint(run_dir: Path, condition: str, bin_index: int, device):
    import torch

    path = probe_dir(run_dir, condition, bin_index) / "decoder.pt"
    if not path.exists():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    return checkpoint


def bootstrap_nds(
    model_losses: np.ndarray,
    baseline_losses: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    model_losses = np.asarray(model_losses, dtype=np.float64)
    baseline_losses = np.asarray(baseline_losses, dtype=np.float64)
    point = nds(float(model_losses.mean()), float(baseline_losses.mean()))
    n = len(model_losses)
    if n == 0 or int(n_bootstrap) <= 0:
        return {"point": point, "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    model_mean = model_losses[idx].mean(axis=1)
    base_mean = baseline_losses[idx].mean(axis=1)
    samples = np.where(np.abs(base_mean) < EPS, np.nan, 1.0 - model_mean / base_mean)
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return {"point": point, "ci_lo": float("nan"), "ci_hi": float("nan")}
    return {
        "point": point,
        "ci_lo": float(np.percentile(finite, 2.5)),
        "ci_hi": float(np.percentile(finite, 97.5)),
    }


def bootstrap_d(
    mse_h: np.ndarray,
    mse_base: np.ndarray,
    bce_h: np.ndarray,
    bce_base: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    nds_q = bootstrap_nds(mse_h, mse_base, n_bootstrap=n_bootstrap, seed=seed)
    nds_o = bootstrap_nds(bce_h, bce_base, n_bootstrap=n_bootstrap, seed=seed + 1)
    n = len(mse_h)
    point = d_o_minus_q(nds_o["point"], nds_q["point"])
    if n == 0 or int(n_bootstrap) <= 0:
        return {"point": point, "ci_lo": float("nan"), "ci_hi": float("nan")}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(n_bootstrap), n))
    q = 1.0 - mse_h[idx].mean(axis=1) / np.clip(mse_base[idx].mean(axis=1), EPS, None)
    o = 1.0 - bce_h[idx].mean(axis=1) / np.clip(bce_base[idx].mean(axis=1), EPS, None)
    samples = o - q
    return {
        "point": point,
        "ci_lo": float(np.percentile(samples, 2.5)),
        "ci_hi": float(np.percentile(samples, 97.5)),
        "nds_q_ci_lo": nds_q["ci_lo"],
        "nds_q_ci_hi": nds_q["ci_hi"],
        "nds_o_ci_lo": nds_o["ci_lo"],
        "nds_o_ci_hi": nds_o["ci_hi"],
    }


def make_q_baseline_per_sample(q_z: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    """Train-standardized Q has train mean 0, so the baseline predictor is 0."""
    zeros = np.zeros_like(q_z[test_idx])
    return per_sample_mse(zeros, q_z[test_idx])


def train_bottleneck_regressor(
    *,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device,
    kind: str,
    pos_weight=None,
) -> dict[str, Any]:
    """Train H→Q (MSE) or H→O (pos-weighted BCE+Dice). Returns per-sample test losses."""
    import torch
    from src.pi05_occupancy_cmi import occupancy_train_loss

    torch.manual_seed(int(seed))
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    decoder = bottleneck_decoder(x.shape[1], hidden, y.shape[1], device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
    last_train = float("nan")
    for _epoch in range(int(epochs)):
        decoder.train()
        losses = []
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start : start + batch_size]
            xb = torch.from_numpy(x[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            pred = decoder(xb)
            if kind == "q":
                loss = torch.nn.functional.mse_loss(pred, yb)
                loss_value = float(loss.detach().cpu())
            else:
                loss, loss_value = occupancy_train_loss(pred, yb, pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss_value)
        last_train = float(np.mean(losses))
    decoder.eval()
    with torch.inference_mode():
        test_pred = decoder(torch.from_numpy(x[test_idx]).to(device)).cpu().numpy()
    param_count = int(sum(p.numel() for p in decoder.parameters()))
    state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
    del decoder, opt
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()
    if kind == "q":
        per_sample = per_sample_mse(test_pred, y[test_idx])
    else:
        per_sample = per_sample_bce_with_logits(test_pred, y[test_idx])
    return {
        "train_loss": last_train,
        "test_loss": float(per_sample.mean()),
        "per_sample": per_sample.astype(np.float32),
        "param_count": param_count,
        "input_dim": int(x.shape[1]),
        "output_dim": int(y.shape[1]),
        "state_dict": state,
    }


def eval_occupancy_checkpoint(
    *,
    checkpoint: dict[str, Any],
    x_raw: np.ndarray,
    occupancy: np.ndarray,
    test_idx: np.ndarray,
    bottleneck: int,
    device,
) -> np.ndarray:
    import torch

    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    x = (x_raw.astype(np.float32) - mean) / std
    decoder = bottleneck_decoder(int(checkpoint["input_dim"]), bottleneck, OUTPUT_DIM_O, device)
    decoder.load_state_dict(checkpoint["state_dict"])
    decoder.eval()
    targets = occupancy.reshape(len(occupancy), -1).astype(np.float32)
    with torch.inference_mode():
        logits = decoder(torch.from_numpy(x[test_idx]).to(device)).cpu().numpy()
    del decoder
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()
    return per_sample_bce_with_logits(logits, targets[test_idx]).astype(np.float32)


def heatmap_groups(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], pd.DataFrame]:
    frame = pd.DataFrame(list(rows))
    groups = {}
    for (tower, flow), subset in frame.groupby(["tower", "flow"], dropna=False, sort=True):
        flow_key = "static" if pd.isna(flow) else str(flow)
        groups[(str(tower), flow_key)] = subset
    return groups


def plot_nds_heatmaps(rows: list[dict[str, Any]], output_dir: Path, dpi: int = 180) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    specs = (
        ("nds_q", "NDS_Q", "01_nds_q"),
        ("nds_o", "NDS_O", "02_nds_o"),
        ("d_o_minus_q", r"$D_{O-Q}$", "03_d_o_minus_q"),
    )
    for (tower, flow), subset in heatmap_groups(rows).items():
        layers = sorted(int(v) for v in subset["layer"].unique())
        bins = sorted(int(v) for v in subset["bin"].unique())
        for column, label, prefix in specs:
            matrix = np.full((len(layers), len(bins)), np.nan)
            for raw in subset.itertuples(index=False):
                matrix[layers.index(int(raw.layer)), bins.index(int(raw.bin))] = float(getattr(raw, column))
            fig, ax = plt.subplots(
                figsize=(max(7.5, 0.45 * len(bins) + 3.5), max(4.2, 0.28 * len(layers) + 2.2)),
                dpi=dpi,
            )
            if column == "d_o_minus_q":
                lim = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
                lim = lim if lim > 0 else 1.0
                image = ax.imshow(
                    matrix, aspect="auto", cmap="coolwarm", interpolation="nearest", vmin=-lim, vmax=lim
                )
            else:
                image = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
            ax.set_yticks(range(len(layers)), [str(v) for v in layers], fontsize=7)
            ax.set_xticks(range(len(bins)), [str(v) for v in bins], fontsize=7, rotation=90)
            ax.set_ylabel("layer")
            ax.set_xlabel("token-position bin")
            ax.set_title(f"{label}  {tower}  flow={flow}")
            fig.colorbar(image, ax=ax, label=label)
            fig.tight_layout()
            path = output_dir / f"{prefix}_{tower}_{flow.replace('.', 'p')}.png"
            fig.savefig(path)
            plt.close(fig)
            saved.append(path)
    return saved


def summarize_nds_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    real = frame[frame["arm"] == "real"] if "arm" in frame.columns else frame
    d = pd.to_numeric(real["d_o_minus_q"], errors="coerce").dropna()
    by_tower = []
    for tower, subset in real.groupby("tower", sort=True):
        values = pd.to_numeric(subset["d_o_minus_q"], errors="coerce").dropna()
        by_tower.append(
            {
                "tower": tower,
                "n": int(len(values)),
                "d_mean": float(values.mean()) if len(values) else None,
                "frac_o_dominant": float((values > 0).mean()) if len(values) else None,
                "frac_q_dominant": float((values < 0).mean()) if len(values) else None,
                "nds_q_mean": float(pd.to_numeric(subset["nds_q"], errors="coerce").mean()),
                "nds_o_mean": float(pd.to_numeric(subset["nds_o"], errors="coerce").mean()),
            }
        )
    return {
        "n_real": int(len(real)),
        "d_mean": float(d.mean()) if len(d) else None,
        "frac_o_dominant": float((d > 0).mean()) if len(d) else None,
        "frac_q_dominant": float((d < 0).mean()) if len(d) else None,
        "nds_q_mean": float(pd.to_numeric(real["nds_q"], errors="coerce").mean()) if len(real) else None,
        "nds_o_mean": float(pd.to_numeric(real["nds_o"], errors="coerce").mean()) if len(real) else None,
        "by_tower": by_tower,
        "note": (
            "D>0 is O-dominant vs task-specific baselines; D<0 is Q-dominant. "
            "Scores are not mutual information and are not clipped at zero."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_paired_scores_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
