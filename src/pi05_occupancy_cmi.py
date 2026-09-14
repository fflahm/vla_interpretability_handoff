"""Capacity-constrained I_C(H;O|Q) helpers for PI0.5 self-occupancy.

Q is proprioception. By default this is ``observation_state`` (EE + gripper)
already stored in ``samples.jsonl``. That vector does **not** uniquely determine
whole-body occupancy (voxel GT uses MuJoCo robot/gripper collision geoms). It is
used because it is already aligned to the occupancy run and is close to the
state PI0.5 actually consumes.

H_C is estimated by held-out unweighted BCE (Bernoulli NLL), not BCE+Dice.
Training still uses the script-23 pos-weighted BCE + Dice recipe.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

CONDITION_RE = re.compile(
    r"^(?P<tower>paligemma|expert)/layer_(?P<layer>\d+)(?:/(?:static|t=(?P<flow>[0-9.]+)))?$"
)

HIDDEN = 64
OUTPUT_DIM = 4096
DECODER_ARCHITECTURES = ("mlp", "linear")


def normalize_decoder_architecture(name: str) -> str:
    key = str(name).strip().lower()
    aliases = {
        "mlp": "mlp",
        "mlp64": "mlp",
        "hidden": "mlp",
        "linear": "linear",
        "lin": "linear",
        "logistic": "linear",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown decoder architecture {name!r}. "
            f"Supported: {', '.join(DECODER_ARCHITECTURES)}"
        )
    return aliases[key]


def decoder_description(architecture: str, hidden_size: int = HIDDEN) -> str:
    arch = normalize_decoder_architecture(architecture)
    if arch == "linear":
        return f"Linear(in,{OUTPUT_DIM})"
    return f"Linear(in,{int(hidden_size)})-GELU-Linear({int(hidden_size)},{OUTPUT_DIM})"


def parse_condition(condition: str) -> dict[str, Any]:
    match = CONDITION_RE.match(condition)
    if not match:
        raise ValueError(f"Unrecognized condition: {condition}")
    flow = match.group("flow")
    return {
        "condition": condition,
        "tower": match.group("tower"),
        "layer": int(match.group("layer")),
        "flow": None if flow is None else float(flow),
    }


def probe_seed(seed: int, condition_index: int, bin_index: int) -> int:
    return int(seed + condition_index * 1000 + bin_index)


def occupancy_train_loss(logits, yb, pos_weight):
    """Script-23 training objective: pos-weighted BCE + soft Dice."""
    from torch import nn

    prob = logits.sigmoid()
    dice = 1 - ((2 * (prob * yb).sum(1) + 1e-6) / (prob.sum(1) + yb.sum(1) + 1e-6)).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
    return bce + dice, float((bce + dice).detach().cpu())


def unweighted_bce(logits, yb) -> float:
    """Held-out H_C estimator: mean Bernoulli NLL, no pos-weight, no Dice."""
    from torch import nn

    return float(nn.functional.binary_cross_entropy_with_logits(logits, yb).detach().cpu())


def zscore_train(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_idx].mean(0)
    std = x[train_idx].std(0).copy()
    std[std < 1e-5] = 1.0
    return (x - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def load_q_matrix(samples: Sequence[Any], source: str = "observation_state") -> np.ndarray:
    if source != "observation_state":
        raise ValueError(
            f"Unsupported q source {source!r}. Use observation_state "
            "(joint-qpos extraction is not wired in this helper)."
        )
    rows = [np.asarray(sample.observation_state, dtype=np.float32).reshape(-1) for sample in samples]
    widths = {row.shape[0] for row in rows}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent observation_state widths: {sorted(widths)}")
    return np.stack(rows, axis=0)


def assemble_features(
    *,
    h: np.ndarray | None,
    q: np.ndarray,
    arm: str,
    matched_params: bool,
) -> np.ndarray:
    """Build decoder inputs. ``h`` may be omitted for the unmatched Q-only arm."""
    if arm == "q" and not matched_params:
        return np.asarray(q, dtype=np.float32)
    if h is None:
        raise ValueError("Hidden features are required for hq and matched-params q arms.")
    h = np.asarray(h, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    if h.ndim != 2 or q.ndim != 2 or h.shape[0] != q.shape[0]:
        raise ValueError(f"Bad feature shapes h={getattr(h, 'shape', None)} q={q.shape}")
    if arm == "q" and matched_params:
        return np.concatenate([np.zeros_like(h), q], axis=1)
    if arm == "hq":
        return np.concatenate([h, q], axis=1)
    raise ValueError(f"Unknown arm {arm!r}")


def make_decoder(
    input_dim: int,
    device,
    architecture: str = "mlp",
    hidden_size: int = HIDDEN,
):
    from torch import nn

    arch = normalize_decoder_architecture(architecture)
    if arch == "linear":
        return nn.Linear(int(input_dim), OUTPUT_DIM).to(device)
    if int(hidden_size) < 1:
        raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
    return nn.Sequential(
        nn.Linear(int(input_dim), int(hidden_size)),
        nn.GELU(),
        nn.Linear(int(hidden_size), OUTPUT_DIM),
    ).to(device)


def clone_state_dict(decoder) -> dict:
    return {key: value.detach().cpu().clone() for key, value in decoder.state_dict().items()}


def decoder_csv_metrics(result: dict[str, Any]) -> dict[str, Any]:
    skip = {"history"}
    return {key: value for key, value in result.items() if key not in skip}


def plot_train_test_curves(
    history_rows: list[dict[str, Any]],
    path: Path,
    *,
    hq_sample: int = 10,
    seed: int = 42,
    dpi: int = 180,
) -> list[str]:
    """Plot Q plus a random sample of HQ probes. Returns the HQ labels drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(history_rows)
    if frame.empty:
        raise ValueError("No history rows to plot.")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=dpi)
    q = frame[frame["arm"] == "q"].sort_values("epoch")
    if not q.empty:
        axes[0].plot(q["epoch"], q["train_loss"], color="black", lw=2.2, label="Q train")
        axes[0].plot(q["epoch"], q["test_loss"], color="black", lw=2.2, ls="--", label="Q test")
        axes[1].plot(q["epoch"], q["test_soft_iou"], color="black", lw=2.2, label="Q test soft IoU")
    hq = frame[frame["arm"] == "hq"].copy()
    probes = sorted({(str(row.condition), int(row.bin)) for row in hq.itertuples(index=False)})
    rng = np.random.default_rng(int(seed))
    if len(probes) > int(hq_sample):
        chosen = [probes[i] for i in sorted(rng.choice(len(probes), size=int(hq_sample), replace=False).tolist())]
    else:
        chosen = probes
    cmap = plt.get_cmap("tab10")
    labels = []
    for index, (condition, bin_index) in enumerate(chosen):
        subset = hq[(hq["condition"] == condition) & (hq["bin"].astype(int) == bin_index)].sort_values("epoch")
        color = cmap(index % 10)
        label = f"{condition} bin={bin_index}"
        labels.append(label)
        axes[0].plot(subset["epoch"], subset["train_loss"], color=color, lw=1.3, label=f"{label} train")
        axes[0].plot(subset["epoch"], subset["test_loss"], color=color, lw=1.3, ls="--", label=f"{label} test")
        axes[1].plot(subset["epoch"], subset["test_soft_iou"], color=color, lw=1.3, label=label)
    axes[0].set_title("Train / test loss (BCE + Dice)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=6, frameon=False, loc="best")
    axes[1].set_title("Test soft IoU")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("soft IoU")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=6, frameon=False, loc="best")
    fig.suptitle("Q and sampled (H,Q) train/test curves")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return labels


def train_decoder(
    *,
    x: np.ndarray,
    occupancy: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device,
    pos_weight,
    patience: int = 0,
    min_delta: float = 1e-4,
    architecture: str = "mlp",
    hidden_size: int = HIDDEN,
) -> dict[str, Any]:
    """Train on the occupancy train split. Early stop monitors train_loss only.

    No extra validation split: test metrics are logged each epoch for curves
    and for the final H_C report, but they do not decide stopping.
    ``patience <= 0`` runs all ``epochs``. Otherwise training stops after
    ``patience`` epochs without a train_loss improvement of at least ``min_delta``.
    Weights are restored to the best train_loss checkpoint before returning.
    """
    import torch
    from src.libero_self_occupancy import hard_iou, soft_iou

    if int(epochs) < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    torch.manual_seed(int(seed))
    x = np.asarray(x, dtype=np.float32)
    targets = occupancy.reshape(len(occupancy), -1).astype(np.float32)
    architecture = normalize_decoder_architecture(architecture)
    decoder = make_decoder(
        x.shape[1],
        device,
        architecture=architecture,
        hidden_size=hidden_size,
    )
    opt = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
    history: list[dict[str, Any]] = []
    best_train_loss = float("inf")
    best_epoch = 0
    best_state = clone_state_dict(decoder)
    wait = 0
    early_stopped = False
    stopped_epoch = 0
    for epoch in range(int(epochs)):
        decoder.train()
        epoch_losses: list[float] = []
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start : start + batch_size]
            xb = torch.from_numpy(x[idx]).to(device)
            yb = torch.from_numpy(targets[idx]).to(device)
            logits = decoder(xb)
            loss, loss_value = occupancy_train_loss(logits, yb, pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(loss_value)
        train_loss = float(np.mean(epoch_losses))
        decoder.eval()
        with torch.inference_mode():
            test_logits = decoder(torch.from_numpy(x[test_idx]).to(device))
            test_yb = torch.from_numpy(targets[test_idx]).to(device)
            _, test_loss = occupancy_train_loss(test_logits, test_yb, pos_weight)
            test_bce = unweighted_bce(test_logits, test_yb)
            pred = test_logits.sigmoid().cpu().numpy().reshape(-1, 16, 16, 16)
        soft = soft_iou(occupancy[test_idx], pred)
        hard = hard_iou(occupancy[test_idx], pred)
        stopped_epoch = epoch + 1
        history.append(
            {
                "epoch": stopped_epoch,
                "train_loss": train_loss,
                "test_loss": float(test_loss),
                "test_bce": float(test_bce),
                "test_soft_iou": float(soft),
                "test_hard_iou": float(hard),
            }
        )
        if train_loss < best_train_loss - float(min_delta):
            best_train_loss = train_loss
            best_epoch = stopped_epoch
            best_state = clone_state_dict(decoder)
            wait = 0
        else:
            wait += 1
            if int(patience) > 0 and wait >= int(patience):
                early_stopped = True
                break
    decoder.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    decoder.eval()
    with torch.inference_mode():
        test_logits = decoder(torch.from_numpy(x[test_idx]).to(device))
        test_yb = torch.from_numpy(targets[test_idx]).to(device)
        _, test_loss = occupancy_train_loss(test_logits, test_yb, pos_weight)
        test_bce = unweighted_bce(test_logits, test_yb)
        pred = test_logits.sigmoid().cpu().numpy().reshape(-1, 16, 16, 16)
    soft = soft_iou(occupancy[test_idx], pred)
    hard = hard_iou(occupancy[test_idx], pred)
    param_count = int(sum(p.numel() for p in decoder.parameters()))
    del decoder, opt
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()
    return {
        "train_loss": float(best_train_loss if best_epoch else history[-1]["train_loss"]),
        "test_loss": float(test_loss),
        "test_bce": float(test_bce),
        "soft_iou": float(soft),
        "hard_iou": float(hard),
        "input_dim": int(x.shape[1]),
        "param_count": param_count,
        "best_epoch": int(best_epoch or stopped_epoch),
        "stopped_epoch": int(stopped_epoch),
        "early_stopped": bool(early_stopped),
        "patience": int(patience),
        "min_delta": float(min_delta),
        "architecture": architecture,
        "hidden_size": None if architecture == "linear" else int(hidden_size),
        "decoder": decoder_description(architecture, hidden_size),
        "history": history,
    }


def ic_hat(test_bce_q: float, test_bce_hq: float) -> float:
    return float(test_bce_q) - float(test_bce_hq)


def summarize_ic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    summaries = []
    for tower, subset in frame.groupby("tower", sort=True):
        values = pd.to_numeric(subset["ic_hat"], errors="coerce").dropna()
        summaries.append(
            {
                "tower": tower,
                "n": int(len(values)),
                "ic_hat_mean": float(values.mean()) if len(values) else None,
                "ic_hat_median": float(values.median()) if len(values) else None,
                "frac_ic_gt_0": float((values > 0).mean()) if len(values) else None,
                "test_bce_hq_mean": float(pd.to_numeric(subset["test_bce_hq"], errors="coerce").mean()),
                "test_bce_q": float(subset["test_bce_q"].iloc[0]) if len(subset) else None,
            }
        )
    all_ic = pd.to_numeric(frame["ic_hat"], errors="coerce").dropna()
    return {
        "n": int(len(all_ic)),
        "ic_hat_mean": float(all_ic.mean()) if len(all_ic) else None,
        "frac_ic_gt_0": float((all_ic > 0).mean()) if len(all_ic) else None,
        "by_tower": summaries,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def annotate_history(
    history: Sequence[dict[str, Any]],
    *,
    arm: str,
    condition: str = "",
    tower: str = "",
    layer: Any = "",
    flow: Any = "",
    bin_index: Any = "",
    probe_seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for item in history:
        rows.append(
            {
                "arm": arm,
                "condition": condition,
                "tower": tower,
                "layer": layer,
                "flow": flow,
                "bin": bin_index,
                "probe_seed": int(probe_seed),
                **item,
            }
        )
    return rows


def plot_cmi_figures(ic_rows: list[dict[str, Any]], q_row: dict[str, Any], output_dir: Path, dpi: int = 180) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(ic_rows)
    values = pd.to_numeric(frame["ic_hat"], errors="coerce").dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=dpi)
    ax.hist(values, bins=min(24, max(8, len(values) // 4 or 8)), color="#2563eb", alpha=0.85)
    ax.axvline(0.0, color="black", lw=1)
    ax.set_xlabel(r"$\widehat{I}_C(H;O\mid Q)$ (test BCE$_Q$ $-$ test BCE$_{HQ}$)")
    ax.set_ylabel("count")
    ax.set_title("Capacity-constrained CMI over selected cells")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "01_ic_hist.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=dpi, sharey=True)
    for ax, tower in zip(axes, ("paligemma", "expert")):
        subset = frame[frame["tower"] == tower]
        if subset.empty:
            ax.set_title(tower)
            continue
        by_layer = subset.groupby("layer")["ic_hat"].mean()
        ax.plot(by_layer.index.to_numpy(), by_layer.to_numpy(), marker="o")
        ax.axhline(0.0, color="black", lw=1)
        ax.set_title(tower)
        ax.set_xlabel("layer")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"mean $\widehat{I}_C$")
    fig.tight_layout()
    fig.savefig(output_dir / "02_layerwise_ic.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=dpi)
    labels = []
    bce_q = []
    bce_hq = []
    for tower in ("paligemma", "expert"):
        subset = frame[frame["tower"] == tower]
        if subset.empty:
            continue
        labels.append(tower)
        bce_q.append(float(q_row["test_bce"]))
        bce_hq.append(float(pd.to_numeric(subset["test_bce_hq"], errors="coerce").mean()))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, bce_q, 0.36, label="Q only", color="#9ca3af")
    ax.bar(x + 0.18, bce_hq, 0.36, label="H+Q mean", color="#dc2626")
    ax.set_xticks(x, labels)
    ax.set_ylabel("held-out BCE")
    ax.set_title("H_C(O|Q) vs mean H_C(O|H,Q)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "03_bce_q_vs_hq.png")
    plt.close(fig)


def resolve_bin_indices(spec: str, stored_bins: Sequence[int], requested_bins: int) -> list[int]:
    from src.libero_self_occupancy import parse_index_spec

    stored = [int(v) for v in stored_bins]
    if spec in ("stored", "all"):
        return stored
    if spec == "auto-1/4":
        if not stored:
            raise ValueError("No stored bins.")
        n_keep = max(1, int(round(len(stored) / 4.0)))
        positions = np.linspace(0, len(stored) - 1, n_keep)
        return sorted({stored[int(round(pos))] for pos in positions})
    requested = parse_index_spec(spec, max_value=requested_bins)
    return [b for b in requested if b in set(stored)] or requested
