"""Group ablation frames by impact and overlay per-dimension histograms."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .utils import ensure_dir, write_json


GROUP1_LARGE = "group1_large"
GROUP2_NEAR_ZERO = "group2_near_zero"
GROUP3_SMALL = "group3_small"
UNASSIGNED = "unassigned"

GROUP_ORDER = (GROUP1_LARGE, GROUP2_NEAR_ZERO, GROUP3_SMALL)
GROUP_LABELS = {
    GROUP1_LARGE: "Group 1 large",
    GROUP2_NEAR_ZERO: "Group 2 ≈ 0",
    GROUP3_SMALL: "Group 3 small",
}
GROUP_COLORS = {
    GROUP1_LARGE: "#dc2626",
    GROUP2_NEAR_ZERO: "#6b7280",
    GROUP3_SMALL: "#2563eb",
}

# One histogram each, in this order. `task` is the added LIBERO-task dimension.
PLOT_DIMENSIONS: tuple[dict[str, str], ...] = (
    {"column": "good_minus_bad", "kind": "continuous", "title": "Impact value", "xlabel": r"mean $\Delta_{good}$ $-$ mean $\Delta_{bad}$"},
    {"column": "task", "kind": "categorical", "title": "LIBERO task", "xlabel": "task"},
    {"column": "progress", "kind": "continuous", "title": "Action progress t / (T−1)", "xlabel": "progress"},
    {"column": "phase_index", "kind": "discrete", "title": "Action phase", "xlabel": "phase index"},
    {"column": "motion_joint_l2", "kind": "continuous", "title": "Joint motion", "xlabel": r"$\|q_{t+1}-q_t\|_2$"},
    {"column": "motion_ee_pos_l2", "kind": "continuous", "title": "EE position motion", "xlabel": r"$\|\mathrm{EE}_{t+1}-\mathrm{EE}_t\|_2$"},
    {"column": "motion_ee_ori_rad", "kind": "continuous", "title": "EE orientation motion", "xlabel": "geodesic rad"},
    {"column": "motion_gripper_l2", "kind": "continuous", "title": "Gripper motion", "xlabel": r"$\|g_{t+1}-g_t\|_2$"},
    {"column": "vel_joint_l2", "kind": "continuous", "title": "Joint speed", "xlabel": r"$\|\dot q\|_2$"},
    {"column": "vel_ee_pos_l2", "kind": "continuous", "title": "EE linear speed", "xlabel": r"$\|\dot p_{\mathrm{EE}}\|_2$"},
    {"column": "vel_ee_ori_rad", "kind": "continuous", "title": "EE angular speed", "xlabel": "rad / step"},
    {"column": "vel_gripper_l2", "kind": "continuous", "title": "Gripper speed", "xlabel": r"$\|\dot g\|_2$"},
    {"column": "k_limit", "kind": "continuous", "title": "Joint-limit proximity", "xlabel": r"$K_{\mathrm{limit}}$"},
    {"column": "k_change", "kind": "continuous", "title": "Configuration change", "xlabel": r"$\|q_{t+\Delta}-q_t\|_2$"},
    {"column": "gripper_width", "kind": "continuous", "title": "Gripper width", "xlabel": "width"},
    {"column": "gripper_state_code", "kind": "discrete", "title": "Gripper state", "xlabel": "0=closed, 1=partial, 2=open"},
    {"column": "gripper_command", "kind": "discrete", "title": "Gripper command", "xlabel": "action[6]"},
    {"column": "d_ee_target", "kind": "continuous", "title": "EE ↔ target", "xlabel": "distance (m)"},
    {"column": "d_ee_nearest_obj", "kind": "continuous", "title": "EE ↔ nearest object", "xlabel": "distance (m)"},
    {"column": "d_target_nearest_link", "kind": "continuous", "title": "Target ↔ nearest robot link", "xlabel": "distance (m)"},
    {"column": "d_body_env", "kind": "continuous", "title": "Body–environment clearance", "xlabel": r"$d_{\mathrm{body-env}}$ (m)"},
    {"column": "contact_nontable", "kind": "discrete", "title": "Non-table contact", "xlabel": "0=no, 1=yes"},
    {"column": "target_held", "kind": "discrete", "title": "Target held", "xlabel": "0=no, 1=yes"},
)

VALID_MASKS = {
    "k_change": "k_change_valid",
    "motion_joint_l2": "motion_valid",
    "motion_ee_pos_l2": "motion_valid",
    "motion_ee_ori_rad": "motion_valid",
    "motion_gripper_l2": "motion_valid",
}


def merge_stats_and_impact(
    stats: pd.DataFrame,
    impact: pd.DataFrame,
    *,
    impact_column: str = "good_minus_bad",
) -> pd.DataFrame:
    if "row_index" not in stats.columns:
        raise ValueError("frame_stats.csv needs row_index")
    impact = impact.copy()
    if "frame_row" in impact.columns and "row_index" not in impact.columns:
        impact = impact.rename(columns={"frame_row": "row_index"})
    if "row_index" not in impact.columns:
        raise ValueError("impact table needs frame_row or row_index")
    if impact_column not in impact.columns:
        raise ValueError(f"impact table missing {impact_column}")
    keep = ["row_index", impact_column]
    merged = stats.merge(impact[keep], on="row_index", how="inner", validate="one_to_one")
    if len(merged) != len(stats) or len(merged) != len(impact):
        raise ValueError(
            f"row_index mismatch: stats={len(stats)} impact={len(impact)} merged={len(merged)}"
        )
    if impact_column != "good_minus_bad":
        merged = merged.rename(columns={impact_column: "good_minus_bad"})
    return merged.sort_values("row_index").reset_index(drop=True)


def assign_impact_groups(
    values: np.ndarray,
    *,
    tail_frac: float = 0.2,
) -> np.ndarray:
    """Disjoint equal-size tails plus a near-zero set; leftover is unassigned.

    Group 1 = highest ``tail_frac``; Group 3 = lowest ``tail_frac``;
    Group 2 = the same count among the remainder closest to 0.
    This keeps Group 2 actually near zero even when the impact distribution
    is shifted (here mean ≈ −0.4), which tertiles would not.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return np.array([], dtype=object)
    if not (0.0 < tail_frac <= 1.0 / 3.0):
        raise ValueError("tail_frac must be in (0, 1/3]")
    n_group = max(1, int(round(n * float(tail_frac))))
    if 3 * n_group > n:
        n_group = n // 3
    labels = np.full(n, UNASSIGNED, dtype=object)
    finite = np.isfinite(values)
    if finite.sum() < 3 * n_group:
        raise ValueError("Not enough finite impact values to form three groups")
    work = np.where(finite, values, np.nan)
    order = np.argsort(work, kind="mergesort")
    finite_order = order[np.isfinite(work[order])]
    small = finite_order[:n_group]
    large = finite_order[-n_group:]
    labels[small] = GROUP3_SMALL
    labels[large] = GROUP1_LARGE
    rest = finite_order[n_group:-n_group]
    rest_by_abs = rest[np.argsort(np.abs(work[rest]), kind="mergesort")]
    near = rest_by_abs[:n_group]
    labels[near] = GROUP2_NEAR_ZERO
    return labels


def attach_impact_groups(table: pd.DataFrame, *, tail_frac: float = 0.2) -> pd.DataFrame:
    out = table.copy()
    out["impact_group"] = assign_impact_groups(out["good_minus_bad"].to_numpy(), tail_frac=tail_frac)
    return out


def group_summary(table: pd.DataFrame) -> dict[str, Any]:
    impact = table["good_minus_bad"].to_numpy(dtype=np.float64)
    payload: dict[str, Any] = {
        "n_frames": int(len(table)),
        "tail_frac": None,
        "impact": {
            "mean": float(np.nanmean(impact)),
            "median": float(np.nanmedian(impact)),
            "frac_positive": float(np.nanmean(impact > 0)),
        },
        "groups": {},
    }
    for name in (*GROUP_ORDER, UNASSIGNED):
        mask = table["impact_group"].to_numpy() == name
        subset = impact[mask]
        payload["groups"][name] = {
            "n": int(mask.sum()),
            "impact_min": float(np.nanmin(subset)) if mask.any() else None,
            "impact_max": float(np.nanmax(subset)) if mask.any() else None,
            "impact_mean": float(np.nanmean(subset)) if mask.any() else None,
            "impact_mean_abs": float(np.nanmean(np.abs(subset))) if mask.any() else None,
        }
    return payload


def short_task_label(name: str) -> str:
    text = str(name)
    text = text.removeprefix("pick_up_the_black_bowl_")
    text = text.removesuffix("_and_place_it_on_the_plate")
    return text.replace("_", " ")


def plot_group_dimension_hists(
    table: pd.DataFrame,
    output_dir: Path,
    *,
    dpi: int = 180,
    bins: int = 30,
    dimensions: Sequence[dict[str, str]] = PLOT_DIMENSIONS,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = ensure_dir(output_dir)
    paths: list[Path] = []
    grouped = table[table["impact_group"].isin(GROUP_ORDER)]
    for index, spec in enumerate(dimensions):
        column = spec["column"]
        if column not in table.columns:
            continue
        fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=dpi)
        kind = spec["kind"]
        if kind == "categorical":
            _plot_categorical(ax, grouped, column)
        elif kind == "discrete":
            _plot_discrete(ax, grouped, column)
        else:
            _plot_continuous(ax, grouped, table, column, bins)
        ax.set_title(spec["title"])
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel("density" if kind == "continuous" else "fraction of group")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"{index:02d}_{column}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def write_group_outputs(
    table: pd.DataFrame,
    output_dir: Path,
    *,
    extra_summary: dict[str, Any] | None = None,
    dpi: int = 180,
    bins: int = 30,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    csv_path = output_dir / "grouped_frames.csv"
    table.to_csv(csv_path, index=False)
    figure_dir = ensure_dir(output_dir / "figures")
    figure_paths = plot_group_dimension_hists(table, figure_dir, dpi=dpi, bins=bins)
    summary = group_summary(table)
    summary["csv"] = str(csv_path)
    summary["figures"] = [str(path) for path in figure_paths]
    if extra_summary:
        summary.update(extra_summary)
    write_json(output_dir / "summary.json", summary)
    return summary


def _values_for_group(table: pd.DataFrame, column: str, group: str) -> np.ndarray:
    subset = table[table["impact_group"] == group]
    mask_col = VALID_MASKS.get(column)
    if mask_col is not None and mask_col in subset.columns:
        subset = subset[pd.to_numeric(subset[mask_col], errors="coerce").fillna(0) > 0]
    values = pd.to_numeric(subset[column], errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def _plot_continuous(
    ax: Any,
    grouped: pd.DataFrame,
    full: pd.DataFrame,
    column: str,
    bins: int,
) -> None:
    by_group = {group: _values_for_group(grouped, column, group) for group in GROUP_ORDER}
    pooled = pd.to_numeric(full[column], errors="coerce").to_numpy(dtype=np.float64)
    mask_col = VALID_MASKS.get(column)
    if mask_col is not None and mask_col in full.columns:
        valid = pd.to_numeric(full[mask_col], errors="coerce").fillna(0).to_numpy() > 0
        pooled = pooled[valid]
    pooled = pooled[np.isfinite(pooled)]
    if len(pooled) == 0:
        pooled = np.array([0.0, 1.0])
    lo, hi = np.nanpercentile(pooled, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(pooled)), float(np.max(pooled) + 1e-6)
    edges = np.linspace(lo, hi, bins + 1)
    for group in GROUP_ORDER:
        values = by_group[group]
        if len(values) == 0:
            continue
        ax.hist(
            values,
            bins=edges,
            density=True,
            alpha=0.45,
            color=GROUP_COLORS[group],
            label=f"{GROUP_LABELS[group]} (n={len(values)})",
        )


def _plot_discrete(ax: Any, grouped: pd.DataFrame, column: str) -> None:
    by_group = {group: _values_for_group(grouped, column, group) for group in GROUP_ORDER}
    pooled = np.concatenate([x for x in by_group.values() if len(x)]) if any(len(x) for x in by_group.values()) else np.array([0.0])
    categories = np.array(sorted(np.unique(pooled)))
    x = np.arange(len(categories), dtype=float)
    width = 0.24
    for offset, group in zip((-width, 0.0, width), GROUP_ORDER):
        values = by_group[group]
        counts = np.array([(values == cat).sum() for cat in categories], dtype=np.float64)
        frac = counts / max(len(values), 1)
        ax.bar(
            x + offset,
            frac,
            width=width,
            color=GROUP_COLORS[group],
            label=f"{GROUP_LABELS[group]} (n={len(values)})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_discrete_tick(column, cat) for cat in categories])


def _plot_categorical(ax: Any, grouped: pd.DataFrame, column: str) -> None:
    categories = sorted(grouped[column].astype(str).unique(), key=short_task_label)
    x = np.arange(len(categories), dtype=float)
    width = 0.24
    for offset, group in zip((-width, 0.0, width), GROUP_ORDER):
        subset = grouped[grouped["impact_group"] == group]
        counts = np.array([(subset[column].astype(str) == cat).sum() for cat in categories], dtype=np.float64)
        frac = counts / max(len(subset), 1)
        ax.bar(
            x + offset,
            frac,
            width=width,
            color=GROUP_COLORS[group],
            label=f"{GROUP_LABELS[group]} (n={len(subset)})",
        )
    ax.set_xticks(x)
    labels = [short_task_label(cat) for cat in categories]
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.figure.set_size_inches(11.2, 4.8)


def _discrete_tick(column: str, value: float) -> str:
    if column == "gripper_state_code":
        return {0: "closed", 1: "partial", 2: "open"}.get(int(value), str(int(value)))
    if column in {"contact_nontable", "target_held"}:
        return {0: "no", 1: "yes"}.get(int(value), str(value))
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"
