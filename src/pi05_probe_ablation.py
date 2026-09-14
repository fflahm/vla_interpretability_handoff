"""Selection, comparison, and plotting helpers for PI0.5 probe-guided ablations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRICS = (
    "mean_policy_action_delta_l2",
    "mean_env_action_delta_l2",
    "mean_gripper_position_delta_l2",
    "success_drop",
    "causal_impact_score",
)


def _occupancy_probe_cells(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"condition", "bin", "soft_iou"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Probe metrics are missing columns: {sorted(missing)}")
    frame = metrics.copy()
    parsed = frame["condition"].str.extract(
        r"^(?P<tower>paligemma|expert)/layer_(?P<layer>\d+)(?:/(?:static|t=(?P<flow_time>[0-9.]+)))?$"
    )
    if parsed[["tower", "layer"]].isna().any(axis=None):
        bad = frame.loc[parsed["tower"].isna(), "condition"].astype(str).unique().tolist()
        raise ValueError(f"Unrecognized PI0.5 condition names: {bad[:5]}")
    frame["tower"] = parsed["tower"]
    frame["layer_index"] = parsed["layer"].astype(int)
    frame["flow_time"] = pd.to_numeric(parsed["flow_time"], errors="coerce")
    frame["token_bin_index"] = pd.to_numeric(frame["bin"], errors="raise").astype(int)
    frame["soft_iou"] = pd.to_numeric(frame["soft_iou"], errors="raise")
    return (
        frame.groupby(["tower", "layer_index", "token_bin_index"], as_index=False)
        .agg(
            probe_soft_iou=("soft_iou", "mean"),
            probe_soft_iou_std=("soft_iou", "std"),
            probe_measurements=("soft_iou", "size"),
        )
    )


def select_probe_conditions(metrics: pd.DataFrame, per_tower: int = 5) -> pd.DataFrame:
    """Select disjoint top/bottom cells per tower after averaging flow-time repeats."""
    if per_tower <= 0:
        raise ValueError("`per_tower` must be positive.")
    cells = _occupancy_probe_cells(metrics)
    selected = []
    for tower in ("paligemma", "expert"):
        tower_cells = cells[cells["tower"] == tower].sort_values(
            ["probe_soft_iou", "layer_index", "token_bin_index"],
            ascending=[False, True, True],
        )
        if len(tower_cells) < 2 * per_tower:
            raise ValueError(
                f"Tower {tower} has {len(tower_cells)} unique cells; need at least {2 * per_tower}."
            )
        best = tower_cells.head(per_tower).copy()
        best["probe_group"] = "best"
        best["probe_rank"] = np.arange(1, len(best) + 1)
        worst = tower_cells.tail(per_tower).sort_values(
            ["probe_soft_iou", "layer_index", "token_bin_index"],
            ascending=[True, True, True],
        ).copy()
        worst["probe_group"] = "worst"
        worst["probe_rank"] = np.arange(1, len(worst) + 1)
        selected.extend([best, worst])
    result = pd.concat(selected, ignore_index=True)
    result["selection_mode"] = "global_extremes"
    result["pair_id"] = pd.NA
    result["iou_gap"] = pd.NA
    return result.sort_values(["tower", "probe_group", "probe_rank"]).reset_index(drop=True)


def select_layer_matched_probe_pairs(
    metrics: pd.DataFrame,
    *,
    pairs_per_tower: int = 5,
    bins_per_group: int = 2,
    min_iou_gap: float = 0.03,
    max_good_bin_repeats: int = 2,
    expert_max_bin: int | None = 48,
) -> pd.DataFrame:
    """Select high-IoU and low-IoU bins in the same layers, spread over depth.

    Each chosen layer contributes ``bins_per_group`` good bins and
    ``bins_per_group`` bad bins. Expert bins above ``expert_max_bin`` are
    dropped (PI0.5 Expert sequences are shorter than 96 bins). The good-group
    may reuse a token bin at most ``max_good_bin_repeats`` times so PaliGemma
    cannot collapse onto bin 80.
    """
    if pairs_per_tower <= 0:
        raise ValueError("`pairs_per_tower` must be positive.")
    if bins_per_group <= 0:
        raise ValueError("`bins_per_group` must be positive.")
    if min_iou_gap < 0:
        raise ValueError("`min_iou_gap` must be >= 0.")
    if max_good_bin_repeats <= 0:
        raise ValueError("`max_good_bin_repeats` must be positive.")
    cells = _occupancy_probe_cells(metrics)
    selected = []
    for tower in ("paligemma", "expert"):
        tower_cells = cells[cells["tower"] == tower].copy()
        if tower == "expert" and expert_max_bin is not None:
            tower_cells = tower_cells[tower_cells["token_bin_index"] <= int(expert_max_bin)]
        pairs = _select_pairs_for_tower(
            tower_cells,
            layers_per_tower=pairs_per_tower,
            bins_per_group=bins_per_group,
            min_iou_gap=min_iou_gap,
            max_good_bin_repeats=max_good_bin_repeats,
        )
        n_layers = len({row["layer_index"] for row in pairs})
        if n_layers < pairs_per_tower:
            raise ValueError(
                f"Tower {tower} only yielded {n_layers} layer-matched layers "
                f"(need {pairs_per_tower} layers x {bins_per_group} good/bad). "
                "Lower --min-iou-gap or --layers-per-tower."
            )
        selected.extend(pairs)
    result = pd.DataFrame(selected)
    result["selection_mode"] = "layer_matched"
    return result.sort_values(["tower", "pair_id", "probe_group"]).reset_index(drop=True)


def _select_pairs_for_tower(
    tower_cells: pd.DataFrame,
    *,
    layers_per_tower: int,
    bins_per_group: int,
    min_iou_gap: float,
    max_good_bin_repeats: int,
) -> list[dict[str, Any]]:
    if tower_cells.empty:
        return []
    need = 2 * bins_per_group
    layer_stats = []
    for layer, layer_cells in tower_cells.groupby("layer_index", sort=True):
        ordered = layer_cells.sort_values(
            ["probe_soft_iou", "token_bin_index"], ascending=[False, True]
        ).reset_index(drop=True)
        if len(ordered) < need:
            continue
        goods = ordered.iloc[:bins_per_group]
        bads = ordered.iloc[-bins_per_group:]
        good_bins = {int(v) for v in goods["token_bin_index"]}
        bad_bins = {int(v) for v in bads["token_bin_index"]}
        if good_bins & bad_bins:
            continue
        gap = float(goods["probe_soft_iou"].min()) - float(bads["probe_soft_iou"].max())
        if gap < min_iou_gap:
            continue
        layer_stats.append({"layer": int(layer), "gap": gap, "ordered": ordered})
    if not layer_stats:
        return []
    chosen_layers = _spread_pick_layers(layer_stats, layers_per_tower)
    good_bin_counts: dict[int, int] = {}
    pairs: list[dict[str, Any]] = []
    unused = [item for item in layer_stats if item["layer"] not in {c["layer"] for c in chosen_layers}]
    unused.sort(key=lambda item: (-item["gap"], item["layer"]))
    candidates = list(chosen_layers) + unused
    used_layers: set[int] = set()
    n_layers = 0
    for item in candidates:
        if n_layers >= layers_per_tower:
            break
        if item["layer"] in used_layers:
            continue
        bads = _pick_bad_bins(item["ordered"], bins_per_group)
        goods = _pick_good_bins(
            item["ordered"],
            bads=bads,
            min_iou_gap=min_iou_gap,
            good_bin_counts=good_bin_counts,
            max_good_bin_repeats=max_good_bin_repeats,
            bins_per_group=bins_per_group,
        )
        if goods is None:
            continue
        used_layers.add(item["layer"])
        n_layers += 1
        for good in goods:
            good_bin = int(good["token_bin_index"])
            good_bin_counts[good_bin] = good_bin_counts.get(good_bin, 0) + 1
        for slot, (good, bad) in enumerate(zip(goods, bads), start=1):
            pair_id = len(pairs) // 2 + 1
            iou_gap = float(good["probe_soft_iou"]) - float(bad["probe_soft_iou"])
            pairs.append(_cell_record(good, "best", pair_id, iou_gap, slot=slot))
            pairs.append(_cell_record(bad, "worst", pair_id, iou_gap, slot=slot))
    for rank_group in ("best", "worst"):
        ranked = [row for row in pairs if row["probe_group"] == rank_group]
        ranked.sort(key=lambda row: int(row["pair_id"]))
        for index, row in enumerate(ranked, start=1):
            row["probe_rank"] = index
    return pairs


def _spread_pick_layers(layer_stats: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    ordered = sorted(layer_stats, key=lambda item: item["layer"])
    if len(ordered) <= n:
        return ordered
    bands = [list(band) for band in np.array_split(np.arange(len(ordered)), n) if len(band)]
    picked = []
    used = set()
    for band in bands:
        options = [ordered[int(i)] for i in band]
        choice = max(options, key=lambda item: (item["gap"], -item["layer"]))
        picked.append(choice)
        used.add(choice["layer"])
    if len(picked) < n:
        rest = [item for item in sorted(ordered, key=lambda item: (-item["gap"], item["layer"])) if item["layer"] not in used]
        picked.extend(rest[: n - len(picked)])
    return picked[:n]


def _pick_bad_bins(ordered: pd.DataFrame, bins_per_group: int) -> list[pd.Series]:
    """Lowest-IoU bins first, so slot 1 is the more extreme bad cell."""
    return [ordered.iloc[-index] for index in range(1, bins_per_group + 1)]


def _pick_good_bins(
    ordered: pd.DataFrame,
    *,
    bads: list[pd.Series],
    min_iou_gap: float,
    good_bin_counts: dict[int, int],
    max_good_bin_repeats: int,
    bins_per_group: int,
) -> list[pd.Series] | None:
    bad_bins = {int(row["token_bin_index"]) for row in bads}
    bad_ceiling = max(float(row["probe_soft_iou"]) for row in bads)
    chosen: list[pd.Series] = []
    chosen_bins: set[int] = set()
    for _, row in ordered.iterrows():
        if len(chosen) >= bins_per_group:
            break
        bin_index = int(row["token_bin_index"])
        if bin_index in bad_bins or bin_index in chosen_bins:
            continue
        if good_bin_counts.get(bin_index, 0) >= max_good_bin_repeats:
            continue
        if float(row["probe_soft_iou"]) - bad_ceiling < min_iou_gap:
            continue
        chosen.append(row)
        chosen_bins.add(bin_index)
    if len(chosen) < bins_per_group:
        return None
    return chosen


def _cell_record(
    row: pd.Series,
    group: str,
    pair_id: int,
    iou_gap: float,
    *,
    slot: int = 1,
) -> dict[str, Any]:
    return {
        "tower": row["tower"],
        "layer_index": int(row["layer_index"]),
        "token_bin_index": int(row["token_bin_index"]),
        "probe_soft_iou": float(row["probe_soft_iou"]),
        "probe_soft_iou_std": None if pd.isna(row["probe_soft_iou_std"]) else float(row["probe_soft_iou_std"]),
        "probe_measurements": int(row["probe_measurements"]),
        "probe_group": group,
        "pair_id": int(pair_id),
        "layer_slot": int(slot),
        "iou_gap": float(iou_gap),
        "probe_rank": int(pair_id),
    }


def load_rollout_for_comparison(rollout_dir: Path) -> dict[str, Any]:
    summary = json.loads((rollout_dir / "summary.json").read_text(encoding="utf-8"))
    steps = []
    for episode in summary.get("episodes", []):
        path = Path(episode["steps_jsonl"])
        if not path.exists():
            path = rollout_dir / f"episode_{int(episode['episode_index']):03d}" / "steps.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            steps.extend(json.loads(line) for line in handle if line.strip())
    return {"summary": summary, "steps": steps}


def compare_rollouts(baseline: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    baseline_steps = {
        (int(row["episode_index"]), int(row["step"])): row
        for row in baseline["steps"]
        if row.get("action") is not None
    }
    policy_deltas: list[float] = []
    env_deltas: list[float] = []
    gripper_deltas: list[float] = []
    for row in observed["steps"]:
        base = baseline_steps.get((int(row["episode_index"]), int(row["step"])))
        if base is None:
            continue
        # PI0.5 rollout records the policy output after environment postprocessing as `action`.
        env_delta = vector_l2(row.get("action"), base.get("action"))
        env_deltas.append(env_delta)
        policy_deltas.append(
            vector_l2(row.get("policy_pred_action", row.get("action")), base.get("policy_pred_action", base.get("action")))
        )
        gripper_deltas.append(vector_l2(row.get("gripper_position"), base.get("gripper_position")))

    baseline_success = float(baseline["summary"].get("success_rate", 0.0))
    observed_success = float(observed["summary"].get("success_rate", 0.0))
    action_delta = safe_mean(policy_deltas)
    env_action_delta = safe_mean(env_deltas)
    gripper_delta = safe_mean(gripper_deltas)
    success_drop = baseline_success - observed_success
    return {
        "baseline_success_rate": baseline_success,
        "success_rate": observed_success,
        "success_drop": success_drop,
        "mean_policy_action_delta_l2": action_delta,
        "mean_env_action_delta_l2": env_action_delta,
        "mean_gripper_position_delta_l2": gripper_delta,
        "num_compared_steps": int(len(policy_deltas)),
        "causal_impact_score": (
            action_delta + env_action_delta + gripper_delta + max(0.0, success_drop) * 10.0
        ),
    }


def vector_l2(left: Any, right: Any) -> float:
    if left is None or right is None:
        return float("nan")
    left_arr = np.asarray(left, dtype=np.float64).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_arr.shape != right_arr.shape:
        return float("nan")
    return float(np.linalg.norm(left_arr - right_arr))


def safe_mean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(finite.mean()) if finite.size else float("nan")


def summarize_results(results: pd.DataFrame) -> dict[str, Any]:
    summaries = []
    for (tower, group), rows in results.groupby(["tower", "probe_group"], sort=True):
        entry: dict[str, Any] = {"tower": tower, "probe_group": group, "n": int(len(rows))}
        for metric in METRICS:
            values = pd.to_numeric(rows[metric], errors="coerce").dropna()
            entry[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            entry[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else None
        summaries.append(entry)

    correlations = []
    for tower, rows in results.groupby("tower", sort=True):
        for metric in METRICS:
            pair = rows[["probe_soft_iou", metric]].dropna()
            corr = pair["probe_soft_iou"].rank().corr(pair[metric].rank()) if len(pair) >= 2 else np.nan
            correlations.append(
                {
                    "tower": tower,
                    "metric": metric,
                    "spearman_r": None if not np.isfinite(corr) else float(corr),
                    "n": int(len(pair)),
                }
            )
    return {"group_summaries": summaries, "spearman_correlations": correlations}


def plot_results(results: pd.DataFrame, output_dir: Path, dpi: int = 180) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"worst": "#9ca3af", "best": "#dc2626"}
    towers = ("paligemma", "expert")

    fig, axes = plt.subplots(len(METRICS), 2, figsize=(11, 3.2 * len(METRICS)), dpi=dpi)
    rng = np.random.default_rng(0)
    for row_index, metric in enumerate(METRICS):
        for col_index, tower in enumerate(towers):
            ax = axes[row_index, col_index]
            subset = results[results["tower"] == tower]
            for x, group in enumerate(("worst", "best")):
                values = pd.to_numeric(
                    subset.loc[subset["probe_group"] == group, metric], errors="coerce"
                ).dropna().to_numpy()
                if len(values):
                    ax.bar(x, values.mean(), color=colors[group], alpha=0.45)
                    ax.scatter(x + rng.uniform(-0.09, 0.09, len(values)), values, color=colors[group], s=28)
            ax.set_xticks([0, 1], ["worst probe", "best probe"])
            ax.set_ylabel(metric)
            ax.set_title(tower)
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("PI0.5 probe quality vs causal ablation impact", y=1.0)
    fig.tight_layout()
    fig.savefig(output_dir / "01_best_vs_worst_impact.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=dpi)
    for ax, tower in zip(axes, towers):
        subset = results[results["tower"] == tower]
        for group in ("worst", "best"):
            rows = subset[subset["probe_group"] == group]
            ax.scatter(
                rows["probe_soft_iou"],
                rows["mean_policy_action_delta_l2"],
                s=50,
                color=colors[group],
                label=group,
            )
            for item in rows.itertuples():
                ax.annotate(
                    f"L{item.layer_index} B{item.token_bin_index}",
                    (item.probe_soft_iou, item.mean_policy_action_delta_l2),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
        pair = subset[["probe_soft_iou", "mean_policy_action_delta_l2"]].dropna()
        corr = pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()) if len(pair) >= 2 else np.nan
        ax.set_title(f"{tower} (Spearman r={corr:.2f})")
        ax.set_xlabel("PI0.5 probe soft IoU")
        ax.set_ylabel("mean policy action delta L2")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "02_soft_iou_vs_action_impact.png", bbox_inches="tight")
    plt.close(fig)


def condition_dir_name(row: Any, mode: str = "zero") -> str:
    return (
        f"{row.probe_group}_{row.tower}_layer_{int(row.layer_index):02d}_"
        f"bin_{int(row.token_bin_index):03d}_{mode}"
    )


def load_step_impacts(rollout_dir: Path, baseline_steps: dict[tuple[int, int], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for episode_dir in sorted(rollout_dir.glob("episode_*")):
        steps_path = episode_dir / "steps.jsonl"
        if not steps_path.exists():
            continue
        with steps_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (int(row["episode_index"]), int(row["step"]))
                base = baseline_steps.get(key)
                if base is None or row.get("action") is None:
                    continue
                rows.append(
                    {
                        "episode_index": key[0],
                        "step": key[1],
                        "is_replan": bool(row.get("is_replan", False)),
                        "action_impact": vector_l2(row.get("action"), base.get("action")),
                        "gripper_impact": vector_l2(row.get("gripper_position"), base.get("gripper_position")),
                    }
                )
    return pd.DataFrame(rows)


def frame_good_vs_bad_impacts(
    ablation_dir: Path,
    selected: pd.DataFrame,
    *,
    mode: str = "zero",
    impact_column: str = "action_impact",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Pair every best cell with every worst cell in the same tower, per frame.

    ``iou_diff`` is best minus worst probe soft IoU. Impact is L2 vs baseline.
    """
    baseline = load_rollout_for_comparison(ablation_dir / "baseline")
    baseline_steps = {
        (int(row["episode_index"]), int(row["step"])): row
        for row in baseline["steps"]
        if row.get("action") is not None
    }
    impacts: dict[tuple[str, int, int], pd.DataFrame] = {}
    for row in selected.itertuples(index=False):
        name = condition_dir_name(row, mode=mode)
        path = ablation_dir / name
        if not (path / "summary.json").exists():
            raise FileNotFoundError(f"Missing ablation rollout: {path}")
        impacts[(row.tower, int(row.layer_index), int(row.token_bin_index))] = load_step_impacts(
            path, baseline_steps
        )

    pair_frames: list[dict[str, Any]] = []
    group_frames: list[dict[str, Any]] = []
    for tower, tower_rows in selected.groupby("tower", sort=True):
        best = tower_rows[tower_rows["probe_group"] == "best"]
        worst = tower_rows[tower_rows["probe_group"] == "worst"]
        if best.empty or worst.empty:
            continue
        merged = None
        for item in pd.concat([best, worst]).itertuples(index=False):
            frame = impacts[(item.tower, int(item.layer_index), int(item.token_bin_index))][
                ["episode_index", "step", "is_replan", impact_column]
            ].rename(columns={impact_column: f"{item.probe_group}_{int(item.layer_index)}_{int(item.token_bin_index)}"})
            if merged is None:
                merged = frame
            else:
                extra = frame.drop(columns=["is_replan"])
                merged = merged.merge(extra, on=["episode_index", "step"], how="inner")
        best_cols = [c for c in merged.columns if c.startswith("best_")]
        worst_cols = [c for c in merged.columns if c.startswith("worst_")]
        merged["impact_good_mean"] = merged[best_cols].mean(axis=1)
        merged["impact_bad_mean"] = merged[worst_cols].mean(axis=1)
        merged["good_gt_bad_group"] = merged["impact_good_mean"] > merged["impact_bad_mean"]
        merged["tower"] = tower
        group_frames.append(
            merged[
                [
                    "tower",
                    "episode_index",
                    "step",
                    "is_replan",
                    "impact_good_mean",
                    "impact_bad_mean",
                    "good_gt_bad_group",
                ]
            ]
        )
        for good in best.itertuples(index=False):
            g_col = f"best_{int(good.layer_index)}_{int(good.token_bin_index)}"
            for bad in worst.itertuples(index=False):
                b_col = f"worst_{int(bad.layer_index)}_{int(bad.token_bin_index)}"
                iou_diff = float(good.probe_soft_iou) - float(bad.probe_soft_iou)
                chunk = merged[["episode_index", "step", "is_replan", g_col, b_col]].copy()
                chunk["tower"] = tower
                chunk["good_layer"] = int(good.layer_index)
                chunk["good_bin"] = int(good.token_bin_index)
                chunk["bad_layer"] = int(bad.layer_index)
                chunk["bad_bin"] = int(bad.token_bin_index)
                chunk["iou_good"] = float(good.probe_soft_iou)
                chunk["iou_bad"] = float(bad.probe_soft_iou)
                chunk["iou_diff"] = iou_diff
                chunk["impact_good"] = chunk[g_col]
                chunk["impact_bad"] = chunk[b_col]
                chunk["impact_diff"] = chunk[g_col] - chunk[b_col]
                chunk["good_gt_bad"] = chunk[g_col] > chunk[b_col]
                pair_frames.append(
                    chunk.drop(columns=[g_col, b_col])
                )

    pair_frame = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    group_frame = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
    summary: dict[str, Any] = {"impact_column": impact_column, "by_tower": []}
    for tower, subset in pair_frame.groupby("tower", sort=True):
        group = group_frame[group_frame["tower"] == tower]
        n = int(len(subset))
        n_gt = int(subset["good_gt_bad"].sum())
        n_group = int(len(group))
        n_group_gt = int(group["good_gt_bad_group"].sum()) if n_group else 0
        replan = subset[subset["is_replan"]]
        summary["by_tower"].append(
            {
                "tower": tower,
                "n_pairs": int(subset.groupby(["good_layer", "good_bin", "bad_layer", "bad_bin"]).ngroups),
                "n_pair_frames": n,
                "n_pair_frames_good_gt_bad": n_gt,
                "frac_pair_frames_good_gt_bad": float(n_gt / n) if n else None,
                "n_group_frames": n_group,
                "n_group_frames_good_mean_gt_bad_mean": n_group_gt,
                "frac_group_frames_good_mean_gt_bad_mean": float(n_group_gt / n_group) if n_group else None,
                "n_replan_pair_frames": int(len(replan)),
                "frac_replan_pair_frames_good_gt_bad": float(replan["good_gt_bad"].mean()) if len(replan) else None,
                "mean_iou_diff": float(subset["iou_diff"].mean()) if n else None,
                "mean_impact_diff": float(subset["impact_diff"].mean()) if n else None,
            }
        )
    if len(pair_frame):
        n = int(len(pair_frame))
        n_gt = int(pair_frame["good_gt_bad"].sum())
        n_group = int(len(group_frame))
        n_group_gt = int(group_frame["good_gt_bad_group"].sum()) if n_group else 0
        summary["overall"] = {
            "n_pair_frames": n,
            "n_pair_frames_good_gt_bad": n_gt,
            "frac_pair_frames_good_gt_bad": float(n_gt / n),
            "n_group_frames": n_group,
            "n_group_frames_good_mean_gt_bad_mean": n_group_gt,
            "frac_group_frames_good_mean_gt_bad_mean": float(n_group_gt / n_group) if n_group else None,
        }
    return pair_frame, group_frame, summary


def plot_frame_iou_diff_distribution(
    pair_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int = 180,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    towers = [t for t in ("paligemma", "expert") if t in set(pair_frame["tower"])]

    fig, axes = plt.subplots(1, len(towers), figsize=(6.2 * len(towers), 4.8), dpi=dpi, squeeze=False)
    for ax, tower in zip(axes[0], towers):
        subset = pair_frame[pair_frame["tower"] == tower]
        hb = ax.hexbin(
            subset["iou_diff"],
            subset["impact_diff"],
            gridsize=35,
            cmap="viridis",
            mincnt=1,
            linewidths=0,
        )
        ax.axhline(0.0, color="black", lw=1)
        frac = float(subset["good_gt_bad"].mean())
        ax.set_title(f"{tower}: P(I_good>I_bad)={frac:.3f}  n={len(subset)} frames×pairs")
        ax.set_xlabel("IoU difference (good − bad)")
        ax.set_ylabel("impact difference (I_good − I_bad)")
        ax.grid(alpha=0.25)
        fig.colorbar(hb, ax=ax, label="frame-pair count")
    fig.tight_layout()
    path = output_dir / "03_frame_impact_vs_iou_diff.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    pair_summary = (
        pair_frame.groupby(
            ["tower", "good_layer", "good_bin", "bad_layer", "bad_bin", "iou_diff"],
            as_index=False,
        )
        .agg(
            n_frames=("good_gt_bad", "size"),
            n_good_gt_bad=("good_gt_bad", "sum"),
            frac_good_gt_bad=("good_gt_bad", "mean"),
            mean_impact_diff=("impact_diff", "mean"),
        )
    )
    fig, axes = plt.subplots(1, len(towers), figsize=(6.2 * len(towers), 4.6), dpi=dpi, squeeze=False)
    for ax, tower in zip(axes[0], towers):
        subset = pair_summary[pair_summary["tower"] == tower]
        ax.scatter(subset["iou_diff"], subset["frac_good_gt_bad"], s=42, c="#2563eb")
        ax.axhline(0.5, color="black", lw=1, ls="--")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"{tower}: per good×bad pair")
        ax.set_xlabel("IoU difference (good − bad)")
        ax.set_ylabel("fraction of frames with I_good > I_bad")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "04_pair_frac_good_gt_bad_vs_iou_diff.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    pair_summary.to_csv(output_dir / "frame_pair_iou_diff_summary.csv", index=False)
    return saved


def plot_group_action_delta_diff_hist(
    group_frame: pd.DataFrame,
    output_dir: Path,
    dpi: int = 180,
    bins: int = 40,
) -> Path:
    """Histogram of per-frame (mean good − mean bad) action delta vs baseline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = group_frame.copy()
    frame["action_delta_diff"] = frame["impact_good_mean"] - frame["impact_bad_mean"]
    towers = [t for t in ("paligemma", "expert") if t in set(frame["tower"])]
    colors = {"paligemma": "#2563eb", "expert": "#dc2626"}
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=dpi)
    values_all = frame["action_delta_diff"].to_numpy()
    lo, hi = np.nanpercentile(values_all, [0.5, 99.5])
    span = max(hi - lo, 1e-6)
    edges = np.linspace(lo - 0.05 * span, hi + 0.05 * span, bins + 1)
    for tower in towers:
        subset = frame.loc[frame["tower"] == tower, "action_delta_diff"].to_numpy()
        n_gt = int((subset > 0).sum())
        ax.hist(
            subset,
            bins=edges,
            alpha=0.55,
            color=colors.get(tower, "#4b5563"),
            label=f"{tower}  good>bad in {n_gt}/{len(subset)} frames",
        )
    ax.axvline(0.0, color="black", lw=1.2)
    n_gt_all = int((values_all > 0).sum())
    ax.set_xlabel(r"mean action $\Delta$ (good) $-$ mean action $\Delta$ (bad)")
    ax.set_ylabel("number of frames")
    ax.set_title(f"Per-frame group-mean action impact  (good>bad in {n_gt_all}/{len(values_all)} frames)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "05_frame_group_action_delta_diff_hist.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


