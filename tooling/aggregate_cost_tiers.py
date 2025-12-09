from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


AGG_DIR = Path("experiments/artifacts/walk_forward_aggregates")
ROOT = Path("experiments/artifacts")
COST_ORDER = ["low", "base", "high"]
REGIME_ORDER = [
    "full_grid",
    "stress_w240_s5",
    "stress_high_w240_s5",
    "heavy_w480_s10",
    "heavy_high_w480_s10",
    "heavy_base_stoch_w480_s10",
    "heavy_high_stoch_w480_s10",
]
SYMBOLS = ["XBT", "ETH", "SOL", "LTC", "ADA"]


@dataclass
class RunInfo:
    symbol: str
    cost_tier: str
    regime: str
    results_path: Path
    mean_sharpe: float
    mean_final_value: float
    n_seeds: int
    window: Optional[int]


def find_results_json(run_dir: Path) -> Optional[Path]:
    for candidate in run_dir.glob("*/results.json"):
        return candidate
    return None


def extract_window(dir_name: str) -> Optional[int]:
    match = re.search(r"w(\d+)", dir_name)
    return int(match.group(1)) if match else None


def load_aggregate(path: Path) -> tuple[float, float, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    agg = data.get("aggregate", {})
    return agg.get("mean_sharpe"), agg.get("mean_final_value"), agg.get("n_seeds", 0)


def load_seed_metrics(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    seeds = data.get("seeds", [])
    rows: List[dict] = []
    for seed_entry in seeds:
        rows.append(
            {
                "seed": seed_entry.get("seed"),
                "mean_sharpe": seed_entry.get("mean_sharpe"),
                "mean_final_value": seed_entry.get("mean_final_value"),
            }
        )
    return rows


def collect_full_runs() -> List[RunInfo]:
    runs: List[RunInfo] = []
    base_dir = ROOT / "walk_forward_full"
    if not base_dir.exists():
        return runs

    for symbol in ["XBT", "ETH", "SOL"]:
        for cost_tier in COST_ORDER:
            dir_path = base_dir / f"{symbol}_{cost_tier}"
            if not dir_path.exists():
                continue
            result_path = find_results_json(dir_path)
            if not result_path:
                continue
            mean_sharpe, mean_final_value, n_seeds = load_aggregate(result_path)
            runs.append(
                RunInfo(
                    symbol=symbol,
                    cost_tier=cost_tier,
                    regime="full_grid",
                    results_path=result_path,
                    mean_sharpe=mean_sharpe,
                    mean_final_value=mean_final_value,
                    n_seeds=n_seeds,
                    window=extract_window(dir_path.name),
                )
            )
    return runs


def collect_stress_runs() -> List[RunInfo]:
    runs: List[RunInfo] = []
    base_dir = ROOT / "walk_forward_stress"
    if not base_dir.exists():
        return runs

    for symbol in SYMBOLS:
        dir_path = base_dir / f"{symbol}_seed5_w240_base"
        if not dir_path.exists():
            continue
        result_path = find_results_json(dir_path)
        if not result_path:
            continue
        mean_sharpe, mean_final_value, n_seeds = load_aggregate(result_path)
        runs.append(
            RunInfo(
                symbol=symbol,
                cost_tier="base",
                regime="stress_w240_s5",
                results_path=result_path,
                mean_sharpe=mean_sharpe,
                mean_final_value=mean_final_value,
                n_seeds=n_seeds,
                window=extract_window(dir_path.name),
            )
        )
    return runs


def collect_stress_high_runs() -> List[RunInfo]:
    runs: List[RunInfo] = []
    base_dir = ROOT / "walk_forward_stress_high"
    if not base_dir.exists():
        return runs

    for symbol in SYMBOLS:
        dir_path = base_dir / f"{symbol}_high"
        if not dir_path.exists():
            continue
        result_path = find_results_json(dir_path)
        if not result_path:
            continue
        mean_sharpe, mean_final_value, n_seeds = load_aggregate(result_path)
        runs.append(
            RunInfo(
                symbol=symbol,
                cost_tier="high",
                regime="stress_high_w240_s5",
                results_path=result_path,
                mean_sharpe=mean_sharpe,
                mean_final_value=mean_final_value,
                n_seeds=n_seeds,
                window=extract_window(dir_path.name),
            )
        )
    return runs


def collect_heavy_runs() -> List[RunInfo]:
    runs: List[RunInfo] = []
    base_dir = ROOT / "walk_forward_heavy"
    if not base_dir.exists():
        return runs

    for symbol in SYMBOLS:
        dir_path = base_dir / f"{symbol}_base_w480_s10"
        if not dir_path.exists():
            continue
        result_path = find_results_json(dir_path)
        if not result_path:
            continue
        mean_sharpe, mean_final_value, n_seeds = load_aggregate(result_path)
        runs.append(
            RunInfo(
                symbol=symbol,
                cost_tier="base",
                regime="heavy_w480_s10",
                results_path=result_path,
                mean_sharpe=mean_sharpe,
                mean_final_value=mean_final_value,
                n_seeds=n_seeds,
                window=extract_window(dir_path.name),
            )
        )
    return runs


def collect_heavy_high_runs() -> List[RunInfo]:
    runs: List[RunInfo] = []
    base_dir = ROOT / "walk_forward_heavy_high"
    if not base_dir.exists():
        return runs

    for symbol in SYMBOLS:
        dir_path = base_dir / f"{symbol}_high_w480_s10"
        if not dir_path.exists():
            continue
        result_path = find_results_json(dir_path)
        if not result_path:
            continue
        mean_sharpe, mean_final_value, n_seeds = load_aggregate(result_path)
        runs.append(
            RunInfo(
                symbol=symbol,
                cost_tier="high",
                regime="heavy_high_w480_s10",
                results_path=result_path,
                mean_sharpe=mean_sharpe,
                mean_final_value=mean_final_value,
                n_seeds=n_seeds,
                window=extract_window(dir_path.name),
            )
        )
    return runs


def collect_heavy_high_stoch_runs() -> List[RunInfo]:
    runs: List[RunInfo] = []
    base_dir = ROOT / "walk_forward_heavy_high_stoch"
    if not base_dir.exists():
        return runs

    for symbol in SYMBOLS:
        dir_path = base_dir / f"{symbol}_high_stoch_w480_s10"
        if not dir_path.exists():
            continue
        result_path = find_results_json(dir_path)
        if not result_path:
            continue
        mean_sharpe, mean_final_value, n_seeds = load_aggregate(result_path)
        runs.append(
            RunInfo(
                symbol=symbol,
                cost_tier="high",
                regime="heavy_high_stoch_w480_s10",
                results_path=result_path,
                mean_sharpe=mean_sharpe,
                mean_final_value=mean_final_value,
                n_seeds=n_seeds,
                window=extract_window(dir_path.name),
            )
        )
    return runs


def collect_heavy_base_stoch_runs() -> List[RunInfo]:
    runs: List[RunInfo] = []
    base_dir = ROOT / "walk_forward_heavy_base_stoch"
    if not base_dir.exists():
        return runs

    for symbol in SYMBOLS:
        dir_path = base_dir / f"{symbol}_base_stoch_w480_s10"
        if not dir_path.exists():
            continue
        result_path = find_results_json(dir_path)
        if not result_path:
            continue
        mean_sharpe, mean_final_value, n_seeds = load_aggregate(result_path)
        runs.append(
            RunInfo(
                symbol=symbol,
                cost_tier="base",
                regime="heavy_base_stoch_w480_s10",
                results_path=result_path,
                mean_sharpe=mean_sharpe,
                mean_final_value=mean_final_value,
                n_seeds=n_seeds,
                window=extract_window(dir_path.name),
            )
        )
    return runs


def as_dataframe(entries: Iterable[RunInfo]) -> pd.DataFrame:
    df = pd.DataFrame([
        {
            "symbol": e.symbol,
            "cost_tier": e.cost_tier,
            "regime": e.regime,
            "results_path": str(e.results_path),
            "mean_sharpe": e.mean_sharpe,
            "mean_final_value": e.mean_final_value,
            "n_seeds": e.n_seeds,
            "window": e.window,
        }
        for e in entries
    ])
    if not df.empty:
        df["cost_tier"] = pd.Categorical(df["cost_tier"], categories=COST_ORDER, ordered=True)
        df["regime"] = pd.Categorical(df["regime"], categories=[r for r in REGIME_ORDER if r in df["regime"].unique()], ordered=True)
    return df


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, output: Path) -> None:
    symbols = sorted(df["symbol"].unique())
    regimes = [r for r in REGIME_ORDER if r in df["regime"].unique()]
    width = 0.8 / max(len(regimes), 1)
    x_base = np.arange(len(COST_ORDER))

    fig, axes = plt.subplots(1, len(symbols), figsize=(4 * len(symbols), 4), sharey="row")
    if len(symbols) == 1:
        axes = [axes]

    for idx, symbol in enumerate(symbols):
        ax = axes[idx]
        subset = df[df["symbol"] == symbol]
        for r_idx, regime in enumerate(regimes):
            reg_df = subset[subset["regime"] == regime].set_index("cost_tier")
            values = [reg_df[metric].get(tier) if tier in reg_df.index else np.nan for tier in COST_ORDER]
            offsets = x_base + (r_idx - (len(regimes) - 1) / 2) * width
            ax.bar(offsets, values, width=width, label=regime)
        ax.set_title(symbol)
        ax.set_xticks(x_base)
        ax.set_xticklabels(COST_ORDER)
        ax.set_ylabel(ylabel)
        ax.axhline(0.0, color="gray", linewidth=0.8)
    fig.legend(loc="upper center", ncol=min(len(regimes), 4), bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def add_baseline_deltas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["sharpe_delta_vs_low"] = np.nan
    df["final_value_delta_vs_low"] = np.nan

    # Baseline per symbol+regime = lowest available cost tier in COST_ORDER
    for (symbol, regime), group in df.groupby(["symbol", "regime"], observed=True):
        for tier in COST_ORDER:
            candidates = group[group["cost_tier"] == tier]
            if not candidates.empty:
                baseline_sharpe = candidates.iloc[0]["mean_sharpe"]
                baseline_fv = candidates.iloc[0]["mean_final_value"]
                df.loc[(df["symbol"] == symbol) & (df["regime"] == regime), "sharpe_delta_vs_low"] = (
                    df.loc[(df["symbol"] == symbol) & (df["regime"] == regime), "mean_sharpe"] - baseline_sharpe
                )
                df.loc[(df["symbol"] == symbol) & (df["regime"] == regime), "final_value_delta_vs_low"] = (
                    df.loc[(df["symbol"] == symbol) & (df["regime"] == regime), "mean_final_value"] - baseline_fv
                )
                break
    return df


def plot_seed_boxplots(df: pd.DataFrame, metric: str, ylabel: str, output: Path) -> None:
    if df.empty:
        return
    symbols = sorted(df["symbol"].unique())
    regimes = [r for r in REGIME_ORDER if r in df["regime"].unique()]
    fig, axes = plt.subplots(1, len(symbols), figsize=(4 * len(symbols), 4), sharey="row")
    if len(symbols) == 1:
        axes = [axes]

    for idx, symbol in enumerate(symbols):
        ax = axes[idx]
        subset = df[df["symbol"] == symbol]
        labels: List[str] = []
        series: List[pd.Series] = []
        for regime in regimes:
            for cost in COST_ORDER:
                sel = subset[(subset["regime"] == regime) & (subset["cost_tier"] == cost)]
                if sel.empty:
                    continue
                labels.append(f"{cost}-{regime}")
                series.append(sel[metric].dropna())
        if not series:
            ax.set_visible(False)
            continue
        ax.boxplot(series, tick_labels=labels, vert=True, patch_artist=True)
        ax.set_title(symbol)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(0.0, color="gray", linewidth=0.8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_det_vs_stoch_bars(comp_df: pd.DataFrame, output: Path, metric: str, title: str, ylabel: str) -> None:
    if comp_df.empty:
        return
    symbols = sorted(comp_df["symbol"].unique())
    tiers = ["base", "high"]
    width = 0.35
    x = np.arange(len(symbols))
    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = {"base": -width / 2, "high": width / 2}
    colors = {"base": "#4c72b0", "high": "#dd8452"}
    for tier in tiers:
        subset = comp_df[comp_df["cost_tier"] == tier]
        vals = [subset[metric][subset["symbol"] == s].values[0] if not subset[subset["symbol"] == s].empty else np.nan for s in symbols]
        ax.bar(x + offsets[tier], vals, width=width, label=f"{tier} Δ", color=colors.get(tier, None))
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(symbols)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def export_pdf_bundle(report_path: Path, image_paths: List[Path], output: Path) -> None:
    if not report_path.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        # page 1: report text (wrapped)
        text = report_path.read_text(encoding="utf-8")
        wrapped = textwrap.fill(text, width=110)
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.02, 0.98, wrapped, va="top", ha="left", fontsize=8, family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # subsequent pages: plots
        for img_path in image_paths:
            if not img_path.exists():
                continue
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            img = plt.imread(img_path)
            ax.imshow(img)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def build_report(df: pd.DataFrame, dispersion_df: Optional[pd.DataFrame] = None) -> str:
    lines: List[str] = []
    lines.append("# Cost-tier summary\n")
    lines.append("Columns: symbol | cost_tier | regime | n_seeds | window | mean_sharpe | mean_final_value\n")

    # Table
    table_df = df.copy()
    table_df["mean_sharpe"] = table_df["mean_sharpe"].round(4)
    table_df["mean_final_value"] = table_df["mean_final_value"].round(2)
    if "sharpe_delta_vs_low" in table_df.columns:
        table_df["sharpe_delta_vs_low"] = table_df["sharpe_delta_vs_low"].round(4)
    if "final_value_delta_vs_low" in table_df.columns:
        table_df["final_value_delta_vs_low"] = table_df["final_value_delta_vs_low"].round(2)
    try:
        lines.append(table_df.to_markdown(index=False))
    except ImportError:
        # Fallback when optional tabulate dependency is missing.
        lines.append(table_df.to_string(index=False))
    lines.append("")

    for symbol in sorted(df["symbol"].unique()):
        subset = df[df["symbol"] == symbol]
        if subset.empty:
            continue
        best_sharpe = subset.loc[subset["mean_sharpe"].idxmax()]
        worst_sharpe = subset.loc[subset["mean_sharpe"].idxmin()]
        best_fv = subset.loc[subset["mean_final_value"].idxmax()]
        worst_fv = subset.loc[subset["mean_final_value"].idxmin()]

        lines.append(f"## {symbol}\n")
        lines.append(
            f"- Best Sharpe: {best_sharpe['mean_sharpe']:.4f} ({best_sharpe['regime']} | {best_sharpe['cost_tier']}, seeds={best_sharpe['n_seeds']}, window={best_sharpe['window']})"
        )
        lines.append(
            f"- Worst Sharpe: {worst_sharpe['mean_sharpe']:.4f} ({worst_sharpe['regime']} | {worst_sharpe['cost_tier']}, seeds={worst_sharpe['n_seeds']}, window={worst_sharpe['window']})"
        )
        lines.append(
            f"- Best Final Value: {best_fv['mean_final_value']:.2f} ({best_fv['regime']} | {best_fv['cost_tier']}, seeds={best_fv['n_seeds']}, window={best_fv['window']})"
        )
        lines.append(
            f"- Worst Final Value: {worst_fv['mean_final_value']:.2f} ({worst_fv['regime']} | {worst_fv['cost_tier']}, seeds={worst_fv['n_seeds']}, window={worst_fv['window']})\n"
        )

    if dispersion_df is not None and not dispersion_df.empty:
        disp = dispersion_df.copy()
        disp["sharpe_std"] = disp["sharpe_std"].round(4)
        disp["sharpe_iqr"] = disp["sharpe_iqr"].round(4)
        disp["final_value_std"] = disp["final_value_std"].round(2)
        disp["final_value_iqr"] = disp["final_value_iqr"].round(2)
        lines.append("## Seed dispersion (std / IQR)\n")
        try:
            lines.append(disp.to_markdown(index=False))
        except ImportError:
            lines.append(disp.to_string(index=False))
        lines.append("")

        # Deterministic vs stochastic comparison for base/high tiers
        pairs = [
            ("base", "heavy_w480_s10", "heavy_base_stoch_w480_s10"),
            ("high", "heavy_high_w480_s10", "heavy_high_stoch_w480_s10"),
        ]
        comp_rows: List[dict] = []
        for symbol in sorted(df["symbol"].unique()):
            for cost_tier, det_reg, stoch_reg in pairs:
                det_row = df[(df["symbol"] == symbol) & (df["regime"] == det_reg) & (df["cost_tier"] == cost_tier)]
                stoch_row = df[(df["symbol"] == symbol) & (df["regime"] == stoch_reg) & (df["cost_tier"] == cost_tier)]
                if det_row.empty and stoch_row.empty:
                    continue
                det = det_row.iloc[0] if not det_row.empty else None
                stoch = stoch_row.iloc[0] if not stoch_row.empty else None
                disp_row = dispersion_df[
                    (dispersion_df["symbol"] == symbol)
                    & (dispersion_df["regime"] == stoch_reg)
                    & (dispersion_df["cost_tier"] == cost_tier)
                ]
                comp_rows.append(
                    {
                        "symbol": symbol,
                        "cost_tier": cost_tier,
                        "det_regime": det_reg,
                        "stoch_regime": stoch_reg,
                        "det_mean_sharpe": det["mean_sharpe"] if det is not None else None,
                        "stoch_mean_sharpe": stoch["mean_sharpe"] if stoch is not None else None,
                        "delta_sharpe": (stoch["mean_sharpe"] - det["mean_sharpe"]) if det is not None and stoch is not None else None,
                        "det_mean_fv": det["mean_final_value"] if det is not None else None,
                        "stoch_mean_fv": stoch["mean_final_value"] if stoch is not None else None,
                        "delta_fv": (stoch["mean_final_value"] - det["mean_final_value"]) if det is not None and stoch is not None else None,
                        "stoch_sharpe_std": disp_row.iloc[0]["sharpe_std"] if not disp_row.empty else None,
                        "stoch_fv_std": disp_row.iloc[0]["final_value_std"] if not disp_row.empty else None,
                    }
                )
        comp_df = pd.DataFrame(comp_rows)
        if not comp_df.empty:
            for col in ["det_mean_sharpe", "stoch_mean_sharpe", "delta_sharpe", "stoch_sharpe_std"]:
                comp_df[col] = comp_df[col].round(4)
            for col in ["det_mean_fv", "stoch_mean_fv", "delta_fv", "stoch_fv_std"]:
                comp_df[col] = comp_df[col].round(2)
            lines.append("## Deterministic vs stochastic (base/high)\n")
            try:
                lines.append(comp_df.to_markdown(index=False))
            except ImportError:
                lines.append(comp_df.to_string(index=False))
            lines.append("")
            lines.append("Bar charts: see det_vs_stoch_delta_sharpe.png and det_vs_stoch_delta_fv.png (both included in cost_tier_report_bundle.pdf).\n")
            # Save comparison to CSV for plotting reuse
            comp_df.to_csv(AGG_DIR / "det_vs_stoch_comparison.csv", index=False)
    return "\n".join(lines)


def main() -> None:
    runs: List[RunInfo] = []
    runs.extend(collect_full_runs())
    runs.extend(collect_stress_runs())
    runs.extend(collect_stress_high_runs())
    runs.extend(collect_heavy_runs())
    runs.extend(collect_heavy_high_runs())
    runs.extend(collect_heavy_base_stoch_runs())
    runs.extend(collect_heavy_high_stoch_runs())

    df = as_dataframe(runs)
    df = add_baseline_deltas(df)
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    combined_csv = AGG_DIR / "cost_tier_combined.csv"
    df.to_csv(combined_csv, index=False)

    if not df.empty:
        plot_metric(df, "mean_sharpe", "Mean Sharpe", AGG_DIR / "cost_tier_sharpe_by_symbol_regime.png")
        plot_metric(df, "mean_final_value", "Mean Final Value", AGG_DIR / "cost_tier_final_value_by_symbol_regime.png")
        plot_metric(df, "sharpe_delta_vs_low", "Sharpe Δ vs low", AGG_DIR / "cost_tier_sharpe_delta_by_symbol_regime.png")
        plot_metric(df, "final_value_delta_vs_low", "Final Value Δ vs low", AGG_DIR / "cost_tier_final_value_delta_by_symbol_regime.png")
        seed_rows: List[dict] = []
        for run in runs:
            for seed_row in load_seed_metrics(run.results_path):
                seed_rows.append(
                    {
                        "symbol": run.symbol,
                        "cost_tier": run.cost_tier,
                        "regime": run.regime,
                        "seed": seed_row.get("seed"),
                        "mean_sharpe": seed_row.get("mean_sharpe"),
                        "mean_final_value": seed_row.get("mean_final_value"),
                    }
                )
        seed_df = pd.DataFrame(seed_rows)
        dispersion_df: Optional[pd.DataFrame] = None
        if not seed_df.empty:
            seed_df["cost_tier"] = pd.Categorical(seed_df["cost_tier"], categories=COST_ORDER, ordered=True)
            seed_df["regime"] = pd.Categorical(seed_df["regime"], categories=[r for r in REGIME_ORDER if r in seed_df["regime"].unique()], ordered=True)
            seed_df.to_csv(AGG_DIR / "cost_tier_seeds.csv", index=False)
            plot_seed_boxplots(seed_df, "mean_sharpe", "Sharpe (per seed)", AGG_DIR / "cost_tier_sharpe_box_by_symbol.png")
            plot_seed_boxplots(seed_df, "mean_final_value", "Final Value (per seed)", AGG_DIR / "cost_tier_final_value_box_by_symbol.png")
            dispersion_df = (
                seed_df.groupby(["symbol", "regime", "cost_tier"], observed=True)
                .agg(
                    seed_count=("seed", "count"),
                    sharpe_std=("mean_sharpe", "std"),
                    sharpe_iqr=("mean_sharpe", lambda s: s.quantile(0.75) - s.quantile(0.25)),
                    final_value_std=("mean_final_value", "std"),
                    final_value_iqr=("mean_final_value", lambda s: s.quantile(0.75) - s.quantile(0.25)),
                )
                .reset_index()
            )
            dispersion_df.to_csv(AGG_DIR / "cost_tier_seed_dispersion.csv", index=False)
        report_path = AGG_DIR / "cost_tier_report.md"
        report_text = build_report(df, dispersion_df)
        report_path.write_text(report_text, encoding="utf-8")

        # Deterministic vs stochastic comparison plots
        comp_csv = AGG_DIR / "det_vs_stoch_comparison.csv"
        if comp_csv.exists():
            comp_df = pd.read_csv(comp_csv)
            plot_det_vs_stoch_bars(comp_df, AGG_DIR / "det_vs_stoch_delta_sharpe.png", "delta_sharpe", "Δ Sharpe: stochastic - deterministic", "Δ Sharpe")
            plot_det_vs_stoch_bars(comp_df, AGG_DIR / "det_vs_stoch_delta_fv.png", "delta_fv", "Δ Final Value: stochastic - deterministic", "Δ Final Value")

        pdf_images = [
            AGG_DIR / "cost_tier_sharpe_by_symbol_regime.png",
            AGG_DIR / "cost_tier_final_value_by_symbol_regime.png",
            AGG_DIR / "cost_tier_sharpe_delta_by_symbol_regime.png",
            AGG_DIR / "cost_tier_final_value_delta_by_symbol_regime.png",
            AGG_DIR / "cost_tier_sharpe_box_by_symbol.png",
            AGG_DIR / "cost_tier_final_value_box_by_symbol.png",
            AGG_DIR / "det_vs_stoch_delta_sharpe.png",
            AGG_DIR / "det_vs_stoch_delta_fv.png",
        ]
        export_pdf_bundle(report_path, pdf_images, AGG_DIR / "cost_tier_report_bundle.pdf")
        print(f"Wrote {combined_csv}, plots, and report to {AGG_DIR}")
    else:
        print("No runs found; nothing to plot.")


if __name__ == "__main__":
    main()
