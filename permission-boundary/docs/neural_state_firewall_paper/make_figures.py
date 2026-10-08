#!/usr/bin/env python3
"""Make evidence plots from sanitized case and CUSUM CSVs."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent

def configure_plot_cache():
    cache_root = Path(tempfile.gettempdir()) / "neural-state-firewall-plot-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    fontconfig = cache_root / "fonts.conf"
    if "FONTCONFIG_FILE" not in os.environ:
        import importlib.util
        mpl_spec = importlib.util.find_spec("matplotlib")
        if mpl_spec and mpl_spec.submodule_search_locations:
            mpl_root = Path(next(iter(mpl_spec.submodule_search_locations)))
            font_dir = mpl_root / "mpl-data" / "fonts" / "ttf"
            cache_root.mkdir(parents=True, exist_ok=True)
            fontconfig.write_text(
                '<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd">'
                f'<fontconfig><dir>{font_dir}</dir><cachedir>{cache_root / "fontconfig"}</cachedir></fontconfig>'
            )
            os.environ["FONTCONFIG_FILE"] = str(fontconfig)


configure_plot_cache()
import matplotlib.pyplot as plt


def export_data(results_path):
    raw = json.loads(Path(results_path).read_text())
    cases_path = ROOT / "data" / "paired_smoke_cases.csv"
    trace_path = ROOT / "data" / "cusum_trajectories.csv"
    with cases_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_alias", "manifest_condition", "guard_status", "alarm", "guarded_steps"])
        for i, row in enumerate(raw["rows"], 1):
            w.writerow([f"{'B' if row['condition'] == 'benign' else 'I'}{sum(r['condition'] == row['condition'] for r in raw['rows'][:i])}",
                        row["condition"], row["guarded"]["status"], row["guarded"]["alarm_observed"],
                        row["guarded"]["observed_steps"]])
    with trace_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_alias", "manifest_condition", "step", "cusum", "threshold", "alarm"])
        for i, row in enumerate(raw["rows"], 1):
            alias = f"{'B' if row['condition'] == 'benign' else 'I'}{sum(r['condition'] == row['condition'] for r in raw['rows'][:i])}"
            for t in row["guarded"]["telemetry"]:
                w.writerow([alias, row["condition"], t["step"], f"{t['cusum']:.9f}", f"{t['threshold']:.9f}", t["alarm"]])


def rows(name):
    with (ROOT / "data" / name).open(newline="") as f:
        return list(csv.DictReader(f))


def make_figures():
    cases = rows("paired_smoke_cases.csv")
    traces = rows("cusum_trajectories.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.2), sharey=True, constrained_layout=True)
    for ax, case in zip(axes.flat, cases):
        series = [r for r in traces if r["case_alias"] == case["case_alias"]]
        color = "#b24c63" if case["manifest_condition"] == "injection" else "#3b738e"
        x = [int(r["step"]) for r in series]
        y = [float(r["cusum"]) for r in series]
        threshold = float(series[0]["threshold"])
        ax.plot(x, y, marker="o", markersize=2.8, linewidth=1.5, color=color)
        ax.axhline(threshold, linestyle="--", linewidth=1, color="#333333", label="calibration threshold")
        ax.set_xlim(0.5, max(x) + 0.5)
        ax.set_xticks(range(1, max(x) + 1, max(1, max(x) // 6)))
        alarms = [r for r in series if r["alarm"] == "True"]
        if alarms:
            ax.scatter([int(r["step"]) for r in alarms], [float(r["cusum"]) for r in alarms],
                       s=34, marker="x", linewidth=1.8, color="#111111", label="alarm")
        ax.set_title(f"{case['case_alias']} · {case['manifest_condition']} · {case['guard_status']}", fontsize=9)
        ax.set_xlabel("Observed generation step")
        ax.grid(alpha=.22)
    axes[0, 0].set_ylabel("CUSUM statistic")
    axes[1, 0].set_ylabel("CUSUM statistic")
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Observed monitor trajectories in the six-case paired smoke run\n(manifest classes are author-assigned; not human-adjudicated outcomes)", fontsize=11)
    fig.savefig(ROOT / "figures" / "cusum_trajectories.png", dpi=200)
    plt.close(fig)

    counts = {condition: {"allowed": 0, "blocked": 0} for condition in ("benign", "injection")}
    for case in cases:
        counts[case["manifest_condition"]][case["guard_status"]] += 1
    fig, ax = plt.subplots(figsize=(6.7, 3.6), constrained_layout=True)
    labels = ["Manifest benign\n(n=3)", "Manifest injection\n(n=3)"]
    allowed = [counts["benign"]["allowed"], counts["injection"]["allowed"]]
    blocked = [counts["benign"]["blocked"], counts["injection"]["blocked"]]
    x = range(2)
    ax.bar(x, allowed, color="#6c9e84", label="Allowed")
    ax.bar(x, blocked, bottom=allowed, color="#bd6573", label="Blocked")
    for j in x:
        ax.text(j, allowed[j] / 2, str(allowed[j]), ha="center", va="center", color="white", weight="bold")
        if blocked[j]:
            ax.text(j, allowed[j] + blocked[j] / 2, str(blocked[j]), ha="center", va="center", color="white", weight="bold")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Cases")
    ax.set_ylim(0, 3.5)
    ax.legend(frameon=False, ncols=2, loc="upper center")
    ax.set_title("Raw guarded dispositions; no output labels applied")
    ax.text(.5, -.26, "These proportions are not attack-success or false-block estimates.", ha="center", transform=ax.transAxes, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(ROOT / "figures" / "guard_dispositions.png", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", help="private raw paired-results JSON")
    p.add_argument("--export-data", action="store_true", help="export sanitized CSVs from --results")
    args = p.parse_args()
    if args.export_data:
        if not args.results:
            p.error("--export-data requires --results")
        export_data(args.results)
    make_figures()


if __name__ == "__main__":
    main()
