import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

METHOD_ORDER = ["baseline", "cot", "self_repair", "trial_first"]
METHOD_LABELS = {
    "baseline": "Baseline",
    "cot": "CoT",
    "self_repair": "Self-Repair",
    "trial_first": "Trial-First\n(Ours)",
}
COLORS = {
    "baseline": "#aec6e8",
    "cot": "#ffb347",
    "self_repair": "#c5e8b0",
    "trial_first": "#4a90d9",
}


def load_results(results_dir: str = "results") -> pd.DataFrame:
    rows = []
    for fpath in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        with open(fpath) as f:
            for line in f:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarize_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []
    for method, group in df.groupby("method"):
        has_breakdown = (
            "tokens_per_attempt" in group.columns
            and group["tokens_per_attempt"].dropna().apply(lambda x: isinstance(x, list)).any()
        )
        if has_breakdown:
            tokens = group["tokens_per_attempt"].apply(lambda x: sum(x[:k]) if isinstance(x, list) else x)
            passed = group["passed_per_attempt"].apply(lambda x: any(x[:k]) if isinstance(x, list) else x)
        else:
            tokens = group["total_tokens"]
            passed = group["passed"]
        rows.append({
            "method": method,
            "pass_rate": passed.mean(),
            "avg_tokens": tokens.mean(),
        })
    return pd.DataFrame(rows).set_index("method")


def attempt_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ["self_repair", "trial_first"]:
        sub = df[df["method"] == method]
        if "passed_per_attempt" not in sub.columns:
            continue
        total = len(sub)
        for k in [1, 2, 3]:
            passed = sub["passed_per_attempt"].apply(
                lambda x: any(x[:k]) if isinstance(x, list) else x
            ).sum()
            rows.append({"method": method, "k": k, "pass_rate": passed / total})
    return pd.DataFrame(rows)


# Figure 1: pass@1 bar chart (k=2)
def plot_pass_rate(df: pd.DataFrame, out: str = "fig1_pass_rate.pdf"):
    summary = summarize_k(df, k=2)
    methods = [m for m in METHOD_ORDER if m in summary.index]
    values = [summary.loc[m, "pass_rate"] * 100 for m in methods]
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("pass@1 (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("pass@1 on HumanEval (qwen2.5-coder 1.5B, k=2)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


# Figure 2: attempt별 누적 pass rate 꺾은선
def plot_attempt_curve(df: pd.DataFrame, out: str = "fig2_attempt_curve.pdf"):
    bd = attempt_breakdown(df)
    if bd.empty:
        print("No attempt breakdown data yet.")
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    for method, style in [("self_repair", "--"), ("trial_first", "-")]:
        sub = bd[bd["method"] == method].sort_values("k")
        ax.plot(sub["k"], sub["pass_rate"] * 100,
                style, marker="o", color=COLORS[method],
                label=METHOD_LABELS[method].replace("\n", " "), linewidth=2, markersize=7)

    ax.set_xlabel("Number of Attempts (k)", fontsize=11)
    ax.set_ylabel("pass@1 (%)", fontsize=11)
    ax.set_title("Cumulative pass@1 by Attempt", fontsize=11)
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(50, 80)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


# Figure 3: token efficiency bar chart (Δacc / ΔToken)
def plot_cost_accuracy(df: pd.DataFrame, out: str = "fig3_cost_accuracy.pdf"):
    summary = summarize_k(df, k=2)
    baseline_pass = summary.loc["baseline", "pass_rate"]
    baseline_tok = summary.loc["baseline", "avg_tokens"]

    methods = [m for m in METHOD_ORDER if m != "baseline" and m in summary.index]
    labels = [METHOD_LABELS[m] for m in methods]
    colors = [COLORS[m] for m in methods]
    efficiencies = []
    for m in methods:
        delta_acc = (summary.loc[m, "pass_rate"] - baseline_pass) * 100
        delta_tok = summary.loc[m, "avg_tokens"] - baseline_tok
        efficiencies.append(delta_acc / delta_tok * 100 if delta_tok > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, efficiencies, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, efficiencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Δacc (%) / ΔToken × 100", fontsize=11)
    ax.set_title("Token Efficiency (k=2)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


# Figure 0: Trial-First 흐름도
def plot_flowchart(out: str = "fig0_method.pdf"):
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0, 14)
    ax.axis("off")

    def box(cx, cy, w, h, text, fc="#ddeeff"):
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.15",
            facecolor=fc, edgecolor="#333333", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=9.5)

    def diamond(cx, cy, w, h, text, fc="#fff3cd"):
        pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
        ax.add_patch(plt.Polygon(pts, facecolor=fc, edgecolor="#333333", linewidth=1.2))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=9.5)

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5))

    def lbl(x, y, text, fs=8.5, color="#555555"):
        ax.text(x, y, text, fontsize=fs, color=color, ha="center", va="center")

    # 노드
    box(5, 13.0, 3.5, 0.85, "Problem P", fc="#f0f0f0")
    arr(5, 12.57, 5, 12.0)
    box(5, 11.55, 3.5, 0.85, "Code Generation", fc="#aec6e8")
    arr(5, 11.12, 5, 10.5)
    box(5, 10.05, 3.5, 0.85, "Execute Code", fc="#ddeeff")
    arr(5, 9.62, 5, 8.95)
    diamond(5, 8.25, 3.2, 1.2, "Pass?", fc="#fff3cd")

    # Pass → Return
    arr(6.6, 8.25, 8.0, 8.25)
    lbl(7.3, 8.55, "Yes")
    box(9.0, 8.25, 2.5, 0.85, "Return", fc="#c5e8b0")

    # Fail → 분석
    arr(5, 7.65, 5, 7.0)
    lbl(5.45, 7.3, "No")
    box(5, 6.5, 4.0, 0.85, "Failure Analysis\n→ Checklist", fc="#ffb347")
    arr(5, 6.07, 5, 5.45)
    box(5, 5.0, 4.0, 0.85, "Reset Context", fc="#f8d7da")
    arr(5, 4.57, 5, 3.95)
    box(5, 3.5, 4.0, 0.85, "Retry: P + Checklist", fc="#aec6e8")

    # 루프 (왼쪽 점선)
    ax.plot([3.0, 1.8, 1.8, 3.25], [3.5, 3.5, 11.55, 11.55],
            color="#888888", lw=1.4, linestyle="--")
    arr(1.8, 11.55, 3.25, 11.55)
    lbl(1.05, 7.5, "k < max", fs=8, color="#888888")

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    df = load_results()
    if df.empty:
        print("No results found.")
    else:
        plot_pass_rate(df)
        plot_cost_accuracy(df)
        plot_attempt_curve(df)
        print("Done.")
