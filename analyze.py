import json
import glob
import os
import pandas as pd

BASELINE_TOKENS = 242.384
BASELINE_PASS = 0.585


def load_results(results_dir: str = "results") -> pd.DataFrame:
    rows = []
    for fpath in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        with open(fpath) as f:
            for line in f:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, k: int = None) -> pd.DataFrame:
    rows = []
    for method, group in df.groupby("method"):
        if k is not None and "tokens_per_attempt" in group.columns and group["tokens_per_attempt"].notna().any():
            tokens = group["tokens_per_attempt"].apply(
                lambda x: sum(x[:k]) if isinstance(x, list) else x
            )
            passed = group["passed_per_attempt"].apply(
                lambda x: any(x[:k]) if isinstance(x, list) else x
            )
        else:
            tokens = group["total_tokens"]
            passed = group["passed"]

        pass_rate = passed.mean()
        avg_tokens = tokens.mean()
        delta_acc = (pass_rate - BASELINE_PASS) * 100
        delta_tok = avg_tokens - BASELINE_TOKENS
        efficiency = delta_acc / delta_tok if delta_tok > 0 else 0.0

        rows.append({
            "method": method,
            "n": len(group),
            "pass_rate": round(pass_rate, 3),
            "avg_tokens": round(avg_tokens, 1),
            "delta_acc(%)": round(delta_acc, 1),
            "efficiency(Δacc/Δtok)": round(efficiency * 100, 3),
            "avg_attempts": round(group["attempts"].mean(), 2),
        })

    result = pd.DataFrame(rows).set_index("method")
    return result.sort_values("pass_rate", ascending=False)


def attempt_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in ["self_repair", "trial_first"]:
        sub = df[df["method"] == method]
        if "passed_per_attempt" not in sub.columns:
            continue
        total = len(sub)
        for k in [1, 2, 3]:
            passed = sub["passed_per_attempt"].apply(
                lambda x: any(x[:k]) if isinstance(x, list) else (x if k == 1 else x)
            ).sum()
            tokens = sub["tokens_per_attempt"].apply(
                lambda x: sum(x[:k]) if isinstance(x, list) else x
            ).mean()
            rows.append({
                "method": method,
                "k": k,
                "pass_rate": round(passed / total, 3),
                "avg_tokens": round(tokens, 1),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_results()
    if df.empty:
        print("No results found.")
    else:
        print("=== Summary (k=2) ===")
        print(summarize(df, k=2).to_string())
        print("\n=== Summary (k=3, full) ===")
        print(summarize(df, k=3).to_string())
        print("\n=== Attempt Breakdown ===")
        print(attempt_breakdown(df).to_string(index=False))
