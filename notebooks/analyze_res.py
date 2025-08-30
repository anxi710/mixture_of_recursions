#!/usr/bin/env python3
"""
Analyze lm-evaluation-harness results JSON file.

Usage:
    python analyze_results.py results.json
"""

import json
import sys
import pandas as pd
import matplotlib.pyplot as plt


def load_results(path: str):
    """Load lm-evaluation-harness results from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def build_dataframe(results: dict) -> pd.DataFrame:
    """Convert results dict to pandas DataFrame."""
    rows = []
    for task, metrics in results.items():
        rows.append({
            "Task": task,
            "Accuracy": metrics.get("acc,none"),
            "Acc_StdErr": metrics.get("acc_stderr,none"),
            "Norm_Acc": metrics.get("acc_norm,none"),
            "Norm_Acc_StdErr": metrics.get("acc_norm_stderr,none"),
        })
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_res.py results.json")
        sys.exit(1)

    json_path = sys.argv[1]
    data = load_results(json_path)
    results = data["results"]

    # Build DataFrame
    df = build_dataframe(results)
    print("\n=== Evaluation Results Summary ===")
    print(df.to_string(index=False))

    # Compute overall averages
    avg_acc = df["Accuracy"].mean()
    avg_norm_acc = df["Norm_Acc"].mean()
    print(f"\nAverage Accuracy: {avg_acc:.4f}")
    print(f"Average Normalized Accuracy: {avg_norm_acc:.4f}")

    # Plot Accuracy with error bars
    plt.figure(figsize=(10, 6))
    plt.barh(df["Task"], df["Accuracy"], 
             xerr=df["Acc_StdErr"], capsize=5, color="skyblue")
    plt.xlabel("Accuracy")
    plt.title("LM Evaluation Harness Results (Accuracy with StdErr)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
