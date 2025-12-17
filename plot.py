import json

import pandas as pd
import matplotlib.pyplot as plt


def read_jsonl(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f.readlines()]

    df = pd.DataFrame(data)

    return df


def plot_series(x, y, title, xlabel="step", smooth_window=0, save_prefix="."):
    y = pd.to_numeric(y, errors="coerce")
    if smooth_window and smooth_window > 1:
        y_plot = y.rolling(smooth_window, min_periods=1).mean()
    else:
        y_plot = y

    plt.figure()
    plt.plot(x, y_plot)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}/{title}.png")


def main(jsonl_path: str, smooth_window: int = 0):
    df = read_jsonl(jsonl_path)
    train_df = df[df.name == "training_loss"].copy()
    eval_df = df[df.name == "validation_loss"].copy()

    metrics = [
        "loss",
        "learning_rate",
        "grad_norm",
        "mean_token_accuracy",
        "time_per_steps",
        "train_tokens_per_second",
        "peak_mem_alloc_gib",
        "peak_mem_reserved_gib",
    ]
    steps = train_df["data"].apply(lambda x: x.get("step"))
    for col in metrics:
        plot_series(steps, train_df[col], col, smooth_window=smooth_window, save_prefix="./tmp/bf16/")


    metrics = [
        "eval_loss",
        "eval_mean_token_accuracy"
    ]

    steps = eval_df["data"].apply(lambda x: x.get("step"))
    for col in metrics:
        plot_series(steps, eval_df[col], col, smooth_window=smooth_window, save_prefix="./tmp/bf16/")


if __name__ == "__main__":
    JSONL_PATH = "/workspace/fms-hf-tuning/tmp/bf16/training_logs.jsonl"
    SMOOTH_WINDOW = 20

    main(JSONL_PATH, smooth_window=SMOOTH_WINDOW)
