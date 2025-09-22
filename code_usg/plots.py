import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_corr_heatmap(df: pd.DataFrame, out_png: str, title: str = "Correlation Matrix") -> None:
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_timeseries(df: pd.DataFrame, column: str, out_png: str, title: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[column], label=column)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_multi_timeseries(df: pd.DataFrame, columns: list[str], out_png: str, title: str) -> None:
    plt.figure(figsize=(12, 6))
    for c in columns:
        if c in df.columns:
            plt.plot(df.index, df[c], label=c)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
