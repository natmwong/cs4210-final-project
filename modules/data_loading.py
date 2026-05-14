"""
Data loading and sampling module for Steam Review Sentiment Analysis.
Contains functions for loading CSV files and creating balanced samples.
"""

import os
import sys
import glob
import pandas as pd
from rich.console import Console

console = Console()

def _raw_from_csv(path: str, cap=None) -> pd.DataFrame:
    """Load raw data from a CSV file and preprocess labels."""
    df = pd.read_csv(path, low_memory=False)
    if 'language' in df.columns:
        df = df[df['language'] == 'english'].copy()
    df = df.dropna(subset=['review', 'voted_up'])
    df = df[df['review'].str.strip() != ''].copy()
    df['label'] = df['voted_up'].astype(bool).astype(int)
    if cap and len(df) > cap:
        n_pos = int(cap * df['label'].mean())
        n_neg = cap - n_pos
        pos_df = df[df['label'] == 1]
        neg_df = df[df['label'] == 0]
        n_pos = min(len(pos_df), n_pos)
        n_neg = min(len(neg_df), n_neg)
        sampled_pos = pos_df.sample(n=n_pos, random_state=42) if n_pos > 0 else pos_df
        sampled_neg = neg_df.sample(n=n_neg, random_state=42) if n_neg > 0 else neg_df
        df = pd.concat([sampled_pos, sampled_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df[['review', 'label']].reset_index(drop=True)

def _show_counts(df: pd.DataFrame, label: str = "Dataset") -> None:
    """Display counts of positive and negative labels."""
    pos = int(df.label.sum())
    neg = int((df.label == 0).sum())
    ratio = neg / max(pos, 1)
    console.print(
        f"\n  [bold]{label}:[/] "
        f"[green]{pos:,} positive[/]  "
        f"[red]{neg:,} negative[/]  "
        f"[dim]{len(df):,} total  ({ratio:.1%} negative)[/]"
    )

def load_data(path: str) -> pd.DataFrame:
    """Load data from a file or directory."""
    if os.path.isdir(path):
        return _load_dir(path)
    console.print(f"\n[bold cyan]Loading:[/] {path}")
    df = _raw_from_csv(path)
    _show_counts(df)
    return df

def _load_dir(dir_path: str) -> pd.DataFrame:
    """Load and combine data from all CSV files in a directory."""
    csv_files = sorted(glob.glob(os.path.join(dir_path, '*.csv')))
    if not csv_files:
        console.print(f"[red]No CSV files found in {dir_path}[/]")
        sys.exit(1)

    console.print(
        f"\n[bold cyan]Loading {len(csv_files)} game CSVs from:[/] {dir_path}\n"
    )
    frames = []
    for idx, f in enumerate(csv_files, 1):
        name = os.path.basename(f)
        try:
            sub = _raw_from_csv(f, cap=5000)
            pos = int(sub.label.sum())
            neg = int((sub.label == 0).sum())
            console.print(
                f"  [dim]{idx:3d}/{len(csv_files)}[/] "
                f"{name:<58} "
                f"[green]{pos:5,}[/]+ [red]{neg:4,}[/]-"
            )
            frames.append(sub)
        except Exception as exc:
            console.print(f"  [yellow]Skipping {name}: {exc}[/]")

    df = pd.concat(frames, ignore_index=True)
    _show_counts(df, label="Combined")
    return df

def balanced_sample(df: pd.DataFrame, sample_size: int = 10000, neg_ratio: float = 0.30) -> pd.DataFrame:
    """Create a balanced sample from the dataset."""
    neg_df = df[df.label == 0]
    pos_df = df[df.label == 1]
    ratio = len(neg_df) / max(len(pos_df), 1)

    if ratio < 0.10:
        n_neg = min(len(neg_df), int(sample_size * neg_ratio))
        n_pos = min(len(pos_df), sample_size - n_neg)
        console.print(
            f"  [yellow]Class imbalance detected ({ratio:.2%} negative). "
            f"Stratified sample: {n_pos:,} pos / {n_neg:,} neg[/]"
        )
        df = pd.concat([
            pos_df.sample(n=n_pos, random_state=42),
            neg_df.sample(n=n_neg, random_state=42),
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
    elif len(df) > sample_size:
        n_pos = int(sample_size * df['label'].mean())
        n_neg = sample_size - n_pos
        pos_df = df[df['label'] == 1]
        neg_df = df[df['label'] == 0]
        n_pos = min(len(pos_df), n_pos)
        n_neg = min(len(neg_df), n_neg)
        sampled_pos = pos_df.sample(n=n_pos, random_state=42) if n_pos > 0 else pos_df
        sampled_neg = neg_df.sample(n=n_neg, random_state=42) if n_neg > 0 else neg_df
        df = pd.concat([sampled_pos, sampled_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        console.print(
            f"  [yellow]Large dataset — stratified sample of {sample_size:,}[/]"
        )
    return df