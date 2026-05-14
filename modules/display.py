"""
Display module for Steam Review Sentiment Analysis.
Contains functions for displaying results and metrics.
"""

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.rule import Rule
from rich.rule import Text
from rich.panel import Panel
from rich.columns import Columns

from rich import box

console = Console()

def print_header():
    """Print the program header."""
    console.print()
    console.print(Rule(style="bright_blue"))
    console.print(
        "[bold bright_blue]  Steam Review Sentiment & Keyword Analysis System[/]"
    )
    console.print("[dim]  CS 4210 Final Project | Natasha Wong[/]")
    console.print(Rule(style="bright_blue"))

def print_metrics(metrics: dict):
    """Print evaluation metrics for the model."""
    console.print()
    console.print(Rule("[bold]Model Evaluation Results[/]", style="cyan"))

    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan",
                title="Test Set Metrics (Logistic Regression)", title_style="bold")
    tbl.add_column("Metric",  style="bold", width=20)
    tbl.add_column("Score",   justify="right", width=10)
    tbl.add_column("Bar",     width=30)

    rows = [
        ("Accuracy",          metrics['accuracy']),
        ("Precision (pos)",   metrics['precision']),
        ("Recall (pos)",      metrics['recall']),
        ("F1 (pos)",          metrics['f1']),
        ("Neg Precision",     metrics['neg_precision']),
        ("Neg Recall",        metrics['neg_recall']),
        ("Neg F1",            metrics['neg_f1']),
    ]
    for label, val in rows:
        filled = int(val * 25)
        bar = "[green]" + "█" * filled + "[/][dim]" + "░" * (25 - filled) + "[/]"
        tbl.add_row(label, f"{val:.4f}", bar)

    console.print(tbl)

    cm = metrics['cm']
    console.print()
    cm_tbl = Table(box=box.SIMPLE_HEAVY, title="Confusion Matrix",
                   title_style="bold", show_header=True,
                   header_style="bold magenta")
    cm_tbl.add_column("",             style="bold magenta", width=16)
    cm_tbl.add_column("Pred Neg",     justify="center", width=12)
    cm_tbl.add_column("Pred Pos",     justify="center", width=12)
    cm_tbl.add_row("True Negative",   str(cm[0][0]), str(cm[0][1]))
    cm_tbl.add_row("True Positive",   str(cm[1][0]), str(cm[1][1]))
    console.print(cm_tbl)

    if 'best_params' in metrics:
        bp = metrics['best_params']
        console.print(
            f"\n[dim]Best hyperparameters (grid-search): "
            f"max_features={bp['tfidf__max_features']}, "
            f"C={bp['lr__C']}, "
            f"CV f1_macro={metrics['cv_score']:.4f}[/]"
        )

def global_keywords(model, vectorizer, n: int = 10):
    names = np.array(vectorizer.get_feature_names_out())
    coef  = model.coef_[0]
    pos   = list(zip(names[np.argsort(coef)[-n:][::-1]],
                     coef[np.argsort(coef)[-n:][::-1]]))
    neg   = list(zip(names[np.argsort(coef)[:n]],
                     coef[np.argsort(coef)[:n]]))
    return pos, neg

def print_global_keywords(model, vectorizer):
    pos_kw, neg_kw = global_keywords(model, vectorizer, n=10)
    console.print()
    console.print(Rule(
        "[bold]Top Global Keywords (Logistic Regression Coefficients)[/]",
        style="cyan"
    ))

    pos_tbl = Table(box=box.ROUNDED, title="[green]Positive Keywords[/]",
                    title_style="bold green", show_header=True,
                    header_style="bold green")
    pos_tbl.add_column("Keyword", width=22)
    pos_tbl.add_column("Weight",  justify="right", width=10)

    neg_tbl = Table(box=box.ROUNDED, title="[red]Negative Keywords[/]",
                    title_style="bold red", show_header=True,
                    header_style="bold red")
    neg_tbl.add_column("Keyword", width=22)
    neg_tbl.add_column("Weight",  justify="right", width=10)

    for w, s in pos_kw:
        pos_tbl.add_row(w, f"[green]+{s:.4f}[/]")
    for w, s in neg_kw:
        neg_tbl.add_row(w, f"[red]{s:.4f}[/]")

    console.print(Columns([pos_tbl, neg_tbl]))

def print_nb_comparison(metrics: dict, nb_metrics: dict):
    console.print()
    console.print(Rule(
        "[bold]Logistic Regression vs. Naive Bayes Baseline[/]",
        style="cyan"
    ))

    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan",
                title="Model Comparison (same TF-IDF features)", title_style="bold")
    tbl.add_column("Metric",       style="bold", width=20)
    tbl.add_column("Logistic Reg", justify="right", width=14)
    tbl.add_column("Naive Bayes",  justify="right", width=14)
    tbl.add_column("LR - NB",     justify="right", width=14)

    rows = [
        ("Accuracy",        metrics['accuracy'],      nb_metrics['accuracy']),
        ("Precision (pos)", metrics['precision'],     nb_metrics['precision']),
        ("Recall (pos)",    metrics['recall'],        nb_metrics['recall']),
        ("F1 (pos)",        metrics['f1'],            nb_metrics['f1']),
        ("Neg Precision",   metrics['neg_precision'], nb_metrics['neg_precision']),
        ("Neg Recall",      metrics['neg_recall'],    nb_metrics['neg_recall']),
        ("Neg F1",          metrics['neg_f1'],        nb_metrics['neg_f1']),
    ]
    for label, lr_v, nb_v in rows:
        delta = lr_v - nb_v
        col   = "green" if delta > 0 else "red" if delta < 0 else "dim"
        sign  = "+" if delta > 0 else ""
        tbl.add_row(
            label,
            f"{lr_v:.4f}",
            f"{nb_v:.4f}",
            f"[{col}]{sign}{delta:.4f}[/{col}]",
        )
    console.print(tbl)

def print_prediction(review_text: str, sentiment: str,
                     confidence: float, keywords):
    console.print()
    display = (review_text if len(review_text) <= 120
               else review_text[:117] + "...")
    console.print(Panel(
        f"[dim]{display}[/]",
        title="[bold]Review[/]", border_style="dim"
    ))

    color  = "green" if sentiment == "Positive" else "red"
    filled = int(confidence * 20)
    bar    = (f"[{color}]" + "█" * filled + "[/]"
              + "[dim]" + "░" * (20 - filled) + "[/]")

    result = Text()
    result.append("  Sentiment : ", style="bold")
    result.append(f"{sentiment}", style=f"bold {color}")
    result.append(f"  ({confidence * 100:.1f}%)\n")
    result.append(f"  Confidence: {bar}\n")

    if keywords:
        result.append("  Keywords  : ", style="bold")
        kw_parts = []
        for w, s in keywords:
            c = "green" if s > 0 else "red"
            kw_parts.append(f"[{c}]{w}[/{c}]")
        result.append("  ".join(kw_parts) + "\n")

    console.print(Panel(result, title="[bold cyan]Prediction[/]",
                        border_style="cyan"))
    
def print_batch_results(results: list):
    console.print()
    console.print(Rule("[bold]Batch Results[/]", style="cyan"))
    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    tbl.add_column("#",           width=4,  justify="right")
    tbl.add_column("Review",      width=50)
    tbl.add_column("Sentiment",   width=12, justify="center")
    tbl.add_column("Confidence",  width=12, justify="right")
    tbl.add_column("Top Keyword", width=20)

    for i, r in enumerate(results, 1):
        color = "green" if r['sentiment'] == 'Positive' else "red"
        review_short = (r['review'][:48] + ".."
                        if len(r['review']) > 48 else r['review'])
        tbl.add_row(
            str(i),
            review_short,
            f"[{color}]{r['sentiment']}[/{color}]",
            f"{r['confidence'] * 100:.1f}%",
            r['top_kw'],
        )
    console.print(tbl)