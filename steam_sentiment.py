"""
Main script for Steam Review Sentiment Analysis.
Handles user interaction and integrates modules for preprocessing, data loading,
training, and displaying results.
"""

import sys
from modules.data_loading import load_data, balanced_sample
from modules.training import train
from modules.display import print_header, print_metrics, print_global_keywords, print_nb_comparison, print_prediction, print_batch_results
from modules.preprocessing import preprocess_text, review_keywords
import pandas as pd
from rich.table import Table
from rich.text import Text
from rich import box
from rich.rule import Rule
from rich.console import Console

console = Console()

def action_single(model, vectorizer):
    console.print(
        "\n[bold]Enter a review to analyze[/] "
        "[dim](leave blank to go back)[/]"
    )
    while True:
        console.print()
        review = console.input("[cyan]Review > [/]").strip()
        if not review:
            console.print("[yellow]Exiting single review analysis...[/]")
            break
        sentiment, confidence, keywords = review_keywords(
            review, model, vectorizer
        )
        print_prediction(review, sentiment, confidence, keywords)


def action_batch(df, model, vectorizer):
    console.print(
        "\n[bold]How many random reviews to analyze?[/] [dim](default 10)[/]"
    )
    raw = console.input("[cyan]Count > [/]").strip()
    n   = int(raw) if raw.isdigit() else 10
    n   = min(n, len(df))

    sample  = df.sample(n=n, random_state=None)
    results = []
    for _, row in sample.iterrows():
        sentiment, confidence, keywords = review_keywords(
            row['review'], model, vectorizer
        )
        results.append({
            'review':     row['review'],
            'sentiment':  sentiment,
            'confidence': confidence,
            'top_kw':     keywords[0][0] if keywords else 'N/A',
        })

    print_batch_results(results)

    save = console.input("\n[cyan]Save results to CSV? (y/n) > [/]").strip().lower()
    if save == 'y':
        out = 'steam_reviews_predictions.csv'
        pd.DataFrame(results)[
            ['review', 'sentiment', 'confidence', 'top_kw']
        ].to_csv(out, index=False)
        console.print(f"[green]Saved → {out}[/]")


def main_menu(df, model, vectorizer, metrics, nb_metrics):
    print_header()
    print_metrics(metrics)
    print_global_keywords(model, vectorizer)
    print_nb_comparison(metrics, nb_metrics)

    menu = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    menu.add_column(width=4,  style="bold cyan")
    menu.add_column(width=55)
    menu.add_row("[1]", "Analyze a single review  (type your own text)")
    menu.add_row("[2]", "Batch analyze random reviews from dataset")
    menu.add_row("[3]", "Show evaluation metrics")
    menu.add_row("[4]", "Show global keyword analysis")
    menu.add_row("[5]", "Show LR vs. Naive Bayes comparison")
    menu.add_row(Text("[q]"), "Quit")

    while True:
        console.print()
        console.print(Rule("[bold]Main Menu[/]", style="bright_blue"))
        console.print(menu)
        choice = console.input("[cyan]Choice > [/]").strip().lower()

        if   choice == '1': action_single(model, vectorizer)
        elif choice == '2': action_batch(df, model, vectorizer)
        elif choice == '3': print_metrics(metrics)
        elif choice == '4': print_global_keywords(model, vectorizer)
        elif choice == '5': print_nb_comparison(metrics, nb_metrics)
        elif choice in ('q', 'quit', 'exit'):
            console.print("\n[dim]Goodbye[/]\n")
            break
        else:
            console.print("[yellow]Invalid — enter 1–5 or q[/]")

if __name__ == '__main__':
    train_path   = sys.argv[1] if len(sys.argv) > 1 else 'game_rvw_csvs'
    analyze_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        train_df = load_data(train_path)
    except FileNotFoundError:
        console.print(f"[red]Error:[/] Not found: {train_path}")
        console.print(
            "Usage:  python steam_sentiment.py [train_path] [analyze_path]"
        )
        sys.exit(1)

    train_df = balanced_sample(train_df)
    model, vectorizer, metrics, nb_metrics = train(train_df)

    if analyze_path:
        console.print(f"\n[bold cyan]Loading analysis dataset:[/] {analyze_path}")
        try:
            analyze_df = load_data(analyze_path)
        except FileNotFoundError:
            console.print(f"[red]Error:[/] Not found: {analyze_path}")
            sys.exit(1)
    else:
        analyze_df = train_df

    main_menu(analyze_df, model, vectorizer, metrics, nb_metrics)
