"""
Steam Review Sentiment and Keyword Analysis System
CS 4210 Final Project | Natasha Wong | Hao Ji | Spring 2026

Usage:
    python steam_sentiment.py <reviews.csv>          # train + interactive on your CSV
"""

import re
import nltk
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule
from rich import box

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# STOPWORDS  (self-contained — no NLTK network dependency)
# ─────────────────────────────────────────────────────────────────────────────
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than',
    'too','very','s','t','can','will','just','don','should','now','d','ll',
    'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren',
    'won','wouldn','game','games','play','played','playing','get','got','like',
}


# ─────────────────────────────────────────────────────────────────────────────

# PREPROCESSING (using NLTK)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

NLTK_STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in NLTK_STOPWORDS and t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    console.print(f"\n[bold cyan]Loading:[/] {csv_path}")
    df = pd.read_csv(csv_path)

    # Kaggle Steam format includes a 'language' column — filter English only
    if 'language' in df.columns:
        df = df[df['language'] == 'english'].copy()

    df = df.dropna(subset=['review', 'voted_up'])
    df = df[df['review'].str.strip() != '']
    df['label'] = df['voted_up'].astype(bool).astype(int)
    df['clean_review'] = df['review'].apply(preprocess_text)

    pos = df.label.sum()
    neg = (df.label == 0).sum()
    console.print(
        f"  [green]Positive:[/] {pos:,}   [red]Negative:[/] {neg:,}   "
        f"[dim]Total: {len(df):,}[/]"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLING  (handle class imbalance for large Kaggle datasets)
# ─────────────────────────────────────────────────────────────────────────────
def balanced_sample(df: pd.DataFrame, sample_size: int = 10000,
                    neg_ratio: float = 0.30) -> pd.DataFrame:
    neg_df = df[df.label == 0]
    pos_df = df[df.label == 1]
    ratio  = len(neg_df) / max(len(pos_df), 1)

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
        df = df.sample(
            n=sample_size, random_state=42, stratify=df['label']
        ).reset_index(drop=True)
        console.print(
            f"  [yellow]Large dataset — using stratified sample of {sample_size:,}[/]"
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    console.print("\n[bold cyan]Training model…[/]")
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_review'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'cm':        confusion_matrix(y_test, y_pred),
        'report':    classification_report(
                         y_test, y_pred,
                         target_names=['Negative', 'Positive'], zero_division=0),
    }
    console.print("  [green]Done.[/]")
    return model, vectorizer, metrics


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def global_keywords(model, vectorizer, n: int = 10):
    names = np.array(vectorizer.get_feature_names_out())
    coef  = model.coef_[0]
    pos   = list(zip(names[np.argsort(coef)[-n:][::-1]],
                     coef[np.argsort(coef)[-n:][::-1]]))
    neg   = list(zip(names[np.argsort(coef)[:n]],
                     coef[np.argsort(coef)[:n]]))
    return pos, neg


def review_keywords(review_text: str, model, vectorizer, n: int = 5):
    clean      = preprocess_text(review_text)
    vec        = vectorizer.transform([clean])
    proba      = model.predict_proba(vec)[0]
    pred       = int(np.argmax(proba))
    confidence = proba[pred]

    names = np.array(vectorizer.get_feature_names_out())
    coef  = model.coef_[0]
    rv    = vec.toarray()[0]
    nz    = np.where(rv > 0)[0]

    if len(nz):
        scores   = coef[nz] * rv[nz]
        top      = np.argsort(np.abs(scores))[-n:][::-1]
        keywords = [(names[nz[i]], scores[i]) for i in top]
    else:
        keywords = []

    sentiment = 'Positive' if pred == 1 else 'Negative'
    return sentiment, confidence, keywords


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def print_header():
    console.print()
    console.print(Rule(style="bright_blue"))
    console.print(
        "[bold bright_blue]  Steam Review Sentiment & Keyword Analysis System[/]"
    )
    console.print("[dim]  CS 4210 Final Project | Natasha Wong[/]")
    console.print(Rule(style="bright_blue"))


def print_metrics(metrics: dict):
    console.print()
    console.print(Rule("[bold]Model Evaluation Results[/]", style="cyan"))

    tbl = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan",
                title="Test Set Metrics", title_style="bold")
    tbl.add_column("Metric",  style="bold", width=14)
    tbl.add_column("Score",   justify="right", width=10)
    tbl.add_column("Bar",     width=30)

    for label, val in [("Accuracy",  metrics['accuracy']),
                        ("Precision", metrics['precision']),
                        ("Recall",    metrics['recall']),
                        ("F1-Score",  metrics['f1'])]:
        filled = int(val * 25)
        bar = "[green]" + "█" * filled + "[/][dim]" + "░" * (25 - filled) + "[/]"
        tbl.add_row(label, f"{val:.4f}", bar)

    console.print(tbl)

    cm = metrics['cm']
    console.print()
    cm_tbl = Table(box=box.SIMPLE_HEAVY, title="Confusion Matrix",
                   title_style="bold", show_header=True, header_style="bold magenta")
    cm_tbl.add_column("",             style="bold magenta", width=16)
    cm_tbl.add_column("Pred Neg",     justify="center", width=12)
    cm_tbl.add_column("Pred Pos",     justify="center", width=12)
    cm_tbl.add_row("True Negative",   str(cm[0][0]), str(cm[0][1]))
    cm_tbl.add_row("True Positive",   str(cm[1][0]), str(cm[1][1]))
    console.print(cm_tbl)


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


# ─────────────────────────────────────────────────────────────────────────────
# MENU ACTIONS
# ─────────────────────────────────────────────────────────────────────────────
def action_single(model, vectorizer):
    console.print(
        "\n[bold]Enter a review to analyze[/] "
        "[dim](leave blank to go back)[/]"
    )
    while True:
        console.print()
        review = console.input("[cyan]Review > [/]").strip()
        if not review:
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MENU LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main_menu(df, model, vectorizer, metrics):
    print_header()
    print_metrics(metrics)
    print_global_keywords(model, vectorizer)

    menu = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    menu.add_column(width=4,  style="bold cyan")
    menu.add_column(width=50)
    menu.add_row("[1]", "Analyze a single review  (type your own text)")
    menu.add_row("[2]", "Batch analyze random reviews from dataset")
    menu.add_row("[3]", "Show evaluation metrics")
    menu.add_row("[4]", "Show global keyword analysis")
    menu.add_row("[q]", "Quit")

    while True:
        console.print()
        console.print(Rule("[bold]Main Menu[/]", style="bright_blue"))
        console.print(menu)
        choice = console.input("[cyan]Choice > [/]").strip().lower()

        if   choice == '1': action_single(model, vectorizer)
        elif choice == '2': action_batch(df, model, vectorizer)
        elif choice == '3': print_metrics(metrics)
        elif choice == '4': print_global_keywords(model, vectorizer)
        elif choice in ('q', 'quit', 'exit'):
            console.print("\n[dim]Goodbye.[/]\n")
            break
        else:
            console.print("[yellow]Invalid — enter 1, 2, 3, 4, or q[/]")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'steam_reviews_sample.csv'

    try:
        df = load_data(csv_path)
    except FileNotFoundError:
        console.print(f"[red]Error:[/] File not found: {csv_path}")
        console.print("Usage:  python steam_sentiment.py <reviews.csv>")
        sys.exit(1)

    df = balanced_sample(df)
    model, vectorizer, metrics = train(df)
    main_menu(df, model, vectorizer, metrics)
