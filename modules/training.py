"""
Training module for Steam Review Sentiment Analysis.
Contains functions for training models and computing evaluation metrics.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from rich.console import Console

console = Console()

def _compute_metrics(y_test, y_pred) -> dict:
    """Compute evaluation metrics for model predictions."""
    return {
        'accuracy':      accuracy_score(y_test, y_pred),
        'precision':     precision_score(y_test, y_pred, zero_division=0),
        'recall':        recall_score(y_test, y_pred, zero_division=0),
        'f1':            f1_score(y_test, y_pred, zero_division=0),
        'cm':            confusion_matrix(y_test, y_pred),
        'report':        classification_report(
                             y_test, y_pred,
                             target_names=['Negative', 'Positive'],
                             zero_division=0),
        'neg_precision': precision_score(y_test, y_pred, pos_label=0, zero_division=0),
        'neg_recall':    recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        'neg_f1':        f1_score(y_test, y_pred, pos_label=0, zero_division=0),
    }

def train(df):
    """Train a logistic regression model and evaluate it."""
    console.print("\n[bold cyan]Training model…[/]")

    # Preprocess only the sampled data (not the full raw corpus)
    console.print("  [dim]Preprocessing sampled reviews…[/]")
    from .preprocessing import preprocess_text
    clean = df['review'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        clean, df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )

    # ── Hyperparameter search via 3-fold CV ──────────────────────────────────
    console.print(
        "  [dim]3-fold grid search: "
        "max_features in {3000,5000,8000}, C in {0.1,1.0,10.0}...[/]"
    )
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), lowercase=False)),
        ('lr',    LogisticRegression(
                      max_iter=1000, random_state=42,
                      class_weight='balanced')),   # step 1: balanced weighting
    ])
    param_grid = {
        'tfidf__max_features': [3000, 5000, 8000],
        'lr__C':               [0.1, 1.0, 10.0],
    }
    grid = GridSearchCV(
        pipeline, param_grid, cv=3,
        scoring='f1_macro', n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)

    best_p = grid.best_params_
    console.print(
        f"  [green]Best params:[/] "
        f"max_features={best_p['tfidf__max_features']}, "
        f"C={best_p['lr__C']}  "
        f"(CV f1_macro={grid.best_score_:.4f})"
    )

    model      = grid.best_estimator_.named_steps['lr']
    vectorizer = grid.best_estimator_.named_steps['tfidf']
    X_te       = vectorizer.transform(X_test)
    metrics    = _compute_metrics(y_test, model.predict(X_te))
    metrics['best_params'] = best_p
    metrics['cv_score']    = grid.best_score_

    # ── Naive Bayes baseline (same vectorizer for a fair comparison) ─────────
    console.print("  [dim]Training Naive Bayes baseline…[/]")
    X_tr_vec   = vectorizer.transform(X_train)
    nb         = MultinomialNB()
    nb.fit(X_tr_vec, y_train)
    nb_metrics = _compute_metrics(y_test, nb.predict(X_te))

    console.print("  [green]Done.[/]")
    return model, vectorizer, metrics, nb_metrics