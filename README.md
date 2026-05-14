# Steam Review Sentiment and Keyword Analysis System
CS 4210 Final Project | Natasha Wong | Hao Ji | Spring 2026

# Steam Review Sentiment Analysis

This project analyzes Steam game reviews to predict sentiment (positive/negative) and extract keywords that influence the sentiment. It uses machine learning models trained on review data.

## How to Run the Code

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/natmwong/cs4210-final-project.git
   cd cs4210-final-project
   ```

2. **Run the Script**:
   - To train and analyze reviews for all games in the `game_rvw_csvs/` directory:
     ```bash
     python steam_sentiment.py
     ```
   - To train on all games but analyze reviews from a specific game (e.g., Terraria):
     ```bash
     python steam_sentiment.py game_rvw_csvs/ 105600_Terraria.csv
     ```
   - To train and analyze reviews for a single game (e.g., Terraria):
     ```bash
     python steam_sentiment.py 105600_Terraria.csv
     ```

3. **Interactive Menu**:
   After running the script, you will be presented with an interactive menu to:
   - Analyze a single review.
   - Batch analyze random reviews.
   - View evaluation metrics.
   - View global keyword analysis.
   - Compare Logistic Regression and Naive Bayes models.

## Required Packages / Dependencies

Ensure you have Python 3.x installed. Install the required packages using:
```bash
pip install -r requirements.txt
```

### Dependencies:
- `pandas`: For data manipulation.
- `numpy`: For numerical operations.
- `nltk`: For text preprocessing.
- `scikit-learn`: For machine learning models.
- `rich`: For formatted console output.

## Code Structure

- **Main Script**: `steam_sentiment.py`
  - Entry point for training, inference, and demo.
  - Handles user interaction and integrates all modules.

- **Modules**:
  - `modules/preprocessing.py`: Text preprocessing functions.
  - `modules/data_loading.py`: Data loading and sampling functions.
  - `modules/training.py`: Model training and evaluation functions.
  - `modules/display.py`: Functions for displaying results and metrics.

## Main Training, Inference, and Demo Code

- **Training**: The `train` function in `modules/training.py` handles model training and evaluation.
- **Inference**: The `action_single` and `action_batch` functions in `steam_sentiment.py` handle single and batch review analysis.
- **Demo**: Run `steam_sentiment.py` to access the interactive menu for training and inference.

## Example Usage

- Train and analyze reviews for all games:
  ```bash
  python steam_sentiment.py
  ```
- Train on all games but analyze reviews from Terraria:
  ```bash
  python steam_sentiment.py game_rvw_csvs/ 105600_Terraria.csv
  ```

