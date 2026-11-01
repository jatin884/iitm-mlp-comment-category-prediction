# Comment Category Prediction Challenge

**Machine Learning Practice (MLP) Project, IIT Madras BS Degree Program**
Author: Jatin Kumar

> Multi-class text classification: predict the category (4 classes) of an online comment from its text and metadata, scored by **macro-F1**. Built for the MLP Kaggle competition.
>
> **Result: 0.81411 private leaderboard score; best notebook score on Kaggle: 0.81859.**

---

## 1. Problem Statement

Given a comment and its metadata (reactions, votes, identity-related flags, timestamp), predict which of **4 categories** (labels 0 to 3) the comment belongs to. Submissions are evaluated with **macro-averaged F1**, so the rare classes matter as much as the common ones.

## 2. Dataset

The data comes from the competition (`comment-category-prediction-challenge`) and is **not included here** because it belongs to the course competition. Only the notebook is shared.

| Item | Detail |
|------|--------|
| Training rows | 198,000 |
| Test rows | 102,000 |
| Features | comment text, 3 emoticon counts, upvotes, downvotes, `if_1`, `if_2`, race / religion / gender (mostly missing), disability flag, post id, created date |
| Target | `label` (0, 1, 2, 3) |

**Class distribution (training set)**

| Label | Count | Share |
|-------|-------|-------|
| 0 | 114,173 | 57.7% |
| 2 | 62,440 | 31.5% |
| 1 | 15,918 | 8.0% |
| 3 | 5,469 | 2.8% |

The classes are strongly imbalanced, so the model uses class weighting, and accuracy alone would be misleading.

## 3. Approach

1. **Text features:** TF-IDF on the comment text (unigrams, 25,000 features, sublinear TF scaling, `min_df=3`, `max_df=0.9`).
2. **Numeric features:** emoticon counts, upvotes, downvotes, `if_1`, `if_2`, and the race, religion, gender and disability fields converted to numbers (missing values filled with 0).
3. **Combine:** text and numeric features are stacked into one sparse matrix.
4. **Model:** LightGBM multiclass classifier, 500 trees, learning rate 0.07, 96 leaves, `class_weight="balanced"`.
5. **Final fit:** trained on the full training set and used to predict the test set.

## 4. Experiments

Several models and feature sets were tried before the final one (scores from my project notes):

| Model | Notes | Score (macro-F1) |
|-------|-------|------------------|
| Multinomial Naive Bayes | TF-IDF + date features + encoded categoricals | 0.473 |
| Linear SVM | TF-IDF (1 to 2 grams) + date and categorical features | 0.678 |
| Voting ensemble (Logistic Regression + Linear SVM + LightGBM) | Engineered text-statistics and date features | 0.809 |
| LightGBM | TF-IDF + text-length features + date features | 0.807 |
| LightGBM (stratified split) | Same features, stratified train/validation split | 0.808 |
| **LightGBM (final)** | **TF-IDF (25k unigrams) + numeric features, balanced class weights** | **0.814 (private leaderboard)** |

## 5. Results

| Metric | Score |
|--------|-------|
| Private leaderboard score (macro-F1) | **0.81411** |
| Best score across my Kaggle notebook versions | **0.81859** |

- Gradient boosting beat the linear models and Naive Bayes by a wide margin on this mix of text and numeric features.
- Using `class_weight="balanced"` helped the rare classes (labels 1 and 3), which matters because macro-F1 weights every class equally.

## 6. Tools and Libraries

Python, pandas, NumPy, scikit-learn, LightGBM, SciPy, Kaggle Notebooks

## 7. Repository Contents

| File | Description |
|------|-------------|
| `mlp_comment_category_notebook.ipynb` | Cleaned notebook: data loading, quick exploration and the final model |
| `README.md` | This file |

## 8. How to Run

1. Open the notebook on Kaggle with the competition data attached, or download the data from the competition page.
2. Update the file paths at the top of the notebook if you run it locally.
3. Install the requirements: `pandas numpy scikit-learn lightgbm scipy`.
4. Run all cells. The last cell writes `submission.csv`.

## 9. Possible Improvements

- Add bigrams or character n-grams to the TF-IDF features.
- Tune the LightGBM parameters with cross-validation.
- Try pretrained text embeddings, or a fine-tuned transformer.
- Tune the decision thresholds per class to push macro-F1 higher.

## 10. Author

**Jatin Kumar**
IIT Madras BS Degree Program
