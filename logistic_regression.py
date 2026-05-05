# CAP 5610 - Group 10 | Emotion Detection
# logistic_regression.py - Nicolas Orellana
#
# HOW TO RUN THIS FILE
#
# STEP 0 - Install required packages (only need to do this once)
#   pip install pandas numpy scipy scikit-learn
#
# STEP 1 - Prepare the data
#   Run preprocessing.py first so the CSV and TF-IDF files exist:
#     python preprocessing.py
#   This creates these files inside the data/ folder:
#     data/train_processed.csv
#     data/test_processed.csv
#     data/tfidf_train.npz
#     data/tfidf_test.npz
#
# STEP 2 - Run the Logistic Regression
#   python logistic_regression.py
#
# STEP 3 - Check the results
#   Accuracy, F1, and error analysis will print in the terminal.

import pandas as pd
import numpy as np
import scipy.sparse as sparse
from sklearn.linear_model import LogisticRegression


# STEP 1 - LOAD PREPROCESSED DATA

train_data = pd.read_csv("data/train_processed.csv")
test_data  = pd.read_csv("data/test_processed.csv")

train_features = sparse.load_npz("data/tfidf_train.npz")
test_features  = sparse.load_npz("data/tfidf_test.npz")

train_labels = train_data["label"]
test_labels  = test_data["label"]

emotion_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]


# STEP 2 - TRAIN THE MODEL

# class_weight="balanced" makes the model pay more attention to rarer emotions
# so it does not just predict the most common class all the time.
logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=1.0,
    solver="lbfgs",
    multi_class="multinomial"
)
logistic_model.fit(train_features, train_labels)


# STEP 3 - PREDICT ON THE TEST SET

test_predictions = logistic_model.predict(test_features)
test_proba       = logistic_model.predict_proba(test_features)  # shape: (n, 6)


# STEP 4 - BUILD THE ERROR ANALYSIS TABLE

# TEXT_COL is the column name for the raw tweet text — update if yours is named differently.
TEXT_COL = "text"

error_df = test_data.copy().reset_index(drop=True)
error_df["true_label"]      = test_labels.values
error_df["pred_label"]      = test_predictions
error_df["true_emotion"]    = error_df["true_label"].map(lambda x: emotion_names[x])
error_df["pred_emotion"]    = error_df["pred_label"].map(lambda x: emotion_names[x])
error_df["confidence"]      = test_proba.max(axis=1)           # prob of the predicted class
error_df["true_confidence"] = [                                 # prob the model gave to the TRUE class
    test_proba[i, error_df.loc[i, "true_label"]]
    for i in range(len(error_df))
]

# Keep only rows where the model was wrong, sorted by how confident it was
misclassified = (
    error_df[error_df["true_label"] != error_df["pred_label"]]
    .copy()
    .sort_values("confidence", ascending=False)
    .reset_index(drop=True)
)

print(f"\nTotal misclassified: {len(misclassified)} / {len(test_data)}")
print(f"Error rate: {len(misclassified)/len(test_data):.2%}\n")


# STEP 5 - PRINT ERROR EXAMPLES

# You can filter by true emotion, predicted emotion, or both.
# Example: show_errors(misclassified, n=5, true_em="fear", pred_em="sadness")
def show_errors(df, n=10, true_em=None, pred_em=None):
    subset = df.copy()
    if true_em:
        subset = subset[subset["true_emotion"] == true_em]
    if pred_em:
        subset = subset[subset["pred_emotion"] == pred_em]

    subset = subset.head(n)
    sep = "─" * 72

    for _, row in subset.iterrows():
        print(sep)
        print(f"  Tweet       : {row[TEXT_COL]}")
        print(f"  True label  : {row['true_emotion']}  (model gave it {row['true_confidence']:.2%} prob)")
        print(f"  Pred label  : {row['pred_emotion']}  (model confidence: {row['confidence']:.2%})")

    print(sep)
    print(f"Shown {len(subset)} example(s).\n")


# STEP 6 - RUN ERROR ANALYSIS

print("=== Top 10 Most Confident Mistakes ===")
show_errors(misclassified, n=10)

print("=== Sadness -> Joy Confusions ===")
show_errors(misclassified, n=5, true_em="sadness", pred_em="joy")

print("=== Fear -> Sadness Confusions ===")
show_errors(misclassified, n=5, true_em="fear", pred_em="sadness")

print("=== Love -> Joy Confusions ===")
show_errors(misclassified, n=5, true_em="love", pred_em="joy")

# Summary table — how many times each (true, predicted) pair was confused
print("=== Confusion Pair Counts (True -> Predicted) ===")
pair_counts = (
    misclassified
    .groupby(["true_emotion", "pred_emotion"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
print(pair_counts.to_string(index=False))
