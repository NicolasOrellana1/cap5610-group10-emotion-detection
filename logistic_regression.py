# CAP 5610 - Group 10 | Logistic Regression - Nicolas Orellana

import pandas as pd
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

# load data
train_data = pd.read_csv("data/train_processed.csv")
test_data  = pd.read_csv("data/test_processed.csv")
val_data   = pd.read_csv("data/val_processed.csv")

train_features = sparse.load_npz("data/tfidf_train.npz")
test_features  = sparse.load_npz("data/tfidf_test.npz")
val_features   = sparse.load_npz("data/tfidf_val.npz")

train_labels = train_data["label"]
test_labels  = test_data["label"]
val_labels   = val_data["label"]

emotion_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# train model
print("Training model...")

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    C=1.0,
    solver="lbfgs",
    multi_class="multinomial"
)

logistic_model.fit(train_features, train_labels)

# evaluate on validation set
val_predictions = logistic_model.predict(val_features)

print("Validation Accuracy: ", round(accuracy_score(val_labels, val_predictions), 4))
print("Validation Macro F1: ", round(f1_score(val_labels, val_predictions, average="macro"), 4))
print("Validation Precision:", round(precision_score(val_labels, val_predictions, average="macro"), 4))
print("Validation Recall:   ", round(recall_score(val_labels, val_predictions, average="macro"), 4))

# evaluate on test set
test_predictions = logistic_model.predict(test_features)

print("Test Accuracy: ", round(accuracy_score(test_labels, test_predictions), 4))
print("Test Macro F1: ", round(f1_score(test_labels, test_predictions, average="macro"), 4))
print("Test Precision:", round(precision_score(test_labels, test_predictions, average="macro"), 4))
print("Test Recall:   ", round(recall_score(test_labels, test_predictions, average="macro"), 4))

print(classification_report(test_labels, test_predictions, target_names=emotion_names))

# confusion matrix
confusion = confusion_matrix(test_labels, test_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_names, yticklabels=emotion_names)
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("True Emotion")
plt.tight_layout()
plt.savefig("results/logistic_regression_confusion_matrix.png")
plt.show()