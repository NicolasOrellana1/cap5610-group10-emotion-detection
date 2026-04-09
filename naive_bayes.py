# CAP 5610 - Group 10 | Naive Bayes - Junior Chaj-Mejia
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sparse
import seaborn as sns

# Load data set
training_set = pd.read_csv('data/train_processed.csv')
validation_set = pd.read_csv('data/val_processed.csv')
testing_set = pd.read_csv('data/test_processed.csv')

# Load features
training_features = sparse.load_npz('data/tfidf_train.npz')
validation_features = sparse.load_npz('data/tfidf_val.npz')
testing_features = sparse.load_npz('data/tfidf_test.npz')

# Load labels
training_labels = training_set['label']
validation_labels = validation_set['label']
testing_labels = testing_set['label']
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Tune hyperparameter alpha
best_alpha = None
best_f1 = -1

for alpha in [0.1, 0.5, 1.0, 2.0]:
    tune_model = MultinomialNB(alpha=alpha)
    tune_model.fit(training_features, training_labels)
    
    preds = tune_model.predict(validation_features)
    score = f1_score(validation_labels, preds, average='macro')
    
    print(f'alpha={alpha}, val_macro_f1={round(score, 4)}')
    
    if score > best_f1:
        best_f1 = score
        best_alpha = alpha

print(f'Best alpha selected: {best_alpha}')

# Instantiate model with laplace smoothing
model = MultinomialNB(alpha=best_alpha)
model.fit(training_features, training_labels)


# Evaluation helper method
def evaluate(labels, predictions, set='', dec=4):
    acc = round(accuracy_score(labels, predictions), dec)
    macro_f1 = round(f1_score(labels, predictions, average='macro'), dec)
    precis = round(precision_score(labels, predictions, average='macro'), dec)
    recall = round(recall_score(labels, predictions, average='macro'), dec)
    print(f'{set} Accuracy: {acc}')
    print(f'{set} Macro F1: {macro_f1}')
    print(f'{set} Precision: {precis}')
    print(f'{set} Recall: {recall}')


# Evaluate on validation predictions
validation_predictions = model.predict(validation_features)
evaluate(validation_labels, validation_predictions, set='Validation', dec=4)

# Evaluate on testing predictions
testing_predictions = model.predict(testing_features)
evaluate(testing_labels, testing_predictions, set='Test', dec=4)

# Present evaluation metrics
print(classification_report(testing_labels, testing_predictions, target_names=emotion_labels))

confusion = confusion_matrix(testing_labels, testing_predictions)
plt.figure(figsize=(8, 6))
plt.title("Naive Bayes - Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("True Emotion")
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.tight_layout()
plt.savefig("results/naive_bayes_confusion_matrix.png")
plt.show()