# CAP 5610 - Group 10 | Emotion Detection
# CNN.py - Nicolas Orellana
#
# ─────────────────────────────────────────────────────────────
# HOW TO RUN THIS FILE 
# ─────────────────────────────────────────────────────────────
#
# STEP 0 - Install required packages (only need to do this once)
#   pip install torch pandas numpy scikit-learn matplotlib seaborn
#
# STEP 1 - Prepare the data
#   Run preprocessing.py first so the CSV files exist:
#     python preprocessing.py
#   This creates three files inside the data/ folder:
#     data/train_processed.csv
#     data/test_processed.csv
#     data/val_processed.csv
#
# STEP 2 - Run the CNN
#   python CNN.py
#
# STEP 3 - Check the results
#   Two images will be saved inside the results/ folder:
#     results/cnn_learning_curve.png   - shows accuracy over time
#     results/cnn_confusion_matrix.png - shows which emotions were confused
#   Accuracy, F1, Precision, and Recall will also print in the terminal.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

# Check if a GPU is available. If yes, training will be much faster.
# If no GPU is found it will still run on the CPU, just slower.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# STEP 1 - LOAD PREPROCESSED DATA

# We load the three CSV files that preprocessing.py already cleaned.
# Each CSV has two columns: "cleaned_text" (the tweet) and "label" (0-5).
# The six emotion labels are: 0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise

print("Loading data...")

train_data = pd.read_csv("data/train_processed.csv")
test_data  = pd.read_csv("data/test_processed.csv")
val_data   = pd.read_csv("data/val_processed.csv")

emotion_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]


# STEP 2 - BUILD A VOCABULARY FROM THE TRAINING DATA

print("Building vocabulary...")

all_words = []
for tweet in train_data["cleaned_text"]:
    all_words.extend(str(tweet).split())

# Count how many times each word appears across all training tweets
word_counts = {}
for word in all_words:
    word_counts[word] = word_counts.get(word, 0) + 1

# Only keep words that appear at least 2 times.
# Words that appear only once are usually typos or rare slang — they add noise.
vocabulary = {"<PAD>": 0, "<UNK>": 1}
for word, count in word_counts.items():
    if count >= 2:
        vocabulary[word] = len(vocabulary)

vocabulary_size = len(vocabulary)
print("Vocabulary size:", vocabulary_size)



# STEP 3 - CONVERT EACH TWEET INTO A LIST OF NUMBERS

# Now that we have the vocabulary, we replace each word in a tweet with its ID.
# Example: "i feel happy" → [45, 12, 7, 0, 0, 0 ...] (padded to length 50)
#
# MAX_LENGTH = 50 means every tweet is exactly 50 numbers long.
# Tweets shorter than 50 are padded with 0s at the end.
# Tweets longer than 50 are cut off after the first 50 words.


MAX_LENGTH = 50

def tweet_to_integers(tweet, vocabulary, max_length):
    words = str(tweet).split()
    integer_sequence = []
    for word in words[:max_length]:
        # If the word is in our vocabulary use its ID, otherwise use 1 
        integer_sequence.append(vocabulary.get(word, 1))
    # Pad with zeros until we reach max_length
    while len(integer_sequence) < max_length:
        integer_sequence.append(0)
    return integer_sequence



# STEP 4 - CREATE PYTORCH DATASETS AND DATALOADERS

class TweetDataset(Dataset):
    def __init__(self, dataframe, vocabulary, max_length):
        self.sequences = []
        self.labels = []
        for _, row in dataframe.iterrows():
            sequence = tweet_to_integers(row["cleaned_text"], vocabulary, max_length)
            self.sequences.append(sequence)
            self.labels.append(row["label"])

    def __len__(self):
        # Returns how many tweets are in this dataset
        return len(self.labels)

    def __getitem__(self, index):
        # Returns one tweet (as a tensor of integers) and its label
        sequence = torch.tensor(self.sequences[index], dtype=torch.long)
        label    = torch.tensor(self.labels[index],    dtype=torch.long)
        return sequence, label


print("Creating datasets...")

train_dataset = TweetDataset(train_data, vocabulary, MAX_LENGTH)
val_dataset   = TweetDataset(val_data,   vocabulary, MAX_LENGTH)
test_dataset  = TweetDataset(test_data,  vocabulary, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)



# STEP 5 - DEFINE THE CNN ARCHITECTURE

# Here is how my CNN processes a tweet step by step:
#
# 1. Embedding layer
#    Converts each word ID into a vector of 128 numbers.
#    These 128 numbers capture the "meaning" of the word.
#    The model learns these vectors during training.
#
# 2. Three convolutional (Conv1d) layers with filter sizes 2, 3, and 4
#    Think of a filter like a sliding window that moves across the tweet.
#    A size-2 filter looks at 2 words at a time (bigrams).
#    A size-3 filter looks at 3 words at a time (trigrams).
#    A size-4 filter looks at 4 words at a time.
#    Each filter learns to detect a specific pattern, for example
#    a trigram filter might learn that "feel so happy" signals joy.
#
# 3. Max Pooling
#    After each convolutional layer we keep only the highest value found.
#    This picks the most important pattern the filter detected in the whole tweet.
#
# 4. Dropout (rate = 0.5)
#    During training, 50% of neurons are randomly turned off each pass.
#    This forces the model to not rely on any single neuron too much,
#    which helps it generalize and not just memorize the training data.
#
# 5. Fully Connected layer
#    Takes all the features gathered by the filters and maps them to
#    6 output scores — one score per emotion class.
#    The emotion with the highest score is the prediction.

class EmotionCNN(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, number_of_filters, number_of_classes, dropout_rate):
        super(EmotionCNN, self).__init__()

        # Converts word IDs into dense vectors. padding_idx=0 means the PAD
        # token always maps to a zero vector and does not affect learning.
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)

        # Three convolutional layers — each looks at a different window size.
        # Having all three lets the model pick up short AND longer phrase patterns.
        self.conv_size2 = nn.Conv1d(embedding_dim, number_of_filters, kernel_size=2)
        self.conv_size3 = nn.Conv1d(embedding_dim, number_of_filters, kernel_size=3)
        self.conv_size4 = nn.Conv1d(embedding_dim, number_of_filters, kernel_size=4)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu    = nn.ReLU()  # ReLU just sets any negative value to 0

        # The three conv layers each produce number_of_filters features,
        # so the input to this final layer is number_of_filters * 3.
        self.fully_connected = nn.Linear(number_of_filters * 3, number_of_classes)

    def forward(self, input_sequence):
        # input_sequence shape: (batch_size, MAX_LENGTH)

        embedded = self.embedding(input_sequence)
        # embedded shape is now: (batch_size, MAX_LENGTH, embedding_dim)

        # Conv1d needs shape (batch_size, embedding_dim, sequence_length)
        # so we swap the last two dimensions with permute
        embedded = embedded.permute(0, 2, 1)

        # Run each conv filter, apply ReLU, then keep the single max value
        features_size2 = self.relu(self.conv_size2(embedded))
        features_size2 = torch.max(features_size2, dim=2).values

        features_size3 = self.relu(self.conv_size3(embedded))
        features_size3 = torch.max(features_size3, dim=2).values

        features_size4 = self.relu(self.conv_size4(embedded))
        features_size4 = torch.max(features_size4, dim=2).values

        # Combine all three filter outputs into one big feature vector
        combined_features = torch.cat([features_size2, features_size3, features_size4], dim=1)
        combined_features = self.dropout(combined_features)

        # Produce 6 scores — one for each emotion
        output = self.fully_connected(combined_features)
        return output



# STEP 6 - SET HYPERPARAMETERS AND INITIALIZE THE MODEL


EMBEDDING_DIM      = 128
NUMBER_OF_FILTERS  = 128
NUMBER_OF_CLASSES  = 6
DROPOUT_RATE       = 0.5
LEARNING_RATE      = 0.001
NUMBER_OF_EPOCHS   = 10

model = EmotionCNN(
    vocabulary_size   = vocabulary_size,
    embedding_dim     = EMBEDDING_DIM,
    number_of_filters = NUMBER_OF_FILTERS,
    number_of_classes = NUMBER_OF_CLASSES,
    dropout_rate      = DROPOUT_RATE
).to(device)

print(model)

# Some emotions appear much more often than others in the dataset.
# Class weights make the model pay extra attention to the rarer emotions
# so it does not just learn to predict the most common one all the time.
class_counts  = train_data["label"].value_counts().sort_index().values
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights.to(device)

# CrossEntropyLoss measures how wrong the model's predictions are.
# Adam is the optimizer — it updates the model's weights to reduce that loss.
loss_function = nn.CrossEntropyLoss(weight=class_weights)
optimizer     = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)



# STEP 7 - TRAINING LOOP


print("\nTraining...")

train_accuracy_history = []
val_accuracy_history   = []

for epoch in range(NUMBER_OF_EPOCHS):

    # model.train() turns on dropout — important for the training phase
    model.train()
    correct_train = 0
    total_train   = 0

    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()               # clear leftover gradients from last batch
        predictions = model(sequences)      # forward pass
        loss = loss_function(predictions, labels)
        loss.backward()                     # compute gradients (backward pass)
        optimizer.step()                    # update weights

        predicted_classes = predictions.argmax(dim=1)  # pick the emotion with highest score
        correct_train += (predicted_classes == labels).sum().item()
        total_train   += labels.size(0)

    train_accuracy = correct_train / total_train

    # model.eval() turns off dropout so we get consistent predictions
    model.eval()
    correct_val = 0
    total_val   = 0

    # torch.no_grad() skips gradient calculations — we do not need them
    # during validation and it speeds things up
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels    = labels.to(device)
            predictions = model(sequences)
            predicted_classes = predictions.argmax(dim=1)
            correct_val += (predicted_classes == labels).sum().item()
            total_val   += labels.size(0)

    val_accuracy = correct_val / total_val

    train_accuracy_history.append(train_accuracy)
    val_accuracy_history.append(val_accuracy)

    print(f"Epoch {epoch+1}/{NUMBER_OF_EPOCHS}  Train Acc: {train_accuracy:.4f}  Val Acc: {val_accuracy:.4f}")



# STEP 8 - PLOT AND SAVE THE LEARNING CURVE

plt.figure(figsize=(8, 5))
plt.plot(range(1, NUMBER_OF_EPOCHS + 1), train_accuracy_history, label="Train Accuracy")
plt.plot(range(1, NUMBER_OF_EPOCHS + 1), val_accuracy_history,   label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN - Learning Curve")
plt.legend()
plt.tight_layout()
plt.savefig("results/cnn_learning_curve.png")
plt.show()
print("Learning curve saved")



# STEP 9 - EVALUATE ON THE TEST SET


print("\nEvaluating on test set...")

model.eval()
all_predictions = []
all_true_labels = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        predictions = model(sequences)
        predicted_classes = predictions.argmax(dim=1).cpu().numpy()
        all_predictions.extend(predicted_classes)
        all_true_labels.extend(labels.numpy())

print("Test Accuracy: ", round(accuracy_score(all_true_labels, all_predictions), 4))
print("Test Macro F1: ", round(f1_score(all_true_labels, all_predictions, average="macro"), 4))
print("Test Precision:", round(precision_score(all_true_labels, all_predictions, average="macro"), 4))
print("Test Recall:   ", round(recall_score(all_true_labels, all_predictions, average="macro"), 4))

# Full breakdown — precision, recall, and F1 for each individual emotion
print(classification_report(all_true_labels, all_predictions, target_names=emotion_names))



# STEP 10 - PLOT AND SAVE THE CONFUSION MATRIX


confusion = confusion_matrix(all_true_labels, all_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
            xticklabels=emotion_names, yticklabels=emotion_names)
plt.title("CNN - Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("True Emotion")
plt.tight_layout()
plt.savefig("results/cnn_confusion_matrix.png")
plt.show()
print("Confusion matrix saved")
