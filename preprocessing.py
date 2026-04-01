# CAP 5610 - Group 10 | Emotion Detection
# preprocessing.py
#
# HOW TO USE
# 1. pip install datasets pandas scikit-learn
# 2. python preprocessing.py
#
# This script loads the emotion dataset from HuggingFace,
# cleans the text, engineers features, and saves the results locally 

import re
import os
import pandas as pd
import scipy.sparse as sparse
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer


# STEP 1 - LOAD DATA FROM HUGGINGFACE
print("Loading dataset from HuggingFace...")

emotion_dataset = load_dataset("dair-ai/emotion")

train_data = pd.DataFrame(emotion_dataset["train"])
test_data  = pd.DataFrame(emotion_dataset["test"])
val_data   = pd.DataFrame(emotion_dataset["validation"])

print("Train:", len(train_data), "rows")
print("Test: ", len(test_data),  "rows")
print("Val:  ", len(val_data),   "rows")


# STEP 2 - MAP NUMERIC LABELS TO EMOTION NAMES
label_to_emotion = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

train_data["emotion"] = train_data["label"].map(label_to_emotion)
test_data["emotion"]  = test_data["label"].map(label_to_emotion)
val_data["emotion"]   = val_data["label"].map(label_to_emotion)


# STEP 3 - DEFINE STOPWORDS
# Stopwords are common words that carry no emotional meaning
# We keep negations like "not" and "never" because they change the meaning of a sentence
stopwords = set([
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his", "she",
    "her", "it", "its", "they", "them", "their", "what", "which", "who",
    "this", "that", "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "a", "an", "the", "and", "but", "if", "or", "as", "of", "at", "by",
    "for", "with", "about", "into", "through", "to", "from", "up", "down",
    "in", "out", "on", "off", "again", "then", "once", "here", "there",
    "when", "where", "how", "all", "both", "each", "more", "most", "other",
    "some", "than", "too", "very", "s", "t", "can", "will", "just", "should",
    "now", "so", "really", "also", "us", "get", "got", "would", "could",
])

negation_words = {"not", "no", "never", "nor", "n't"}

# Remove negation words from stopwords so they are not deleted
stopwords = stopwords - negation_words


# STEP 4 - DEFINE THE CLEANING FUNCTION
def clean_text(raw_text):
    # Lowercase everything
    cleaned = raw_text.lower()
    # Remove URLs
    cleaned = re.sub(r"http\S+|www\S+", "", cleaned)
    # Remove mentions like @username
    cleaned = re.sub(r"@\w+", "", cleaned)
    # Remove hashtags like #happy
    cleaned = re.sub(r"#\w+", "", cleaned)
    # Remove special characters and numbers, keep only letters
    cleaned = re.sub(r"[^a-z\s]", "", cleaned)
    # Collapse multiple spaces into one
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Remove stopwords but keep negation words
    words = [word for word in cleaned.split() if word not in stopwords or word in negation_words]
    return " ".join(words)


# STEP 5 - CLEAN ALL SPLITS
print("Cleaning text...")

train_data["cleaned_text"] = train_data["text"].apply(clean_text)
test_data["cleaned_text"]  = test_data["text"].apply(clean_text)
val_data["cleaned_text"]   = val_data["text"].apply(clean_text)


# STEP 6 - REMOVE DUPLICATE TWEETS
# This prevents the same tweet from appearing in both train and test
rows_before = len(train_data)
train_data = train_data.drop_duplicates(subset="text").reset_index(drop=True)
print("Duplicates removed:", rows_before - len(train_data))


# STEP 7 - ADD FEATURES TO EACH SPLIT
# These are simple numeric features that traditional models can use directly
def add_features(dataframe):
    dataframe["word_count"]     = dataframe["text"].str.split().str.len()
    dataframe["char_count"]     = dataframe["text"].str.len()
    dataframe["avg_word_length"]= dataframe["text"].apply(
                                      lambda text: sum(len(word) for word in text.split()) / max(len(text.split()), 1))
    dataframe["capital_count"]  = dataframe["text"].apply(lambda text: sum(1 for char in text if char.isupper()))
    dataframe["exclaim_count"]  = dataframe["text"].str.count("!")
    dataframe["question_count"] = dataframe["text"].str.count(r"\?")
    dataframe["negation_count"] = dataframe["cleaned_text"].apply(
                                      lambda text: sum(text.split().count(word) for word in negation_words))
    return dataframe

print("Adding features...")

train_data = add_features(train_data)
test_data  = add_features(test_data)
val_data   = add_features(val_data)


# STEP 8 - BUILD TF-IDF MATRIX
# TF-IDF turns cleaned text into numbers that models can learn from
# We fit only on train data so test data stays unseen during training
print("Building TF-IDF matrix...")

tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
tfidf_vectorizer.fit(train_data["cleaned_text"])


# STEP 9 - SAVE EVERYTHING TO LOCAL DATA FOLDER
# The data folder is already in .gitignore so these files will not be pushed to GitHub
# Every teammate just runs this script once to generate the data on their own machine
local_data_folder = "data/"

os.makedirs(local_data_folder, exist_ok=True)

# Save cleaned CSVs
train_data.to_csv(local_data_folder + "train_processed.csv", index=False)
test_data.to_csv( local_data_folder + "test_processed.csv",  index=False)
val_data.to_csv(  local_data_folder + "val_processed.csv",   index=False)

# Save TF-IDF matrices for traditional models
sparse.save_npz(local_data_folder + "tfidf_train.npz", tfidf_vectorizer.transform(train_data["cleaned_text"]))
sparse.save_npz(local_data_folder + "tfidf_test.npz",  tfidf_vectorizer.transform(test_data["cleaned_text"]))
sparse.save_npz(local_data_folder + "tfidf_val.npz",   tfidf_vectorizer.transform(val_data["cleaned_text"]))

print("All files saved to the data folder")
print("  data/train_processed.csv")
print("  data/test_processed.csv")
print("  data/val_processed.csv")
print("  data/tfidf_train.npz  - for Logistic Regression, Naive Bayes, Decision Tree")
print("  data/tfidf_test.npz")
print("  data/tfidf_val.npz")
print("For CNN, LSTM, and Transformer use the cleaned_text column directly")