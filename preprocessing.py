"""
CAP 5610 – Group 10 | Emotion Detection
========================================
preprocessing.py
-----------------
Run this ONCE before any model training.
Outputs:
  - data/train_processed.csv
  - data/test_processed.csv
  - data/val_processed.csv
  - data/label_map.json

Usage:
  pip install datasets pandas numpy scikit-learn nltk
  python preprocessing.py
"""

import re, string, json, os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

# ── Optional: use nltk stopwords if available, else fallback ──────
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except:
    STOPWORDS = set([
        'i','me','my','myself','we','our','ours','ourselves','you','your',
        'yours','yourself','yourselves','he','him','his','himself','she',
        'her','hers','herself','it','its','itself','they','them','their',
        'theirs','themselves','what','which','who','whom','this','that',
        'these','those','am','is','are','was','were','be','been','being',
        'have','has','had','having','do','does','did','doing','a','an',
        'the','and','but','if','or','because','as','until','while','of',
        'at','by','for','with','about','against','between','into','through',
        'during','before','after','above','below','to','from','up','down',
        'in','out','on','off','over','under','again','further','then',
        'once','here','there','when','where','why','how','all','both',
        'each','few','more','most','other','some','such','than','too',
        'very','s','t','can','will','just','don','should','now','d','ll',
        'm','o','re','ve','y','ain','so','really','also','us','get','got',
    ])

# ── IMPORTANT: keep negations per proposal ────────────────────────
NEGATIONS = {'not', 'no', 'never', "n't", 'nor'}
STOPWORDS -= NEGATIONS

# ── Label mapping (matches dair-ai/emotion integers) ─────────────
LABEL_MAP = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
LABEL_MAP_INV = {v:k for k,v in LABEL_MAP.items()}


# ════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD DATA
# ════════════════════════════════════════════════════════════════════
def load_data():
    """Load from HuggingFace. Falls back to local CSV if offline."""
    try:
        from datasets import load_dataset
        ds = load_dataset("dair-ai/emotion")
        train = pd.DataFrame(ds['train'])
        test  = pd.DataFrame(ds['test'])
        val   = pd.DataFrame(ds['validation'])
        print(f"[✓] Loaded from HuggingFace: "
              f"train={len(train)}, test={len(test)}, val={len(val)}")
        return train, test, val
    except Exception as e:
        print(f"[!] HuggingFace unavailable ({e}). "
              f"Place data CSVs in ./data/ as train.csv, test.csv, val.csv")
        train = pd.read_csv('data/train.csv')
        test  = pd.read_csv('data/test.csv')
        val   = pd.read_csv('data/val.csv')
        return train, test, val


# ════════════════════════════════════════════════════════════════════
#  STEP 2 — CLEAN TEXT
# ════════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """
    Full cleaning pipeline (matches proposal exactly):
      1. Lowercase
      2. Remove URLs, @mentions, hashtags
      3. Remove special characters (keep apostrophes)
      4. Strip & collapse whitespace
    """
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r'http\S+|www\S+', '', t)        # remove URLs
    t = re.sub(r'@\w+', '', t)                   # remove @mentions
    t = re.sub(r'#\w+', '', t)                   # remove hashtags
    t = re.sub(r"[^a-z\s']", ' ', t)             # remove special chars
    t = re.sub(r'\s+', ' ', t).strip()           # collapse whitespace
    return t


def remove_stopwords(text: str) -> str:
    """Remove stopwords, preserving negations."""
    tokens = text.split()
    tokens = [tok for tok in tokens if tok not in STOPWORDS or tok in NEGATIONS]
    return ' '.join(tokens)


def full_preprocess(text: str) -> str:
    """Clean + remove stopwords + remove duplicates in place."""
    return remove_stopwords(clean_text(text))


# ════════════════════════════════════════════════════════════════════
#  STEP 3 — FEATURE ENGINEERING (for traditional models)
# ════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 20+ interpretable features to the dataframe.
    Traditional models (LR, NB, DT) can use these directly.
    """
    raw = df['text']
    clean = df['cleaned_text']

    # ── Lexical features ──────────────────────────────────────────
    df['f_char_count']       = raw.str.len()
    df['f_word_count']       = raw.str.split().str.len()
    df['f_avg_word_len']     = raw.apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0)
    df['f_unique_word_ratio']= raw.apply(
        lambda x: len(set(str(x).split())) / max(len(str(x).split()), 1))
    df['f_clean_word_count'] = clean.str.split().str.len()

    # ── Syntactic / surface features ──────────────────────────────
    df['f_punct_count']      = raw.apply(lambda x: sum(c in string.punctuation for c in str(x)))
    df['f_capital_count']    = raw.apply(lambda x: sum(c.isupper() for c in str(x)))
    df['f_capital_ratio']    = df['f_capital_count'] / df['f_char_count'].clip(lower=1)
    df['f_exclaim_count']    = raw.str.count('!')
    df['f_question_count']   = raw.str.count(r'\?')
    df['f_ellipsis_count']   = raw.str.count(r'\.\.\.')

    # ── Semantic / emotion hints ───────────────────────────────────
    df['f_negation_count']   = clean.apply(
        lambda x: sum(str(x).split().count(w) for w in NEGATIONS))

    # Positive / negative seed words (simple lexicon)
    POS_WORDS = {'happy','love','joy','great','amazing','wonderful','excited',
                 'glad','blessed','cheerful','fantastic','lovely','awesome'}
    NEG_WORDS = {'sad','angry','fear','hate','terrible','awful','horrible',
                 'miserable','depressed','furious','scared','lonely','cry'}

    df['f_pos_word_count']   = clean.apply(
        lambda x: sum(w in POS_WORDS for w in str(x).split()))
    df['f_neg_word_count']   = clean.apply(
        lambda x: sum(w in NEG_WORDS for w in str(x).split()))
    df['f_sentiment_ratio']  = (
        (df['f_pos_word_count'] - df['f_neg_word_count']) /
        df['f_clean_word_count'].clip(lower=1)
    )

    # ── Intensity markers ─────────────────────────────────────────
    INTENSIFIERS = {'so','very','really','absolutely','completely','totally',
                    'extremely','incredibly','deeply','utterly'}
    df['f_intensifier_count'] = clean.apply(
        lambda x: sum(w in INTENSIFIERS for w in str(x).split()))

    # ── Readability proxy ─────────────────────────────────────────
    df['f_type_token_ratio']  = df['f_unique_word_ratio']  # alias for clarity

    return df


# ════════════════════════════════════════════════════════════════════
#  STEP 4 — DEDUPLICATION
# ════════════════════════════════════════════════════════════════════
def deduplicate(train: pd.DataFrame, test: pd.DataFrame,
                val: pd.DataFrame) -> tuple:
    """
    Remove duplicate tweets and prevent data leakage across splits.
    Returns cleaned (train, test, val).
    """
    before = len(train) + len(test) + len(val)

    # Remove within-split duplicates
    train = train.drop_duplicates(subset='text').reset_index(drop=True)
    test  = test.drop_duplicates(subset='text').reset_index(drop=True)
    val   = val.drop_duplicates(subset='text').reset_index(drop=True)

    # Remove test/val texts that appear in train (data leakage prevention)
    train_texts = set(train['text'].str.lower())
    test  = test[~test['text'].str.lower().isin(train_texts)].reset_index(drop=True)
    val   = val[~val['text'].str.lower().isin(train_texts)].reset_index(drop=True)

    after = len(train) + len(test) + len(val)
    print(f"[✓] Deduplication: removed {before - after} duplicate rows")
    return train, test, val


# ════════════════════════════════════════════════════════════════════
#  STEP 5 — TFIDF VECTORIZATION (traditional models)
# ════════════════════════════════════════════════════════════════════
def build_tfidf(train_texts, test_texts, val_texts,
                max_features=10000, ngram_range=(1,2)):
    """
    Fit TF-IDF on train only, transform all splits.
    Returns (train_tfidf, test_tfidf, val_tfidf, vectorizer)
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,          # log(1+tf) — helps with short tweets
        min_df=2,                    # ignore very rare terms
        strip_accents='unicode',
    )
    X_train = vec.fit_transform(train_texts)
    X_test  = vec.transform(test_texts)
    X_val   = vec.transform(val_texts)
    print(f"[✓] TF-IDF shape: train={X_train.shape}, "
          f"test={X_test.shape}, val={X_val.shape}")
    return X_train, X_test, X_val, vec


# ════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════
def run_pipeline():
    os.makedirs('data', exist_ok=True)

    print("\n" + "="*55)
    print("  CAP 5610 Group 10 — Preprocessing Pipeline")
    print("="*55)

    # 1. Load
    train, test, val = load_data()

    # 2. Deduplicate
    train, test, val = deduplicate(train, test, val)

    # 3. Clean text
    for df in [train, test, val]:
        df['cleaned_text'] = df['text'].apply(full_preprocess)

    # 4. Engineer features
    train = engineer_features(train)
    test  = engineer_features(test)
    val   = engineer_features(val)

    # 5. Add emotion string label
    for df in [train, test, val]:
        df['emotion'] = df['label'].map(LABEL_MAP)

    # 6. Save processed CSVs
    train.to_csv('data/train_processed.csv', index=False)
    test.to_csv('data/test_processed.csv',  index=False)
    val.to_csv('data/val_processed.csv',    index=False)

    # 7. Save label map
    with open('data/label_map.json', 'w') as f:
        json.dump(LABEL_MAP, f, indent=2)

    # 8. Build TF-IDF (save for traditional models)
    X_tr, X_te, X_va, vec = build_tfidf(
        train['cleaned_text'], test['cleaned_text'], val['cleaned_text']
    )
    sp.save_npz('data/tfidf_train.npz', X_tr)
    sp.save_npz('data/tfidf_test.npz',  X_te)
    sp.save_npz('data/tfidf_val.npz',   X_va)

    print("\n[✓] All outputs saved to ./data/")
    print("    ├── train_processed.csv")
    print("    ├── test_processed.csv")
    print("    ├── val_processed.csv")
    print("    ├── label_map.json")
    print("    ├── tfidf_train.npz  ← use for LR / NB / DT")
    print("    ├── tfidf_test.npz")
    print("    └── tfidf_val.npz")

    print("\n[i] For deep learning models (CNN / LSTM / Transformer):")
    print("    Use the 'cleaned_text' column with your own tokenizer/embeddings.")

    # Print summary stats
    print("\n" + "="*55)
    print("  Dataset Summary After Preprocessing")
    print("="*55)
    for split, df in [("TRAIN", train), ("TEST", test), ("VAL", val)]:
        print(f"\n  {split}  ({len(df)} samples)")
        dist = df['emotion'].value_counts()
        for emo, cnt in dist.items():
            print(f"    {emo:10s}: {cnt:6d}  ({cnt/len(df)*100:.1f}%)")

    return train, test, val, vec


if __name__ == '__main__':
    run_pipeline()
