# CAP 5610 - Group 10 | Transformer - Lydia Emmons

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, TrainerCallback
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse as sparse
from datasets import load_dataset
import numpy as np
import os

# Load in Raw Tweet-Emotion Data
emotion_dataset = load_dataset("dair-ai/emotion")

train_data = emotion_dataset["train"]
test_data  = emotion_dataset["test"]
val_data   = emotion_dataset["validation"]

# Emotion Labels for Raw Data
label_to_emotion = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Load in Transformer Model
model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Build label mappings expected by Hugging Face model configs
emotion_to_label = {emotion: idx for idx, emotion in label_to_emotion.items()}

def build_model():
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=6,
        id2label=label_to_emotion,
        label2id=emotion_to_label
    )


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


# Tokenize all splits and keep tensors friendly to Trainer
tokenized_train = train_data.map(tokenize_batch, batched=True)
tokenized_val = val_data.map(tokenize_batch, batched=True)
tokenized_test = test_data.map(tokenize_batch, batched=True)

tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_val = tokenized_val.rename_column("label", "labels")
tokenized_test = tokenized_test.rename_column("label", "labels")

tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


class PerEpochTestEvalCallback(TrainerCallback):
    """Runs test-set evaluation after each epoch (metrics prefixed with ``test_``)."""

    def __init__(self, test_dataset, trainer_holder):
        self.test_dataset = test_dataset
        self.trainer_holder = trainer_holder

    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = self.trainer_holder[0]
        if trainer is None:
            return control
        trainer.evaluate(eval_dataset=self.test_dataset, metric_key_prefix="test")
        return control


def accuracies_by_epoch_from_log_history(history):
    """Pull validation (eval_*) and per-epoch test (test_*) accuracies keyed by epoch."""
    val_epoch_to_acc = {}
    test_epoch_to_acc = {}
    for log_item in history:
        if "epoch" not in log_item:
            continue
        epoch = float(log_item["epoch"])
        if "eval_accuracy" in log_item:
            val_epoch_to_acc[epoch] = log_item["eval_accuracy"]
        if "test_accuracy" in log_item:
            test_epoch_to_acc[epoch] = log_item["test_accuracy"]
    epochs = sorted(set(val_epoch_to_acc.keys()) | set(test_epoch_to_acc.keys()))
    val_accs = [val_epoch_to_acc.get(ep, np.nan) for ep in epochs]
    test_accs = [test_epoch_to_acc.get(ep, np.nan) for ep in epochs]
    return epochs, val_accs, test_accs


# Manual hyperparameter grid
learning_rates = [2e-5, 3e-5, 5e-5]
batch_sizes = [8, 16]
num_train_epochs = 4
weight_decays = [0.01, 0.05]
warmup_ratio = 0.1

best_f1 = -1.0
best_config = None
best_trainer = None
all_results = []

for lr in learning_rates:
    for bs in batch_sizes:
        for wd in weight_decays:
            run_name = f"lr_{lr}_bs_{bs}_wd_{wd}"
            print(f"\nTraining run: {run_name}")

            training_args = TrainingArguments(
                output_dir=f"./distilbert_emotion_outputs/{run_name}",
                learning_rate=lr,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=bs,
                num_train_epochs=num_train_epochs,
                weight_decay=wd,
                warmup_ratio=warmup_ratio,
                eval_strategy="epoch",
                logging_strategy="epoch",
                save_strategy="no",
                report_to="none",
            )

            trainer_holder = [None]
            callbacks = [PerEpochTestEvalCallback(tokenized_test, trainer_holder)]

            trainer = Trainer(
                model=build_model(),
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
            )
            trainer_holder[0] = trainer

            trainer.train()
            val_metrics = trainer.evaluate(eval_dataset=tokenized_val)
            val_f1 = val_metrics.get("eval_f1_macro", 0.0)
            val_acc = val_metrics.get("eval_accuracy", 0.0)

            all_results.append({
                "learning_rate": lr,
                "batch_size": bs,
                "weight_decay": wd,
                "val_f1_macro": val_f1,
                "val_accuracy": val_acc})
            print(f"Validation metrics ({run_name}):", val_metrics)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_config = {
                    "learning_rate": lr,
                    "batch_size": bs,
                    "weight_decay": wd,
                    "warmup_ratio": warmup_ratio
                }
                best_trainer = trainer

print("\nHyperparameter search results:")
for row in all_results:
    print(row)

print("\nBest config:", best_config, "| best val_f1_macro:", best_f1)

val_metrics = best_trainer.evaluate(eval_dataset=tokenized_val)
test_metrics = best_trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")

print("\nValidation metrics (best model):", val_metrics)
print("Test metrics (best model):", test_metrics)

# Final test predictions for confusion matrix + class report.
test_pred_output = best_trainer.predict(tokenized_test)
test_preds = np.argmax(test_pred_output.predictions, axis=-1)
test_labels = np.array(test_pred_output.label_ids)

conf_mat = confusion_matrix(test_labels, test_preds)
print(
    "\nClassification Report (Test):\n",
    classification_report(
        test_labels,
        test_preds,
        target_names=[label_to_emotion[i] for i in range(6)],
        digits=4
    )
)

# Display confusion matrix in the same style as decision_tree.
os.makedirs("results", exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    test_labels,
    test_preds,
    display_labels=[label_to_emotion[i] for i in range(6)],
    cmap="Blues",
    ax=ax,
    colorbar=True,
)
plt.tight_layout()
plt.savefig("results/confusion_matrix_test.png", dpi=300, bbox_inches="tight")
plt.show()

# Build validation/test accuracy vs epoch from best run's log history
# (per-epoch test metrics come from PerEpochTestEvalCallback during that run).
history = best_trainer.state.log_history
epochs, val_accs, test_accs = accuracies_by_epoch_from_log_history(history)
test_acc_final = test_metrics.get("test_accuracy", np.nan)
for i in range(len(test_accs)):
    if np.isnan(test_accs[i]):
        test_accs[i] = test_acc_final

plt.figure(figsize=(8, 5))
plt.plot(epochs, val_accs, marker="o", label="Validation Accuracy")
plt.plot(epochs, test_accs, marker="s", label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Transformer - Learning Curve")
plt.xticks(epochs)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_vs_epoch.png", dpi=300, bbox_inches="tight")
plt.close()

