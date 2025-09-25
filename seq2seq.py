
import argparse
import os
import json
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    BartTokenizerFast,
    BartForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from sklearn.metrics import f1_score, precision_score, recall_score

# -----------------------
# Emotion list (8)
# -----------------------
EMOTION_LIST = [
    "anger",
    "brain dysfunction (forget)",
    "emptiness",
    "hopelessness",
    "loneliness",
    "sadness",
    "suicide intent",
    "worthlessness"
]
NUM_LABELS = len(EMOTION_LIST)


# -----------------------
# Utilities
# -----------------------
def write_json(path, d):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4, ensure_ascii=False)


def label_from_label_id(value):
    """
    Convert label_id like 10100 (int/str/float) -> multi-hot list length NUM_LABELS (ints 0/1)
    """
    if isinstance(value, float):
        value = int(value)
    if isinstance(value, int):
        s = str(value).zfill(NUM_LABELS)
    elif isinstance(value, str):
        s = value.strip()
        if s.isdigit() and len(s) < NUM_LABELS:
            s = s.zfill(NUM_LABELS)
    else:
        s = "0" * NUM_LABELS
    s = s[:NUM_LABELS]
    return [1 if ch == "1" else 0 for ch in s]


def label_from_emotions(emo_list):
    """
    Convert emotions list of strings -> multi-hot list length NUM_LABELS
    """
    set_em = set(emo_list or [])
    return [1 if emo in set_em else 0 for emo in EMOTION_LIST]


def add_labels_column(ds):
    """
    Ensure dataset has 'labels' column in float (multi-hot).
    If dataset has 'label_id' or 'emotions', use them.
    Otherwise create zeros (useful for unlabeled predict-only sets).
    """
    cols = ds.column_names
    if "labels" in cols:
        # ensure float
        def to_float(batch):
            batch["labels"] = [[float(v) for v in lab] for lab in batch["labels"]]
            return batch
        return ds.map(to_float, batched=True)

    if "label_id" in cols:
        def map_fn(batch):
            labs = [label_from_label_id(x) for x in batch["label_id"]]
            batch["labels"] = [[float(v) for v in lab] for lab in labs]
            return batch
        return ds.map(map_fn, batched=True)

    if "emotions" in cols:
        def map_fn(batch):
            labs = [label_from_emotions(x) for x in batch["emotions"]]
            batch["labels"] = [[float(v) for v in lab] for lab in labs]
            return batch
        return ds.map(map_fn, batched=True)

    # fallback: create zeros
    def map_fn(batch):
        batch["labels"] = [[0.0] * NUM_LABELS for _ in range(len(batch[next(iter(batch))]))]
        return batch
    return ds.map(map_fn, batched=True)


# -----------------------
# Tokenize helper
# -----------------------
def make_preprocess_fn(tokenizer, max_source_length):
    def preprocess_function(examples):
        enc = tokenizer(
            examples["text"],
            max_length=max_source_length,
            padding="max_length",
            truncation=True
        )
        # keep labels if present, ensure float
        if "labels" in examples:
            enc["labels"] = [[float(v) for v in lab] for lab in examples["labels"]]
        return enc
    return preprocess_function


# -----------------------
# Metrics (EvalPrediction -> dict)
# -----------------------
def compute_metrics(eval_pred: EvalPrediction):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    # Some models return tuple with logits first
    if isinstance(preds, tuple):
        preds = preds[0]

    # Convert to numpy array (safe)
    logits = np.array(preds)
    # Expect shape (batch, NUM_LABELS)
    if logits.ndim == 3:
        # Unexpected: maybe seq2seq outputs; try to reduce by taking logits[:, 0, :]
        # But this indicates model isn't classification head — return empty metrics to avoid crash
        raise ValueError(f"Logits have 3 dims {logits.shape} — model may not be a classification model with num_labels={NUM_LABELS}")

    probs = torch.sigmoid(torch.from_numpy(logits)).detach().cpu().numpy()
    y_pred = (probs >= 0.5).astype(int)
    y_true = np.array(labels)

    results = {}
    # overall
    results["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    results["recall_micro"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    results["precision_micro"] = precision_score(y_true, y_pred, average="micro", zero_division=0)

    results["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    results["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    results["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)

    # per-class
    for i, emo in enumerate(EMOTION_LIST):
        results[f"f1_micro_{emo}"] = f1_score(y_true[:, i], y_pred[:, i], average="binary", zero_division=0)
        results[f"recall_micro_{emo}"] = recall_score(y_true[:, i], y_pred[:, i], average="binary", zero_division=0)
        results[f"precision_micro_{emo}"] = precision_score(y_true[:, i], y_pred[:, i], average="binary", zero_division=0)

        # for single-class macro == binary but keep names consistent
        results[f"f1_macro_{emo}"] = results[f"f1_micro_{emo}"]
        results[f"recall_macro_{emo}"] = results[f"recall_micro_{emo}"]
        results[f"precision_macro_{emo}"] = results[f"precision_micro_{emo}"]

    # Save file metrics as JSON and text
    write_json("test_results.json", results)
    with open("metrics.txt", "w", encoding="utf-8") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")

    return results


# -----------------------
# Train / Test flows
# -----------------------
def train(args):
    tokenizer = BartTokenizerFast.from_pretrained(args.model_name)
    model = BartForSequenceClassification.from_pretrained(
        args.model_name,
        problem_type="multi_label_classification",
        num_labels=NUM_LABELS
    )

    # load datasets (json)
    data_files = {}
    if args.train_path:
        data_files["train"] = args.train_path
    if args.val_path:
        data_files["validation"] = args.val_path
    if args.test_path:
        data_files["test"] = args.test_path

    dataset = load_dataset("json", data_files=data_files)

    # add labels column and ensure float
    for split in dataset.keys():
        dataset[split] = add_labels_column(dataset[split])

    # preprocess/tokenize
    preprocess_fn = make_preprocess_fn(tokenizer, args.max_source_length)
    for split in dataset.keys():
        dataset[split] = dataset[split].map(preprocess_fn, batched=True)

    # TrainingArguments
    out_dir = args.output_dir or args.model_name.replace("/", "_")
    training_args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(out_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",  # disable wandb/tensorboard integrations
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # save model & tokenizer
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    # final evaluate on test if present
    if "test" in dataset:
        print("Evaluating on test set...")
        res = trainer.evaluate(eval_dataset=dataset["test"])
        print("Final test metrics:", res)


def test(args):
    tokenizer = BartTokenizerFast.from_pretrained(args.model_name)
    # load local checkpoint (model_path) — must be folder with model files
    model = BartForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=NUM_LABELS
    )

    dataset = load_dataset("json", data_files={"test": args.test_path})
    dataset["test"] = add_labels_column(dataset["test"])
    preprocess_fn = make_preprocess_fn(tokenizer, args.max_source_length)
    dataset["test"] = dataset["test"].map(preprocess_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir or "./results_test",
        per_device_eval_batch_size=args.test_batch_size,
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    res = trainer.evaluate(eval_dataset=dataset["test"])
    print("Test results:", res)


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/test multi-label BART classifier")

    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--model_name", type=str, default="facebook/bart-base")
    parser.add_argument("--model_path", type=str, help="Path to local model folder for testing (checkpoint)")

    # data
    parser.add_argument("--train_path", type=str, help="train json file (for train mode)")
    parser.add_argument("--val_path", type=str, help="validation json file (for train mode)")
    parser.add_argument("--test_path", type=str, help="test json file (for train/test)")

    # training hyperparams
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)

    # testing
    parser.add_argument("--test_batch_size", type=int, default=8)

    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    if args.mode == "train":
        if not args.train_path or not args.val_path:
            raise ValueError("Train mode requires --train_path and --val_path")
        train(args)
    else:
        if not args.model_path or not args.test_path:
            raise ValueError("Test mode requires --model_path and --test_path")
        test(args)
