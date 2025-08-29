import argparse
import json
import time
import os
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pickle
import warnings

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference directly on CSVs found in TEST_PATH.")
    p.add_argument("--model", required=True, type=Path, help="Path to Keras model (.keras or SavedModel dir).")
    p.add_argument("--tokenizer", required=True, type=Path, help="Path to PICKLED tokenizer (.pkl).")
    p.add_argument("--test_path", required=True, type=Path, help="Directory containing test CSV files.")
    p.add_argument("--test_pattern", default="*.csv", help="Glob for files in TEST_PATH (default: *.csv).")
    p.add_argument("--result_path", required=True, type=Path, help="Directory to write output CSVs (and metrics).")
    p.add_argument("--maxlen", type=int, default=100, help="Sequence length for padding (must match training).")
    p.add_argument("--batch_size", type=int, default=128, help="Batch size for prediction.")
    p.add_argument("--metrics_only", action="store_true", help="Compute metrics only; do not write per-file CSVs.")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return p.parse_args()


def safe_tokenize(texts: List[str], tokenizer, maxlen: int) -> np.ndarray:
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")


def write_results_to_csv(out_csv: Path, inst_list: List[str], predicted: List[int]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["inst", "predicted"])
        for inst, p in zip(inst_list, predicted):
            w.writerow([inst, int(p)])
    logging.info("Saved predictions to %s", out_csv)


def report_metrics(y_true: List[int], y_pred: List[int], star_cutoff: bool = False) -> dict:
    """
    Compute classification metrics. If star_cutoff=True,
    truncate both y_true and y_pred up to the last 0 in y_true.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if star_cutoff and 0 in y_true:
        last_zero = len(y_true) - 1 - y_true[::-1].tolist().index(0)
        y_true = y_true[: last_zero + 1]
        y_pred = y_pred[: last_zero + 1]

    kw = dict(average=None, zero_division=0)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_per_class": precision_score(y_true, y_pred, **kw).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, **kw).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, **kw).tolist(),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def collect_test_csvs(test_path: Path, input_glob: str) -> List[Path]:
    csv_files = sorted(test_path.glob(input_glob))
    if not csv_files:
        raise FileNotFoundError(f"No input files matched pattern in {test_path}: {input_glob}")
    logging.info("Found %d test CSV(s) in %s matching %s.", len(csv_files), test_path, input_glob)
    return csv_files


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Unpack for convenience
    model_path: Path = args.model
    tokenizer_path: Path = args.tokenizer
    test_path: Path = args.test_path
    result_path: Path = args.result_path
    input_glob: str = args.test_pattern
    maxlen = args.maxlen
    batch_size = args.batch_size

    # Load tokenizer & model
    with tokenizer_path.open("rb") as f:
        tokenizer = pickle.load(f)
    model = load_model(model_path)

    all_y_true: List[int] = []
    all_y_pred: List[int] = []

    all_y_true_star: List[int] = []
    all_y_pred_star: List[int] = []

    test_csv_files = collect_test_csvs(test_path, input_glob)

    for csv_file in test_csv_files:
        df = pd.read_csv(csv_file)
        if "inst" not in df.columns:
            raise KeyError(f"'inst' column not found in {csv_file}")
        inst_list = df["inst"].astype(str).tolist()

        X = safe_tokenize(inst_list, tokenizer, maxlen)
        raw = model.predict(X, batch_size=batch_size, verbose=0)

        preds = (
            raw.argmax(axis=1).astype(int).tolist()
            if raw.ndim > 1 and raw.shape[-1] > 1
            else np.rint(raw.flatten()).astype(int).tolist()
        )

        if "ground_truth" in df.columns:
            gt = df["ground_truth"].astype(int).tolist()
            all_y_true.extend(gt)
            all_y_pred.extend(preds)

            if 0 in gt:
                last_zero = len(gt) - 1 - gt[::-1].index(0)
                gt_star = gt[: last_zero + 1]
                preds_star = preds[: last_zero + 1]
            else:
                gt_star = gt
                preds_star = preds

            all_y_true_star.extend(gt_star)
            all_y_pred_star.extend(preds_star)

            preds_to_write = preds_star
            inst_to_write = inst_list[: len(preds_star)]
        else:
            preds_to_write = preds
            inst_to_write = inst_list

        if not args.metrics_only:
            name = csv_file.stem
            out_csv = result_path / f"{name}.csv"
            write_results_to_csv(out_csv, inst_to_write, preds_to_write)

    result_path.mkdir(parents=True, exist_ok=True)
    if all_y_true and all_y_pred:
        metrics_full = report_metrics(all_y_true, all_y_pred, star_cutoff=False)
        (result_path / "metrics.json").write_text(json.dumps(metrics_full, indent=2))

        metrics_star = report_metrics(all_y_true_star, all_y_pred_star, star_cutoff=True)
        (result_path / "metrics_star.json").write_text(json.dumps(metrics_star, indent=2))

        if args.verbose:
            print("Full metrics:")
            print(json.dumps(metrics_full, indent=2))
            print("\nStar-cutoff metrics:")
            print(json.dumps(metrics_star, indent=2))
    elif args.verbose:
        print("No ground-truth column found; metrics not computed.")

if __name__ == "__main__":
    main()
