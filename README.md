# 🌟 Loadstar
Loadstar is a tool that helps in performing code and data separation of non-standard ARM binaries. Please find the details in our paper.

---

## 🚀 Features
- Load a trained **TensorFlow/Keras** model and tokenizer.  
- Map test `.txt` files to dataset `.csv` files automatically.  
- Tokenize and pad sequences consistently with training.  
- Run inference and save per-instance predictions.  
- Compute **accuracy, precision, recall, F1 (per class, macro, micro)**.  
- Optional **Star Cutoff evaluation** (truncate evaluation up to last `0` in ground truth).  
- Generate `metrics.json` and `metrics_star.json` for reproducibility.  

---

## 📂 Project Structure
```
.
├── e2e_pipeline.py        # Main script
├── requirements.txt       # Python dependencies
├── data/                  # Place your dataset CSVs here
├── test/                  # Place your test .txt files here
├── results/               # Output predictions + metrics
└── README.md              # Documentation
```

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have:
   - a **trained Keras model** (`.keras` or SavedModel directory)  
   - a **pickled tokenizer** (`.pkl`)

---

## ▶️ Usage

Run the pipeline with:

```bash
python e2e_pipeline.py   --model path/to/model.keras   --tokenizer path/to/tokenizer.pkl   --dataset_path ./data   --test_path ./test   --result_path ./results   --maxlen 100   --batch_size 128   --verbose
```

### Arguments

| Argument         | Description |
|------------------|-------------|
| `--model`        | Path to Keras model (`.keras` or SavedModel). |
| `--tokenizer`    | Path to pickled tokenizer (`.pkl`). |
| `--dataset_path` | Directory containing dataset CSV files. |
| `--test_path`    | Directory containing test `.txt` files. |
| `--test_pattern` | Glob for test files (default: `*.txt`). |
| `--result_path`  | Directory to write output CSVs and metrics. |
| `--maxlen`       | Sequence length for padding (default: `100`). |
| `--batch_size`   | Batch size for predictions (default: `128`). |
| `--metrics_only` | Compute metrics only; skip writing per-file CSVs. |
| `--verbose`      | Enable debug logging. |

---

## 📊 Outputs

1. **Predictions per file**  
   Saved in `<result_path>/<dataset_name>.csv` with columns:
   ```csv
   inst,predicted
   ```

2. **Metrics (full evaluation)**  
   Saved in `<result_path>/metrics.json`:
   ```json
   {
     "accuracy": ,
     "precision_per_class": [],
     "recall_per_class": [],
     "f1_per_class": [],
     "macro_f1": ,
     "micro_f1":
   }
   ```

3. **Metrics (Star Cutoff evaluation)**  
   Saved in `<result_path>/metrics_star.json` (evaluation truncated to last `0` in ground truth).

---

## 🛠 Development Notes
- Uses **TensorFlow/Keras** for modeling.  
- Uses **scikit-learn** for metrics.  
- `report_metrics` supports both full and star evaluation.  
- `eval_files` cross-checks saved predictions against ground truth CSVs.  

---

<!-- ## 📌 Future Improvements
- Ad   -->


---

## 📜 License
MIT License. See `LICENSE` for details.  
