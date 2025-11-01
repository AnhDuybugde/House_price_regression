import os
import json
import joblib
import pandas as pd
from datetime import datetime

def save_results(model_name, results, model=None, preprocessor=None, save_model=True):
    os.makedirs("experiments/logs", exist_ok=True)
    os.makedirs("experiments/models", exist_ok=True)

    log_path = f"experiments/logs/{model_name}_best.json"
    model_path = f"experiments/models/{model_name}_best.pkl"
    preprocessor_path = f"experiments/models/{model_name}_preprocessor.pkl"

    if not os.path.exists(log_path):
        _save_all(log_path, model_path, preprocessor_path, model_name, results, model, preprocessor, save_model)
        print(f"First result saved for {model_name}")
        return

    with open(log_path, "r") as f:
        prev = json.load(f)
    prev_score = prev.get("test_r2", -999)
    new_score = results.get("test_r2", -999)

    if new_score > prev_score:
        _save_all(log_path, model_path, preprocessor_path, model_name, results, model, preprocessor, save_model)
        print(f"New best model for {model_name} (R2 improved {prev_score:.4f} → {new_score:.4f})")
    else:
        print(f"Model {model_name} not improved (kept best R2={prev_score:.4f})")

def _save_all(log_path, model_path, preprocessor_path, model_name, results, model, preprocessor, save_model):
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=4)
    if save_model:
        if model is not None:
            joblib.dump(model, model_path)
        if preprocessor is not None:
            joblib.dump(preprocessor, preprocessor_path)

def append_to_summary(model_name, results):
    """Ghi thêm kết quả tổng hợp (không ghi đè)"""
    os.makedirs("experiments", exist_ok=True)
    summary_path = "experiments/summary.csv"
    df = pd.DataFrame([{**{"Model": model_name}, **results}])
    if os.path.exists(summary_path):
        df.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        df.to_csv(summary_path, index=False)
    print("Summary updated in experiments/summary.csv")