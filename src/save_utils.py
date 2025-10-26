import os
import json
import joblib
import pandas as pd
from datetime import datetime

def save_results(model_name, results, model=None, save_model=True):
    """Chỉ lưu log/model nếu kết quả tốt hơn bản hiện có"""
    os.makedirs("experiments/logs", exist_ok=True)
    os.makedirs("experiments/models", exist_ok=True)

    log_path = f"experiments/logs/{model_name}_best.json"
    model_path = f"experiments/models/{model_name}_best.pkl"

    # Nếu chưa có file -> lưu luôn
    if not os.path.exists(log_path):
        _save_all(log_path, model_path, model_name, results, model, save_model)
        print(f"First result saved for {model_name}")
        return

    # Nếu đã có file -> đọc kết quả cũ để so sánh
    with open(log_path, "r") as f:
        prev = json.load(f)
    prev_score = prev.get("Test_R2", -999)
    new_score = results.get("Test_R2", -999)

    if new_score > prev_score:
        _save_all(log_path, model_path, model_name, results, model, save_model)
        print(f"New best model for {model_name} (R2 improved {prev_score:.4f} → {new_score:.4f})")
    else:
        print(f"Model {model_name} not improved (kept best R2={prev_score:.4f})")


def _save_all(log_path, model_path, model_name, results, model, save_model):
    """Lưu log và model"""
    results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=4)
    if save_model and model is not None:
        joblib.dump(model, model_path)


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
