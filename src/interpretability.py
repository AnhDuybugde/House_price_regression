import shap
import pandas as pd
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import matplotlib.pyplot as plt

def explain_model(model, X, sample_size=100):
    """
    Giải thích mô hình với:
    1. Permutation Importance
    2. SHAP values
    3. Partial Dependence Plots (PDP)
    """
    # 1. Permutation Importance
    print("\n--- Permutation Importance ---")
    perm_res = permutation_importance(model, X, model.predict(X), n_repeats=10, random_state=42, n_jobs=-1)
    perm_importance = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': perm_res.importances_mean,
        'importance_std': perm_res.importances_std
    }).sort_values(by='importance_mean', ascending=False)
    print(perm_importance.head(10))
    
    # Vẽ biểu đồ
    plt.figure(figsize=(8,5))
    plt.barh(perm_importance['feature'][:10][::-1], perm_importance['importance_mean'][:10][::-1])
    plt.xlabel("Permutation Importance")
    plt.title("Top 10 Feature Importance")
    plt.show()
    
    # 2. SHAP values
    print("\n--- SHAP values ---")
    # lấy mẫu nhỏ nếu dataset quá lớn
    X_sample = X.sample(min(sample_size, X.shape[0]), random_state=42)
    
    # Tree-based model có TreeExplainer, linear model có LinearExplainer
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        shap.summary_plot(shap_values, X_sample)
    except Exception as e:
        print("SHAP không áp dụng được cho model này:", e)
    
    # 3. Partial Dependence Plots (PDP)
    print("\n--- Partial Dependence Plots ---")
    top_features = perm_importance['feature'].head(3).tolist()  # chọn 3 feature quan trọng nhất
    fig, ax = plt.subplots(figsize=(12,4))
    PartialDependenceDisplay.from_estimator(model, X, features=top_features, ax=ax)
    plt.show()
