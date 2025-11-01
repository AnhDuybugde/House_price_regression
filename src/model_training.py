from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import optuna
from sklearn.metrics import mean_squared_error
import numpy as np

def objective(trial, X, y):
    alpha = trial.suggest_loguniform("elasticnet_alpha", 1e-3, 10)
    l1_ratio = trial.suggest_float("elasticnet_l1_ratio", 0.1, 0.9)
    ridge_alpha = trial.suggest_loguniform("ridge_alpha", 1e-2, 10)

    stack = StackingRegressor(
        estimators=[
            ("elasticnet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=42)),
            ("xgb", XGBRegressor(n_jobs=-1, random_state=42)),
            ("lgb", LGBMRegressor(n_jobs=-1, random_state=42))
        ],
        final_estimator=Ridge(alpha=ridge_alpha, max_iter=5000, random_state=42),
        passthrough=True,
        n_jobs=-1
    )

    stack.fit(X, y)
    y_pred = stack.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return rmse

def train_with_optuna(X, y, n_trials=30):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    print("Best trial:", study.best_trial.params)
    return study
