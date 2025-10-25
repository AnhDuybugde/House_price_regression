import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import preprocess_data
from src.models import get_models
from src.evaluation import evaluate_model

# Load data
df = pd.read_csv("data/train.csv")
y = df["SalePrice"]
X = df.drop(columns=["SalePrice"])

# Preprocess
X = preprocess_data(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate
models = get_models()
for name, model in models.items():
    model.fit(X_train, y_train)
    result = evaluate_model(model, X_train, y_train, X_test, y_test)
    print(f"{name}: {result}")

