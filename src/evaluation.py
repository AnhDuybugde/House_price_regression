from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Huấn luyện và đánh giá mô hình, trả về các chỉ số dạng số thực.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print("==== FINAL EVALUATION ====")
    print(f"Train RMSE : {train_rmse:.4f}")
    print(f"Test  RMSE : {test_rmse:.4f}")
    print(f"Train MAE  : {train_mae:.4f}")
    print(f"Test  MAE  : {test_mae:.4f}")
    print(f"Train R²   : {train_r2:.4f}")
    print(f"Test  R²   : {test_r2:.4f}")

    # Trả về giá trị số thực, không phải chuỗi
    return train_rmse, test_rmse, train_mae, test_mae, train_r2, test_r2
