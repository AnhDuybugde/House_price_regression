import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Bật tính năng experimental
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler 
from scipy.stats.mstats import winsorize
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer


def load_data(path):
    return pd.read_csv(path)

def fit_preprocessor(df, scale=True):
    df = df.copy()

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(exclude=['float64', 'int64']).columns

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=50, random_state=42), random_state=42)
    scaler = StandardScaler() if scale else None

    # Fit & transform train data
    encoded = encoder.fit_transform(df[cat_cols]) if len(cat_cols) > 0 else None
    df[num_cols] = imputer.fit_transform(df[num_cols])
    if scale:
        df[num_cols] = scaler.fit_transform(df[num_cols])

    df_encoded = (
        pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        if len(cat_cols) > 0 else pd.DataFrame(index=df.index)
    )

    df_final = pd.concat([df[num_cols], df_encoded], axis=1)

    return df_final, encoder, imputer, scaler, num_cols, cat_cols

def transform_preprocessor(df, encoder, imputer, scaler, num_cols, cat_cols):
    df = df.copy()

    # Transform bằng object đã fit
    encoded = encoder.transform(df[cat_cols]) if len(cat_cols) > 0 else None
    df[num_cols] = imputer.transform(df[num_cols])
    if scaler:
        df[num_cols] = scaler.transform(df[num_cols])

    df_encoded = (
        pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        if len(cat_cols) > 0 else pd.DataFrame(index=df.index)
    )

    df_final = pd.concat([df[num_cols], df_encoded], axis=1)
    return df_final


def handle_outliers(df, limits=(0.01, 0.01)):
    """Cắt outlier bằng Winsorization trên tất cả cột numeric."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = winsorize(df[col], limits=limits)
    return df

def yeo_johnson_transform(df, threshold=0.5):
    """
    Biến đổi Yeo–Johnson (xử lý được cả giá trị âm/0).
    Trả về (df_mới, danh_sách_cột_được_biến_đổi).
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skewed = [col for col in numeric_cols if abs(df[col].skew()) > threshold]
    pt = PowerTransformer(method='yeo-johnson')
    df[skewed] = pt.fit_transform(df[skewed])
    return df, skewed




def test_iter_preprocess_data(df, method="bayesian"):
    # Xác định cột số & cột phân loại
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(exclude=['float64', 'int64']).columns

    # One-hot encode cho các cột phân loại (nếu có)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if len(cat_cols) > 0:
        encoded = encoder.fit_transform(df[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
    else:
        df_encoded = pd.DataFrame(index=df.index)

    # Chọn estimator cho IterativeImputer
    estimators = {
        "bayesian": BayesianRidge(),
        "rf": RandomForestRegressor(n_estimators=50, random_state=42),
        "et": ExtraTreesRegressor(n_estimators=50, random_state=42)
    }
    estimator = estimators.get(method, BayesianRidge())

    # Áp dụng IterativeImputer cho cột số
    imputer = IterativeImputer(estimator=estimator, random_state=42)
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # Gộp lại DataFrame cuối cùng
    df_final = pd.concat([df[num_cols], df_encoded], axis=1)

    return df_final

def knn_preprocess_data(df):
    # Xác định loại cột
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Fit riêng encoder trước
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    if categorical_cols:
        ohe.fit(df[categorical_cols])
        cat_encoded = ohe.transform(df[categorical_cols])
        cat_feature_names = ohe.get_feature_names_out(categorical_cols)
        df_encoded = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=df.index)
    else:
        df_encoded = pd.DataFrame(index=df.index)
        cat_feature_names = []

    # Gộp lại dataframe gồm numeric + encoded
    df_combined = pd.concat([df[numeric_cols], df_encoded], axis=1)

    # Áp dụng KNNImputer cho toàn bộ numeric (đã encode xong)
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = imputer.fit_transform(df_combined)

    # Trả về DataFrame hoàn chỉnh
    return pd.DataFrame(df_imputed, columns=df_combined.columns)

def si_preprocess_data(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(exclude=['float64', 'int64']).columns

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if len(cat_cols) > 0:
        encoded = encoder.fit_transform(df[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
    else:
        df_encoded = pd.DataFrame(index=df.index)

    # Impute numeric columns
    imputer = SimpleImputer(strategy="mean")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # Combine numeric + encoded
    df = pd.concat([df[num_cols], df_encoded], axis=1)
    return df

