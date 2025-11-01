import streamlit as st
import pandas as pd
import joblib
import os
from src.data_preprocessing import handle_outliers, yeo_johnson_transform, transform_preprocessor
from src.feature_engineering import feature_engineering

# CONFIG
st.set_page_config(page_title=" House Price Prediction", layout="wide")
st.title("Dự đoán giá nhà - House Price Prediction")

# LOAD MODEL + PREPROCESSOR
MODEL_PATH = "experiments/models/house_price_stacking_best.pkl"
PREPROCESS_PATH = "experiments/models/house_price_stacking_preprocessor.pkl"

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    pre = joblib.load(PREPROCESS_PATH)
    return model, pre

model, pre = load_assets()
st.success(" Model & Preprocessor loaded successfully!")

# UPLOAD FILE HOẶC NHẬP LIỆU
st.sidebar.header("Dữ liệu đầu vào")

uploaded_file = st.sidebar.file_uploader("Tải lên file CSV", type=["csv"])
if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    st.write("### Dữ liệu đã tải lên")
    st.dataframe(input_df.head())
else:
    st.info(" Vui lòng upload file `test.csv` để dự đoán.")
    st.stop()

# TIỀN XỬ LÝ DỮ LIỆU
st.write("## Tiền xử lý dữ liệu")

# Bỏ cột SalePrice nếu có
if "SalePrice" in input_df.columns:
    input_df = input_df.drop(columns=["SalePrice"])

# Xử lý outlier + biến đổi phân phối
input_df = handle_outliers(input_df, limits=(0.01, 0.01))
input_df, _ = yeo_johnson_transform(input_df, threshold=0.5)

# Feature engineering (giống khi train)
input_df, _ = feature_engineering(
    input_df,
    poly_degree=pre["feature_engineering_params"]["poly_degree"],
    apply_pca=pre["feature_engineering_params"]["apply_pca"],
    n_pca_components=pre["feature_engineering_params"]["n_pca_components"]
)

# Lấy lại danh sách cột
num_cols = pre.get("imputer_feature_names", pre["num_cols"])
cat_cols = pre["cat_cols"]

# Báo cáo mismatch nếu có
missing_in_input = set(num_cols) - set(input_df.columns)
extra_in_input = set(input_df.columns) - set(num_cols)
if missing_in_input or extra_in_input:
    st.warning(f"Mismatch: thiếu {len(missing_in_input)}, dư {len(extra_in_input)} cột.")

# Transform bằng preprocessor đã fit
X_processed = transform_preprocessor(
    input_df,
    pre["encoder"],
    pre["imputer"],
    pre["scaler"],
    num_cols,
    cat_cols
)

# Reindex để đúng feature order
if "feature_order" in pre:
    X_processed = X_processed.reindex(columns=pre["feature_order"], fill_value=0)
    st.info(f"Dữ liệu đã được sắp xếp lại theo feature order ({len(pre['feature_order'])} features)")

# DỰ ĐOÁN
st.write("## Kết quả dự đoán")

y_pred = model.predict(X_processed)
result_df = pd.DataFrame({
    "PredictedPrice": y_pred
})

st.dataframe(result_df.head())

# TẢI XUỐNG KẾT QUẢ
output_path = "predictions.csv"
result_df.to_csv(output_path, index=False)
st.download_button(
    label="Tải kết quả về CSV",
    data=open(output_path, "rb"),
    file_name="predictions.csv",
    mime="text/csv"
)
st.success("Dự đoán hoàn tất và đã sẵn sàng tải xuống!")
