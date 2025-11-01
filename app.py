# app.py
import streamlit as st
import pandas as pd
import joblib

# ===== Load model và pipeline =====
model = joblib.load("experiments/models/house_price_stacking_best.pkl")        # model cuối cùng
preprocessor = joblib.load("experiments/models/house_price_stacking_preprocessor.pkl")     # pipeline bao gồm PCA

# ===== Streamlit UI =====
st.title("House Price Prediction")

st.write("Nhập các thông số ngôi nhà để dự đoán giá:")

# Ví dụ chỉ lấy một số feature quan trọng
MSSubClass = st.selectbox("MSSubClass", [20, 30, 40, 50, 60, 70, 80, 90, 120, 150, 160, 180, 190])
MSZoning = st.selectbox("MSZoning", ["RL", "RM", "C (all)", "FV", "RH"])
LotArea = st.number_input("Lot Area (sq ft)", min_value=100, max_value=200000, step=100)
OverallQual = st.slider("Overall Quality", 1, 10, 5)
OverallCond = st.slider("Overall Condition", 1, 10, 5)
YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2025, step=1)
GrLivArea = st.number_input("Above Ground Living Area (sq ft)", min_value=100, max_value=10000, step=10)
TotRmsAbvGrd = st.slider("Total Rooms Above Ground", 2, 20, 6)
GarageCars = st.slider("Garage Cars", 0, 5, 1)
GarageArea = st.number_input("Garage Area (sq ft)", min_value=0, max_value=2000, step=10)

# Tạo DataFrame từ input
input_df = pd.DataFrame({
    "MSSubClass": [MSSubClass],
    "MSZoning": [MSZoning],
    "LotArea": [LotArea],
    "OverallQual": [OverallQual],
    "OverallCond": [OverallCond],
    "YearBuilt": [YearBuilt],
    "GrLivArea": [GrLivArea],
    "TotRmsAbvGrd": [TotRmsAbvGrd],
    "GarageCars": [GarageCars],
    "GarageArea": [GarageArea],
})

# Dự đoán
if st.button("Predict"):
    try:
        st.write("Processing input and predicting...")
        X_processed = preprocessor.transform(input_df)
        pred_price = model.predict(X_processed)
        st.success(f"Predicted House Price: ${pred_price[0]:,.2f}")
    except Exception as e:
        st.write("Error")
        st.error(f"Error: {e}")

#streamlit run app.py