from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
import numpy as np

def feature_engineering(df, poly_degree=2, apply_pca=False, n_pca_components=0.95):
    df = df.copy()

    # --- Tính các cột mới ---
    df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["TotalBath"] = df["FullBath"] + df["HalfBath"]*0.5 + df["BsmtFullBath"] + df["BsmtHalfBath"]*0.5
    df["TotalPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasPorch"] = (df["TotalPorchSF"] > 0).astype(int)
    df["LivingAreaRatio"] = df["GrLivArea"] / df["LotArea"]
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]

    # --- Impute missing cho numeric trước Poly/PCA ---
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=50, random_state=42), 
                               max_iter=10, random_state=42)
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # --- PolynomialFeatures ---
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(df[num_cols])
    poly_cols = poly.get_feature_names_out(num_cols)
    df_poly = pd.DataFrame(X_poly, columns=poly_cols, index=df.index)

    # --- PCA ---
    if apply_pca:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_poly)
        pca = PCA(n_components=n_pca_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)
        df = pd.concat([df.drop(columns=num_cols), df_pca], axis=1)
        final_num_cols = pca_cols
    else:
        df = pd.concat([df.drop(columns=num_cols), df_poly], axis=1)
        final_num_cols = poly_cols

    return df, final_num_cols
