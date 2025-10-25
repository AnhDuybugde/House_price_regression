import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(exclude=['float64', 'int64']).columns

    imputer = SimpleImputer(strategy="mean")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(df[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols)

    df_encoded = pd.DataFrame(encoded, columns=encoded_cols)
    df = pd.concat([df[num_cols].reset_index(drop=True), df_encoded], axis=1)

    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df)
    return df
