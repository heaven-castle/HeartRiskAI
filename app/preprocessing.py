import pandas as pd
import pickle
import os

def preprocess_input(df, mode='predict'):
    df = df.copy()
    df = pd.get_dummies(df)

    if mode == 'predict':
        # Path absolut untuk file feature_names.pkl
        path = os.path.join(os.path.dirname(__file__), "..", "models", "feature_names.pkl")
        path = os.path.abspath(path)

        # Baca fitur yang dipakai saat training
        with open(path, "rb") as f:
            trained_features = pickle.load(f)

        # Tambah kolom yang hilang
        for col in trained_features:
            if col not in df.columns:
                df[col] = 0

        # Pastikan urutan kolom sama
        df = df[trained_features]

    return df
