# app/prediction.py

import os
import pickle

def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # direktori file prediction.py
    model_path = os.path.join(base_dir, "..", "models", "heart_attack_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)



def predict_heart_attack(model, input_df, threshold=0.35):
    prob = model.predict_proba(input_df)[0][1]
    pred = 1 if prob > threshold else 0
    return pred, prob

