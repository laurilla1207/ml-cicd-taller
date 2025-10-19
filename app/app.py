import gradio as gr
import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "model.joblib"

def load_model():
    return joblib.load(MODEL_PATH)["pipeline"] if MODEL_PATH.exists() else None

pipe = load_model()

def predict(features_csv: str):
    if pipe is None:
        return "Model not found. Train first."
    try:
        vals = np.array([float(x.strip()) for x in features_csv.split(",")]).reshape(1,-1)
        pred = pipe.predict(vals)[0]
        proba = pipe.predict_proba(vals)[0] if hasattr(pipe, "predict_proba") else None
        return f"Prediction: {int(pred)}" + (f" | Prob(1): {proba[1]:.4f}" if proba is not None else "")
    except Exception as e:
        return f"Error: {e}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, label="30 valores separados por coma"),
    outputs="text",
    title="Breast Cancer Classifier"
)

if __name__ == "__main__":
    demo.launch()
