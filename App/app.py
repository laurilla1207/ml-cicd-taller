import gradio as gr
import joblib
import numpy as np
from pathlib import Path

# Carga el modelo desde /Model/model.joblib (repositorio del Space)
MODEL = Path(__file__).resolve().parents[1] / "Model" / "model.joblib"

def load_model():
    if MODEL.exists():
        bundle = joblib.load(MODEL)
        return bundle["pipeline"]
    return None

pipe = load_model()

def predict(features_csv: str):
    if pipe is None:
        return "Model not found. Train locally and upload Model/model.joblib."
    try:
        vals = np.array([float(x.strip()) for x in features_csv.split(",")]).reshape(1, -1)
        pred = pipe.predict(vals)[0]
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(vals)[0][1]
            return f"Prediction: {int(pred)} | Prob(1): {proba:.4f}"
        return f"Prediction: {int(pred)}"
    except Exception as e:
        return f"Error: {e}"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, label="30 valores separados por coma"),
    outputs="text",
    title="Drug/Cancer Classification",
    description="Pega 30 números separados por coma (dataset Breast Cancer)."
)

if __name__ == "__main__":
    demo.launch()
