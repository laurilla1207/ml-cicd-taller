import gradio as gr, joblib, numpy as np
from pathlib import Path
MODEL = Path(__file__).resolve().parents[1]/'Model'/'model.joblib'
pipe = joblib.load(MODEL)['pipeline'] if MODEL.exists() else None

def predict(csv_text: str):
    if pipe is None:
        return 'Model not found. Train first.'
    try:
        vals = np.array([float(x.strip()) for x in csv_text.split(',')]).reshape(1,-1)
        pred = pipe.predict(vals)[0]
        return f'Prediction: {int(pred)}'
    except Exception as e:
        return f'Error: {e}'

demo = gr.Interface(fn=predict, inputs='text', outputs='text',
                    title='Drug/Cancer Classification')
if __name__ == '__main__':
    demo.launch()
