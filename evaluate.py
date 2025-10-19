from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def evaluate():
    bundle = joblib.load("model/model.joblib")
    pipe = bundle["pipeline"]

    X_test = np.load("model/X_test.npy")
    y_test = np.load("model/y_test.npy")

    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    out = Path("results"); out.mkdir(exist_ok=True)

    with open(out / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"accuracy: {acc:.4f}\n\n")
        f.write(report)

    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar()
    plt.xlabel("Predicted"); plt.ylabel("True")
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    evaluate()
    print("? Resultados en results/")
