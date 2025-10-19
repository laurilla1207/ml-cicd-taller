from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def train_and_save():
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(X_train, y_train)

    out_dir = Path("model")
    out_dir.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe}, out_dir / "model.joblib")

    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)

if __name__ == "__main__":
    train_and_save()
    print("? Modelo entrenado y guardado en model/model.joblib")
