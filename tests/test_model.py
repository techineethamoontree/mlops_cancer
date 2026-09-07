from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MIN_ACCURACY = 0.95
MIN_ROC_AUC = 0.98

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=10.0, max_iter=10000, random_state=42)),
    ]
).fit(X_tr, y_tr)


def test_accuracy_and_auc_gate():
    acc = accuracy_score(y_te, pipe.predict(X_te))
    roc_auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])

    assert acc >= MIN_ACCURACY, f"Accuracy {acc:.4f} ต่ำกว่าเกณฑ์ {MIN_ACCURACY}"
    assert roc_auc >= MIN_ROC_AUC, f"ROC-AUC {roc_auc:.4f} ต่ำกว่าเกณฑ์ {MIN_ROC_AUC}"


def test_pipeline_has_scaler():
    assert "scaler" in pipe.named_steps
