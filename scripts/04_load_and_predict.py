import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_and_predict():
    MODEL_NAME = "cancer-classifier-prod"
    MODEL_ALIAS = "staging"

    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop('target', axis=1)
    y = data.frame['target']

    # สุ่มดึงตัวอย่างอย่างละ 1 รายจากคลาส 0 (malignant) และคลาส 1 (benign)
    sample_class0 = X[y == 0].iloc[0:1]
    sample_class1 = X[y == 1].iloc[0:1]

    samples = pd.concat([sample_class0, sample_class1])
    actuals = [0, 1]
    predictions = model.predict(samples)

    class_map = {0: "malignant", 1: "benign"}

    print("-" * 50)
    for idx, (act, pred) in enumerate(zip(actuals, predictions)):
        act_str = class_map[act]
        pred_str = class_map[pred]
        is_correct = "Correct" if act == pred else "Incorrect"
        print(f"Sample {idx+1}: Actual = {act_str} | Predicted = {pred_str} -> {is_correct}")
    print("-" * 50)


if __name__ == "__main__":
    load_and_predict()