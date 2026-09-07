import os

import mlflow
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def preprocess_data(test_size=0.25, random_state=42):
    mlflow.set_experiment("Breast Cancer - Data Preprocessing")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting preprocessing run with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")

        data = load_breast_cancer(as_frame=True)
        df = data.frame

        X = df.drop("target", axis=1)
        y = df["target"]

        # แบ่ง Train/Test 75:25 แบบ stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        processed_data_dir = "processed_data"
        os.makedirs(processed_data_dir, exist_ok=True)

        pd.concat([X_train, y_train], axis=1).to_csv(
            os.path.join(processed_data_dir, "train.csv"), index=False
        )
        pd.concat([X_test, y_test], axis=1).to_csv(
            os.path.join(processed_data_dir, "test.csv"), index=False
        )

        mlflow.log_param("test_size", test_size)
        mlflow.log_metric("training_set_rows", len(X_train))
        mlflow.log_metric("test_set_rows", len(X_test))

        mlflow.log_artifacts(processed_data_dir, artifact_path="processed_data")

        print("-" * 50)
        print(f"Preprocessing Run ID: {run_id}")
        print("-" * 50)


if __name__ == "__main__":
    preprocess_data()
