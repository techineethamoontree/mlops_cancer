import mlflow
from sklearn.datasets import load_wine


def load_and_predict():
    """
    Simulates a production scenario by loading a model using an alias
    from the MLflow Model Registry and using it for prediction.
    """
    MODEL_NAME = "wine-classifier-prod"
    MODEL_ALIAS = "staging"  # MLflow 3 ใช้ Alias แทน Stage เดิม (เช่น staging, champion)

    print(f"Loading model '{MODEL_NAME}' with alias '@{MODEL_ALIAS}'...")

    # Load the model from the Model Registry ด้วย Alias URI
    try:
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version has the alias '@{MODEL_ALIAS}' in the MLflow UI.")
        return

    # Prepare new sample data (as_frame=True เพื่อให้ชื่อคอลัมน์ตรงกับ signature ของโมเดล)
    X, y = load_wine(return_X_y=True, as_frame=True)
    sample_data = X.iloc[0:1]  # Using the first row as a sample
    actual_label = y.iloc[0]

    # Use the loaded model to make a prediction
    # No manual preprocessing is needed because we logged the entire pipeline
    prediction = model.predict(sample_data)

    print("-" * 30)
    print(f"Sample Data Features:\n{sample_data.iloc[0]}")
    print(f"Actual Label: {actual_label}")
    print(f"Predicted Label: {prediction[0]}")
    print("-" * 30)


if __name__ == "__main__":
    load_and_predict()
