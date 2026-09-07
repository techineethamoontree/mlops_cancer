import mlflow
from sklearn.datasets import load_breast_cancer


def validate_data():
    mlflow.set_experiment("Breast Cancer - Data Validation")

    with mlflow.start_run():
        print("Starting data validation run...")
        mlflow.set_tag("ml.step", "data_validation")

        # 1. โหลดชุดข้อมูล Breast Cancer
        data = load_breast_cancer(as_frame=True)
        df = data.frame

        num_rows, num_cols = df.shape
        num_classes = df["target"].nunique()
        missing_values = df.isnull().sum().sum()

        # คำนวณ Class Balance (สัดส่วนของคลาสน้อยสุด)
        class_counts = df["target"].value_counts(normalize=True)
        min_class_ratio = class_counts.min()

        print(f"Dataset shape: {num_rows} rows, {num_cols} columns")
        print(f"Number of classes: {num_classes}")
        print(f"Missing values: {missing_values}")
        print(f"Min class ratio: {min_class_ratio:.4f}")

        # 2. Log metrics/params ลง MLflow
        mlflow.log_metric("num_rows", num_rows)
        mlflow.log_metric("num_cols", num_cols)
        mlflow.log_metric("missing_values", missing_values)
        mlflow.log_metric("class_balance", min_class_ratio)
        mlflow.log_param("num_classes", num_classes)

        # 3. เงื่อนไข Validation: 2 คลาส และคลาสน้อยสุดต้อง >= 20%
        validation_status = "Success"
        if missing_values > 0 or num_classes != 2 or min_class_ratio < 0.20:
            validation_status = "Failed"

        mlflow.log_param("validation_status", validation_status)
        print(f"Validation status: {validation_status}")

        if validation_status == "Failed":
            raise SystemExit("Data validation failed — ข้อมูลไม่ผ่านเกณฑ์การตรวจสอบ")

        print("Data validation run finished.")


if __name__ == "__main__":
    validate_data()
