from sklearn.datasets import load_breast_cancer

df = load_breast_cancer(as_frame=True).frame


def test_schema():
    assert df.shape[1] == 31
    assert "target" in df.columns


def test_no_missing():
    assert df.isnull().sum().sum() == 0


def test_two_classes():
    assert df["target"].nunique() == 2


def test_class_balance():
    min_ratio = df["target"].value_counts(normalize=True).min()
    assert min_ratio >= 0.20