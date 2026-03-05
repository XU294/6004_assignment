import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    TARGET_COLUMN,
    DROP_COLUMNS,
    LEAKAGE_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
    STRATIFY,
)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def basic_data_report(df: pd.DataFrame):
    print("\n===== Basic Data Report =====")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    print(f"\nTarget column: {TARGET_COLUMN}")
    print("\nTarget counts:")
    print(df[TARGET_COLUMN].value_counts(dropna=False))
    print("\nTarget proportions:")
    print(df[TARGET_COLUMN].value_counts(normalize=True, dropna=False))

    print("\nDtypes summary:")
    print(df.dtypes.value_counts())

    print("\nObject columns:")
    print(df.select_dtypes(include="object").columns.tolist())

    missing_ratio = df.isna().mean().sort_values(ascending=False)
    print("\nTop 20 columns by missing ratio:")
    print(missing_ratio.head(20))

    print(f"\nColumns with missing ratio > 40%: {(missing_ratio > 0.40).sum()}")


def prepare_X_y(df: pd.DataFrame):
    cols_to_drop = set(DROP_COLUMNS + LEAKAGE_COLUMNS)
    existing_drop_cols = [c for c in cols_to_drop if c in df.columns]

    print("\nDropping predefined columns:")
    print(existing_drop_cols)

    X = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    X = X.drop(columns=existing_drop_cols, errors="ignore")

    y = df[TARGET_COLUMN].astype(int).copy()
    return X, y


def split_data(X, y):
    stratify_y = y if STRATIFY else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_y,
    )

    print("\n===== Train/Test Split =====")
    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_train positive rate: {y_train.mean():.4f}")
    print(f"y_test  positive rate: {y_test.mean():.4f}")

    return X_train, X_test, y_train, y_test