import os

from config import DATA_PATH, RESULTS_DIR, TOP_K_FEATURES
from data_utils import load_data, basic_data_report, prepare_X_y, split_data
from feature_selection import (
    RawFeatureFilter,
    build_preprocessor,
    build_supervised_selector,
)
from modeling import run_all_models


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load
    df = load_data(DATA_PATH)
    basic_data_report(df)

    # 2. Prepare X / y
    X, y = prepare_X_y(df)
    print(f"\nFeatures before split: {X.shape[1]}")

    # 3. Split first (avoid leakage)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Stage 1 + Stage 2 feature selection on TRAIN only
    raw_filter = RawFeatureFilter()
    raw_filter.fit(X_train, y_train)
    raw_filter.report(original_n_features=X_train.shape[1])

    X_train_filtered = raw_filter.transform(X_train)
    X_test_filtered = raw_filter.transform(X_test)

    print(f"\nX_train after Stage 1+2: {X_train_filtered.shape}")
    print(f"X_test  after Stage 1+2: {X_test_filtered.shape}")

    numeric_features = raw_filter.numeric_columns_after_filter_
    categorical_features = raw_filter.categorical_columns_after_filter_

    def preprocessor_factory(scale_needed: bool):
        return build_preprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scale_numeric=scale_needed,
        )

    # 5. Stage 3 supervised selector (SelectKBest with K=TOP_K_FEATURES)
    selector = build_supervised_selector()

    # ===== Sanity check: print dimensionality after preprocessing and after SelectKBest =====
    # Use a "scaled" preprocessor (LR/SVM need scaling) to check the full transformed feature dimension.
    preprocessor_check = preprocessor_factory(scale_needed=True)

    X_train_transformed = preprocessor_check.fit_transform(X_train_filtered, y_train)
    X_test_transformed = preprocessor_check.transform(X_test_filtered)

    print("\n===== Dimensionality Check (after preprocessing) =====")
    print(f"After preprocessing (one-hot etc.) X_train: {X_train_transformed.shape}")
    print(f"After preprocessing (one-hot etc.) X_test : {X_test_transformed.shape}")

    X_train_k = selector.fit_transform(X_train_transformed, y_train)
    X_test_k = selector.transform(X_test_transformed)

    print("\n===== Dimensionality Check (after SelectKBest) =====")
    print(f"K (TOP_K_FEATURES) = {TOP_K_FEATURES}")
    print(f"After SelectKBest X_train: {X_train_k.shape}")
    print(f"After SelectKBest X_test : {X_test_k.shape}")
    # ================================================================================

    # 6. Run all models (your training/evaluation pipeline)
    results_df = run_all_models(
        X_train=X_train_filtered,
        y_train=y_train,
        X_test=X_test_filtered,
        y_test=y_test,
        preprocessor_factory=preprocessor_factory,
        selector=selector,
    )

    # 7. Save results
    save_path = RESULTS_DIR / "metrics_summary.csv"
    results_df.to_csv(save_path, index=False)

    print("\n===== Final Model Comparison =====")
    print(results_df)
    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    main()