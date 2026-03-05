import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    MISSING_THRESHOLD,
    LOW_VARIANCE_THRESHOLD,
    CORR_THRESHOLD,
    TOP_K_FEATURES,
    RANDOM_STATE,
)


class RawFeatureFilter(BaseEstimator, TransformerMixin):
    """
    阶段1 + 阶段2：
    1) 删除高缺失列
    2) 删除低方差数值列
    3) 删除高相关数值列
    """

    def __init__(
        self,
        missing_threshold=MISSING_THRESHOLD,
        low_variance_threshold=LOW_VARIANCE_THRESHOLD,
        corr_threshold=CORR_THRESHOLD,
    ):
        self.missing_threshold = missing_threshold
        self.low_variance_threshold = low_variance_threshold
        self.corr_threshold = corr_threshold

        self.keep_columns_ = None
        self.dropped_missing_ = []
        self.dropped_low_variance_ = []
        self.dropped_corr_ = []

        self.numeric_columns_after_filter_ = []
        self.categorical_columns_after_filter_ = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RawFeatureFilter expects a pandas DataFrame.")

        X_work = X.copy()

        # ===== Stage 1a: high missing =====
        missing_ratio = X_work.isna().mean()
        self.dropped_missing_ = missing_ratio[missing_ratio > self.missing_threshold].index.tolist()
        X_work = X_work.drop(columns=self.dropped_missing_, errors="ignore")

        # Separate numeric / categorical
        numeric_cols = X_work.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_work.select_dtypes(exclude=[np.number]).columns.tolist()

        # ===== Stage 1b: low variance numeric =====
        self.dropped_low_variance_ = []
        kept_numeric = []

        if numeric_cols:
            temp_num = X_work[numeric_cols].copy()
            temp_num = temp_num.fillna(temp_num.median(numeric_only=True))

            vt = VarianceThreshold(threshold=self.low_variance_threshold)
            vt.fit(temp_num)

            kept_numeric = temp_num.columns[vt.get_support()].tolist()
            self.dropped_low_variance_ = [c for c in numeric_cols if c not in kept_numeric]

        # ===== Stage 2: high correlation numeric =====
        self.dropped_corr_ = []
        final_numeric = kept_numeric

        if kept_numeric:
            temp_num = X_work[kept_numeric].copy()
            temp_num = temp_num.fillna(temp_num.median(numeric_only=True))

            corr_matrix = temp_num.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            to_drop = [col for col in upper.columns if any(upper[col] > self.corr_threshold)]
            self.dropped_corr_ = to_drop
            final_numeric = [c for c in kept_numeric if c not in to_drop]

        self.numeric_columns_after_filter_ = final_numeric
        self.categorical_columns_after_filter_ = categorical_cols
        self.keep_columns_ = final_numeric + categorical_cols
        return self

    def transform(self, X):
        X = X.copy()
        return X[self.keep_columns_]

    def report(self, original_n_features: int):
        after_stage1 = original_n_features - len(self.dropped_missing_) - len(self.dropped_low_variance_)
        after_stage2 = len(self.keep_columns_)

        print("\n===== Feature Selection Report =====")
        print(f"Original features: {original_n_features}")
        print(f"Dropped (high missing): {len(self.dropped_missing_)}")
        print(f"Dropped (low variance): {len(self.dropped_low_variance_)}")
        print(f"After Stage 1: {after_stage1}")
        print(f"Dropped (high correlation): {len(self.dropped_corr_)}")
        print(f"After Stage 2: {after_stage2}")
        print(f"Remaining numeric: {len(self.numeric_columns_after_filter_)}")
        print(f"Remaining categorical: {len(self.categorical_columns_after_filter_)}")


def build_preprocessor(numeric_features, categorical_features, scale_numeric=False):
    if scale_numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor

def build_supervised_selector():
    """
    阶段3：监督式特征筛选（更快版本）
    使用单变量统计检验，保留前 TOP_K_FEATURES 个特征。
    这仍然是 supervised feature selection，因为它使用了 y。
    """
    selector = SelectKBest(
        score_func=f_classif,
        k=TOP_K_FEATURES
    )
    return selector