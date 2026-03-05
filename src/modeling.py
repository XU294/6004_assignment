import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC

from config import RANDOM_STATE, CM_DIR, ROC_DIR


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            solver="lbfgs",
            l1_ratio=0.0,          # 纯 L2
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }


def safe_predict_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def evaluate_model(name, pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = safe_predict_score(pipeline, X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_score) if y_score is not None else np.nan,
        "PR-AUC": average_precision_score(y_test, y_score) if y_score is not None else np.nan,
    }

    # confusion matrix
    os.makedirs(CM_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_DIR / f"{name.replace(' ', '_').lower()}_cm.png")
    plt.close()

    # ROC curve
    if y_score is not None:
        os.makedirs(ROC_DIR, exist_ok=True)

        RocCurveDisplay.from_predictions(y_test, y_score)
        plt.title(f"{name} - ROC Curve")
        plt.tight_layout()
        plt.savefig(ROC_DIR / f"{name.replace(' ', '_').lower()}_roc.png")
        plt.close()

        PrecisionRecallDisplay.from_predictions(y_test, y_score)
        plt.title(f"{name} - PR Curve")
        plt.tight_layout()
        plt.savefig(ROC_DIR / f"{name.replace(' ', '_').lower()}_pr.png")
        plt.close()

    return metrics


def run_all_models(X_train, y_train, X_test, y_test, preprocessor_factory, selector):
    results = []
    models = get_models()

    for name, model in models.items():
        scale_needed = name in {"Logistic Regression", "SVM"}

        preprocessor = preprocessor_factory(scale_needed)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("selector", clone(selector)),
                ("model", model),
            ]
        )

        print(f"\nTraining model: {name}")
        metrics = evaluate_model(name, pipeline, X_train, y_train, X_test, y_test)
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    return results_df