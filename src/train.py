import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

EVAL_THRESHOLD = 0.70
LABELS = [0, 1, 2]


def _filter_params(params: dict, allowed: set[str]) -> dict:
    return {k: v for k, v in params.items() if k in allowed and v is not None}


def _build_model(params: dict):
    model_type = params.get("model_type", "random_forest")
    model_params = {k: v for k, v in params.items() if k != "model_type"}

    if model_type == "random_forest":
        allowed = {
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "criterion",
            "max_features",
            "class_weight",
            "n_jobs",
        }
        clean = _filter_params(model_params, allowed)
        return model_type, RandomForestClassifier(**clean, random_state=42)

    if model_type == "gradient_boosting":
        allowed = {
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        }
        clean = _filter_params(model_params, allowed)
        return model_type, GradientBoostingClassifier(**clean, random_state=42)

    if model_type == "logistic_regression":
        allowed = {
            "C",
            "max_iter",
            "solver",
            "class_weight",
            "n_jobs",
        }
        clean = _filter_params(model_params, allowed)
        return model_type, LogisticRegression(**clean, random_state=42)

    raise ValueError(
        "Unsupported model_type. Use one of: random_forest, gradient_boosting, logistic_regression"
    )


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho model.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    label_distribution = y_train.value_counts(normalize=True).sort_index()
    distribution_dict = {str(int(k)): float(v) for k, v in label_distribution.items()}
    warnings = []
    for cls in LABELS:
        ratio = distribution_dict.get(str(cls), 0.0)
        if ratio < 0.10:
            msg = f"WARNING: class {cls} ratio is {ratio:.4f} (< 0.10)"
            print(msg)
            warnings.append(msg)

    model_type, model = _build_model(params)

    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_param("model_type", model_type)

        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")
        cm = confusion_matrix(y_eval, preds, labels=LABELS)
        precision, recall, _, _ = precision_recall_fscore_support(
            y_eval, preds, labels=LABELS, zero_division=0
        )

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        for idx, cls in enumerate(LABELS):
            mlflow.log_metric(f"precision_class_{cls}", float(precision[idx]))
            mlflow.log_metric(f"recall_class_{cls}", float(recall[idx]))
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(f"Model type: {model_type}")

        os.makedirs("outputs", exist_ok=True)
        metrics = {
            "accuracy": float(acc),
            "f1_score": float(f1),
            "model_type": model_type,
            "label_distribution": distribution_dict,
            "class_metrics": {
                str(cls): {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                }
                for i, cls in enumerate(LABELS)
            },
            "warnings": warnings,
        }
        with open("outputs/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        report_lines = [
            "Performance Report",
            f"Model: {model_type}",
            f"Accuracy: {acc:.6f}",
            f"F1(weighted): {f1:.6f}",
            "",
            "Label distribution (train):",
        ]
        for cls in LABELS:
            report_lines.append(f"  class_{cls}: {distribution_dict.get(str(cls), 0.0):.6f}")
        report_lines.extend(["", "Confusion matrix (rows=true, cols=pred):"])
        for row in cm:
            report_lines.append("  " + " ".join(str(int(v)) for v in row))
        report_lines.extend(["", "Per-class precision/recall:"])
        for i, cls in enumerate(LABELS):
            report_lines.append(
                f"  class_{cls}: precision={float(precision[i]):.6f}, recall={float(recall[i]):.6f}"
            )
        if warnings:
            report_lines.extend(["", "Warnings:"])
            report_lines.extend([f"  {w}" for w in warnings])

        with open("outputs/report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    train(params)
