import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    ARTIFACT_DIR, EXPERIMENT, FEATURES, FIGURES, META_PATH, MODEL_DIR, MODEL_PATH,
    ROOT, SEED, TEST_SIZE, TRACKING_URI,
)
from src.data import load_data, split_data
from src.models import candidates, evaluate, make_pipeline

MODEL_VERSION = "1.0.0"
SCORING = ["f1", "roc_auc", "recall", "precision"]


def setup_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        ARTIFACT_DIR.mkdir(exist_ok=True)
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT)


def short(params):
    return {k.replace("model__", ""): v for k, v in params.items()}


def plot_confusion(y_test, proba, name, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (proba >= 0.5).astype(int), display_labels=["Stay", "Churn"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def log_grid(search):
    # log every grid combination as a child run so I can compare them in the MLflow UI
    results = pd.DataFrame(search.cv_results_)
    for _, row in results.iterrows():
        with mlflow.start_run(run_name=f"cv_rank_{int(row['rank_test_f1']):02d}", nested=True):
            mlflow.log_params(short(row["params"]))
            mlflow.log_metrics({
                "cv_f1": row["mean_test_f1"],
                "cv_roc_auc": row["mean_test_roc_auc"],
                "cv_recall": row["mean_test_recall"],
                "cv_precision": row["mean_test_precision"],
            })
    return results


def tune(name, model, grid, X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # churners are only about a quarter of the customers, so the grid is refit on F1:
    # a model that never predicts churn can still look fine on accuracy
    search = GridSearchCV(make_pipeline(model), grid, scoring=SCORING, refit="f1", cv=cv, n_jobs=-1)

    with mlflow.start_run(run_name=name):
        search.fit(X_train, y_train)
        proba = search.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, proba)

        mlflow.log_params({"model_type": name, "cv_folds": 5, "test_size": TEST_SIZE, **short(search.best_params_)})
        cv_scores = best_cv(search)
        mlflow.log_metrics({**cv_scores, **metrics})
        results = log_grid(search)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plot_confusion(y_test, proba, name, out / "confusion_matrix.png")
            results.sort_values("rank_test_f1").to_csv(out / "grid_search_results.csv", index=False)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
            joblib.dump(search.best_estimator_, out / "model.joblib", compress=3)
            mlflow.log_artifacts(tmp)

    return search, proba, metrics


def best_cv(search):
    i = search.best_index_
    res = search.cv_results_
    return {"cv_f1": float(res["mean_test_f1"][i]), "cv_roc_auc": float(res["mean_test_roc_auc"][i])}


def plot_roc(curves, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in curves.items():
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC curves on the test set")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_roc_curves.png", dpi=130)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = split_data(load_data())
    print(f"train {len(X_train)} rows, test {len(X_test)} rows, churn rate {y_train.mean():.1%}")

    setup_mlflow()
    searches, curves, scores = {}, {}, {}
    for name, (model, grid) in candidates().items():
        n = 1
        for values in grid.values():
            n *= len(values)
        print(f"\n{name}: {n} combinations x 5 folds")
        search, proba, metrics = tune(name, model, grid, X_train, y_train, X_test, y_test)
        searches[name], curves[name] = search, proba
        scores[name] = {**best_cv(search), **metrics}
        print(f"  best params: {short(search.best_params_)}")
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in scores[name].items()))
        plot_confusion(y_test, proba, name, FIGURES / f"07_confusion_{name}.png")
        (MODEL_DIR / "candidates").mkdir(exist_ok=True)
        joblib.dump(search.best_estimator_, MODEL_DIR / "candidates" / f"{name}.joblib", compress=3)

    plot_roc(curves, y_test)

    # choose using the CV score, the test set is only for the final numbers
    best = max(scores, key=lambda n: scores[n]["cv_f1"])
    pipeline = searches[best].best_estimator_
    joblib.dump(pipeline, MODEL_PATH, compress=3)

    metadata = {
        "model_name": best,
        "model_version": MODEL_VERSION,
        "best_params": short(searches[best].best_params_),
        "features": FEATURES,
        "target": "Churn (1 = customer left)",
        "threshold": 0.5,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "cv_f1": round(scores[best]["cv_f1"], 4),
        "cv_roc_auc": round(scores[best]["cv_roc_auc"], 4),
        "test_metrics": {k: round(v, 4) for k, v in scores[best].items() if not k.startswith("cv_")},
        "all_models": {n: {k: round(v, 4) for k, v in s.items()} for n, s in scores.items()},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": SEED,
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
        },
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))

    with mlflow.start_run(run_name="Final_Model"):
        mlflow.log_params({"chosen_model": best, **short(searches[best].best_params_)})
        mlflow.log_metrics(scores[best])
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(META_PATH))
        mlflow.log_artifact(str(FIGURES / "06_roc_curves.png"))

    print("\nsummary (test set, threshold 0.5)")
    print(pd.DataFrame(scores).T.round(4).to_string())
    print(f"\nchosen model: {best} (best cross-validated F1)")
    print(f"saved {MODEL_PATH.relative_to(ROOT)} and {META_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
