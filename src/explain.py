import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES, META_PATH, MODEL_DIR, MODEL_PATH, ROOT, SEED
from src.data import load_data, split_data
from src.features import feature_names
from src.train import setup_mlflow

OUT = ROOT / "reports" / "explain"


def transform(pipeline, X):
    names = feature_names(pipeline)
    return pd.DataFrame(pipeline.named_steps["prep"].transform(X), columns=names, index=X.index)


def save(name):
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def tree_importance(pipeline, X_test):
    # xgboost's own importance (gain)
    model = pipeline.named_steps["model"]
    imp = pd.Series(model.feature_importances_, index=feature_names(pipeline)).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index, imp.values, color="#4c72b0")
    ax.set_title("XGBoost feature importance (top 15)")
    save("08_feature_importance_xgboost.png")
    return imp.sort_values(ascending=False)


def shap_values(pipeline, X_train, X_test):
    model = pipeline.named_steps["model"]
    Xtr, Xte = transform(pipeline, X_train), transform(pipeline, X_test)
    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, Xtr)
    values = explainer(Xte)
    # random forests give one set of values per class, keep the "churn" class
    if values.values.ndim == 3:
        values = values[:, :, 1]
    return values, Xte


def shap_plots(values, title):
    shap.plots.bar(values, max_display=15, show=False)
    plt.title(f"Mean |SHAP| - {title}")
    save("09_shap_importance.png")

    shap.plots.beeswarm(values, max_display=15, show=False)
    plt.title(f"SHAP summary - {title}")
    save("10_shap_beeswarm.png")

    mean_abs = pd.Series(np.abs(values.values).mean(axis=0), index=values.feature_names)
    return mean_abs.sort_values(ascending=False)


def shap_one_customer(values, index, title):
    shap.plots.waterfall(values[index], max_display=12, show=False)
    plt.title(title)
    save("11_shap_waterfall_customer.png")


def lime_one_customer(pipeline, X_train, X_test, row):
    model = pipeline.named_steps["model"]
    Xtr, Xte = transform(pipeline, X_train), transform(pipeline, X_test)
    explainer = LimeTabularExplainer(
        Xtr.values,
        feature_names=list(Xtr.columns),
        class_names=["Stay", "Churn"],
        mode="classification",
        discretize_continuous=True,
        random_state=SEED,
    )
    exp = explainer.explain_instance(Xte.iloc[row].values, model.predict_proba, num_features=10)
    fig = exp.as_pyplot_figure(label=1)
    fig.set_size_inches(9, 5)
    plt.title("LIME - why this customer is predicted to churn")
    save("12_lime_customer.png")
    OUT.mkdir(parents=True, exist_ok=True)
    exp.save_to_file(str(OUT / "lime_customer.html"))
    return exp.as_list(label=1)


def main():
    meta = json.loads(META_PATH.read_text())
    best = joblib.load(MODEL_PATH)
    xgb = joblib.load(MODEL_DIR / "candidates" / "XGBoost.joblib")
    X_train, X_test, y_train, y_test = split_data(load_data())

    importance = tree_importance(xgb, X_test)

    # SHAP on the model that is actually served by the API
    values, Xte = shap_values(best, X_train, X_test)
    shap_rank = shap_plots(values, meta["model_name"])

    proba = best.predict_proba(X_test)[:, 1]
    row = int(np.argmax(proba))
    customer = X_test.iloc[row]
    shap_one_customer(values, row, f"Highest-risk test customer ({proba[row]:.0%} churn)")
    lime_rank = lime_one_customer(best, X_train, X_test, row)

    summary = {
        "explained_model": meta["model_name"],
        "xgboost_importance_top10": importance.head(10).round(4).to_dict(),
        "shap_mean_abs_top10": shap_rank.head(10).round(4).to_dict(),
        "customer": {"churn_probability": round(float(proba[row]), 4), "actual": int(y_test.iloc[row]), **customer.to_dict()},
        "lime_top10": [[name, round(weight, 4)] for name, weight in lime_rank],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "explain_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    setup_mlflow()
    with mlflow.start_run(run_name="Explainability"):
        mlflow.log_param("explained_model", meta["model_name"])
        for name in ["08_feature_importance_xgboost", "09_shap_importance", "10_shap_beeswarm",
                     "11_shap_waterfall_customer", "12_lime_customer"]:
            mlflow.log_artifact(str(FIGURES / f"{name}.png"))
        mlflow.log_artifact(str(OUT / "lime_customer.html"))

    print(f"explained model: {meta['model_name']}")
    print("top SHAP features:")
    for name, value in shap_rank.head(8).items():
        print(f"  {name:<40} {value:.4f}")
    print(f"\nhighest-risk test customer: {proba[row]:.1%} churn (actual: {'left' if y_test.iloc[row] else 'stayed'})")
    print("LIME reasons:")
    for name, weight in lime_rank[:6]:
        print(f"  {name:<45} {weight:+.4f}")


if __name__ == "__main__":
    main()
