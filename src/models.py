from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import SEED
from src.features import build_preprocessor

# about 2.8 customers stay for every 1 that leaves -> used for scale_pos_weight
NEG_POS_RATIO = 2.8


def candidates():
    # model + grid for each algorithm, small grids so it runs fast on my laptop
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [6, 10, None],
                "model__min_samples_leaf": [1, 5],
                "model__class_weight": [None, "balanced_subsample"],
            },
        ),
        "XGBoost": (
            XGBClassifier(random_state=SEED, n_jobs=1, eval_metric="logloss", tree_method="hist"),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__scale_pos_weight": [1.0, NEG_POS_RATIO],
            },
        ),
    }


def make_pipeline(model):
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def evaluate(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
    }
