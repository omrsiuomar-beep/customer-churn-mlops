from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"
FIGURES = ROOT / "reports" / "figures"

TRACKING_URI = "sqlite:///" + (ROOT / "mlflow.db").as_posix()
ARTIFACT_DIR = ROOT / "mlartifacts"
EXPERIMENT = "Customer_Churn"

SEED = 42
TEST_SIZE = 0.2
TARGET = "Churn"

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURES = NUMERIC + CATEGORICAL
