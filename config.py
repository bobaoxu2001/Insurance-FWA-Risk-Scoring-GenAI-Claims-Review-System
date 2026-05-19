import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
DATA_DOCUMENTS = os.path.join(BASE_DIR, "data", "documents")
OUTPUTS_FIGURES = os.path.join(BASE_DIR, "outputs", "figures")
OUTPUTS_MODELS = os.path.join(BASE_DIR, "outputs", "models")
OUTPUTS_REPORTS = os.path.join(BASE_DIR, "outputs", "reports")
OUTPUTS_REVIEWS = os.path.join(BASE_DIR, "outputs", "sample_reviews")

N_CLAIMS = 5000
RANDOM_SEED = 42
FRAUD_BASE_RATE = 0.08
HIGH_RISK_THRESHOLD = 0.6
