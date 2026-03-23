import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import load

# Load env content to env variables
load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()

DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=(
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    )
)

# Load the trained model only once (module - level cache) -- do not load in the function
model = load(MODEL_PATH)
logging.info("Model Loaded Successfully")

def predict(input_data:dict):
    df = pd.DataFrame([input_data])

    # Get Prediction
    prediction = int(model.predict(df)[0])

    # Get Prediction Probability
    probability = float(model.predict_proba(df)[0][1])

    logging.info(f"Model Provided Prediction: {prediction}")

    return {
        "Prediction": prediction,
        "Probability": probability
    }