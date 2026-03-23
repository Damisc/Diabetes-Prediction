import os 
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    recall_score,
    f1_score
)

def train_model():
    try:
        # Load env file content to env vars
        load_dotenv()

        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT")).resolve()

        DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_DIR") / os.getenv("DATASET_NAME")
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
        LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

        TARGET_COL = os.getenv("TARGET_COL")
        TEST_SIZE = float(os.getenv("TEST_SIZE"))
        RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )

        # Load Data
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset Loaded With Shape: {df.shape}")

        # Seperte X and y
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        # Tarain Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y
        )



        # changing the 0 values to NAN for selected columns ----> Median imutation
        zero_as_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

        numerical_feature = X_train.select_dtypes(include=[np.number]).columns.tolist()


        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(missing_values=0, strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        preprocess = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_feature)
            ]
        )

        # Best Model Parameter from Notebook tuning
        svc_best = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", SVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    C=10000,
                    gamma=0.001,
                    probability=True
                ))
            ]
        )
        svc_best.fit(X_train, y_train)
        logging.info("Model Training Completed")

        # Evaluation
        y_train_pred = svc_best.predict(X_train)
        y_test_pred = svc_best.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_recall = recall_score(y_train, y_train_pred)
        test_recall = recall_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        logging.info(f"Train Accuracy: {train_acc:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
        logging.info(f"Test Accuracy: {test_acc:.4f} | Recall: {test_recall:.4f} | F1: {test_f1:.4f}")

        logging.info("Train Classification Report:\n"+ classification_report(y_train, y_train_pred))
        logging.info("Test Classification Report\n"+ classification_report(y_test, y_test_pred))

        # save trained model
        dump(svc_best, MODEL_PATH)
        logging.info(f"Model Saved to {MODEL_PATH}")
        logging.info("Model Training Completed.")

    except Exception as e:
        print(f"Training Failed: {e}")
        logging.exception(f"Training Scrip Failed: {e}")
        raise

if __name__ == "__main__":
    train_model()