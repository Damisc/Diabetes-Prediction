from fastapi import FastAPI
from pydantic import BaseModel

from backend.predictor import predict

app = FastAPI(
    title="Diabetes Prediction API"
)

# input schema matching training features
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Diabetes Prediction Endpoint
@app.post("/predict-diabetes")
def predict_diabetes(input_data: DiabetesInput):
    input_data = input_data.model_dump()
    result = predict(input_data=input_data)
    return {
        "Prediction":result["Prediction"],
        "Probability": result["Probability"],
        "Diagnosis": (
            "Diabetic"
            if result["Prediction"] == 1
            else "Non-Diabetic"
        )
    }