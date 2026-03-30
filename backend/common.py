import numpy as np

# this function will b eneeded when calling the training and the prediction script
def replace_zeros_with_nan(X, cols):
    X=X.copy()
    cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols:
        if col in X.columns:
            X[col] = X[col].replace(0, np.nan)
    return X

