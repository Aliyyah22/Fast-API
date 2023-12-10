# import libraries
from fastapi import FastAPI, Query
import uvicorn
import joblib
import pandas as pd

# create app object
app = FastAPI()

# create home
@app.get('/')
def home():
    return {'message': 'Welcome to Sepsis Prediction Using FastAPI'}

# load the model
model = joblib.load("best_model.pkl")

# Endpoint to get predictions
@app.post("/predict")
def predict_sepsis(
    Plasma_glucose: int = Query(..., description='Plasma Glucose'),
    Blood_Work_Result1: int = Query(..., description='Blood Work Result 1'),
    Blood_Pressure: int = Query(..., description='Blood Pressure'),
    Blood_Work_Result2: int = Query(..., description='Blood Work Result 2'),
    Blood_Work_Result3: int = Query(..., description='Blood Work Result 3'),
    Body_mass_index: int = Query(..., description='Body mass index'),
    Blood_Work_Result4: int = Query(..., description='Blood Work Result 4'),
    Age: int = Query(..., description='Age'),
    Insurance: int = Query(..., description='Insurance')
):

    # Convert input data to the format expected by the model
    input_data = pd.DataFrame([{
        'Plasma Glucose': Plasma_glucose,
        'Blood Work Result 1': Blood_Work_Result1,
        'Blood Pressure': Blood_Pressure,
        'Blood Work Result 2': Blood_Work_Result2,
        'Blood Work Result 3': Blood_Work_Result3,
        'Body mass index': Body_mass_index,
        'Blood Work Result 4': Blood_Work_Result4,
        'Age': Age,
        'Insurance': Insurance
    }])
    
    # make predictions
    prediction = model.predict(input_data)[0]

    sepsis_status = "Patient has sepsis" if prediction == 1 else "Patient does not have sepsis"

    # return prediction
    return {"prediction": sepsis_status}
