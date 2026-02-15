from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import joblib, pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Load AI brain
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("features.pkl")

class Patient(BaseModel):
    age: int; sex: str; trestbps: int; chol: int; thalch: int

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html") as f: return f.read()

@app.post("/predict")
def predict(p: Patient):
    df = pd.DataFrame([p.dict()])
    df_enc = pd.get_dummies(df).reindex(columns=feature_cols, fill_value=0)
    scaled = scaler.transform(df_enc)
    return {"prediction": int(model.predict(scaled)[0])}