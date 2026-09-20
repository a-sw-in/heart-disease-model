
from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI()
Scalar=joblib.load('scalar.pkl')
model1=joblib.load('model.pkl')
class userData(BaseModel):
  male:int
  age:int
  currentSmoker:int
  cigsPerDay:int
  BPMeds:int
  prevalentStroke:int
  prevalentHyp:int
  diabetes:int
  totChol:int
  sysBP:int
  diaBP:int
  BMI:int
  heartRate:int
  glucose:int


@app.post("/predict")
def predict(data:userData):
  features=[[
      data.male,
      data.age,
      data.currentSmoker,
      data.cigsPerDay,
      data.BPMeds,
      data.prevalentStroke,
      data.prevalentHyp,
      data.diabetes,
      data.totChol,
      data.sysBP,
      data.diaBP,
      data.BMI,
      data.heartRate,
      data.glucose
  ]]
  features = Scalar.transform(features)

  prediction = model1.predict(features)[0]
  probability = model1.predict_proba(features)[0]
  prediction = int(prediction)
  risk = "High" if prediction == 1 else "Low"
  probability = [round(float(value), 5) for value in probability]

  print("Prediction:", prediction)
  print("Probability:", probability)
  return {"prediction": prediction, "risk": risk, "probability": probability}
