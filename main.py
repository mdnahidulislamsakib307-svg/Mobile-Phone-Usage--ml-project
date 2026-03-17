#!/usr/bin/env python
# coding: utf-8

# In[35]:


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib as jb

model = jb.load("LogisticRegression.pkl")

app = FastAPI(title="Mobile Addiction Prediction API")

class UserData(BaseModel):
    age: int
    gender: str
    daily_screen_time: float
    social_media_hours: float
    gaming_hours: float
    sleep_hours: float
    work_hours: float
    exercise_hours: float
    device_usage_years: float
    app_usage_count: float
    notification_count: float
    data_usage_mb: float
    stress_level: float
    mood_score: float

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/predict")
def predict(data: UserData):

    df = pd.DataFrame({
        "age": [data.age],
        "gender": [data.gender],
        "daily_screen_time": [data.daily_screen_time],
        "social_media_hours": [data.social_media_hours],
        "gaming_hours": [data.gaming_hours],
        "sleep_hours": [data.sleep_hours],
        "work_hours": [data.work_hours],
        "exercise_hours": [data.exercise_hours],
        "device_usage_years": [data.device_usage_years],
        "app_usage_count": [data.app_usage_count],
        "notification_count": [data.notification_count],
        "data_usage_mb": [data.data_usage_mb],
        "stress_level": [data.stress_level],
        "mood_score": [data.mood_score]
    })

    
    prediction = model.predict(df)

    return {"prediction": int(prediction[0])}  

