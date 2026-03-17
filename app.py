#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st
import requests

st.title("Mobile Addiction Prediction")

age = st.number_input("Age")
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
daily_screen_time = st.number_input("Daily Screen Time (hours)")
social_media_hours = st.number_input("Social Media Hours")
gaming_hours = st.number_input("Gaming Hours")
sleep_hours = st.number_input("Sleep Hours")
work_hours = st.number_input("Work Hours")
exercise_hours = st.number_input("Exercise Hours")
device_usage_years = st.number_input("Device Usage Years")
app_usage_count = st.number_input("App Usage Count")
notification_count = st.number_input("Notification Count")
data_usage_mb = st.number_input("Data Usage (MB)")
stress_level = st.number_input("Stress Level")
mood_score = st.number_input("Mood Score")

if st.button("Predict"):
    data = {
        "age": age,
        "gender": gender,
        "daily_screen_time": daily_screen_time,
        "social_media_hours": social_media_hours,
        "gaming_hours": gaming_hours,
        "sleep_hours": sleep_hours,
        "work_hours": work_hours,
        "exercise_hours": exercise_hours,
        "device_usage_years": device_usage_years,
        "app_usage_count": app_usage_count,
        "notification_count": notification_count,
        "data_usage_mb": data_usage_mb,
        "stress_level": stress_level,
        "mood_score": mood_score
    }

    response = requests.post(
        "https://mobile-phone-usage-ml-project.onrender.com/predict", 
        json=data
    )

    prediction = response.json()["prediction"]

    if prediction == 1:
        st.success("Prediction: Not Addicted")
    else:
        st.success("Prediction: Addicted")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




