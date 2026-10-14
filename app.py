import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('naive.pkl', 'rb') as f:
  model = pickle.load(f)


@app.route('/')
def home():
  return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
  try:
    # 1. Retrieve form inputs
    education_str = request.form['Education']
    joining_year = float(request.form['JoiningYear'])
    city_str = request.form['City']
    payment_tier = int(request.form['PaymentTier'])
    age = float(request.form['Age'])
    gender_str = request.form['Gender']
    ever_benched_str = request.form['EverBenched']
    experience = float(request.form['ExperienceInCurrentDomain'])

    # 2. Encode categorical variables
    education_map = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}
    city_map = {'Bangalore': 0, 'New Delhi': 1, 'Pune': 2}
    gender_map = {'Male': 0, 'Female': 1}
    benched_map = {'No': 0, 'Yes': 1}

    education = education_map.get(education_str, 0)
    city = city_map.get(city_str, 0)
    gender = gender_map.get(gender_str, 0)
    ever_benched = benched_map.get(ever_benched_str, 0)

    # 3. Create feature array
    features = [[
        education,
        joining_year,
        city,
        payment_tier,
        age,
        gender,
        ever_benched,
        experience,
    ]]

    # 4. Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max() * 100

    result_text = (
        'Likely to Leave / Change Job'
        if prediction == 1
        else 'Likely to Stay'
    )

    # Custom theme style injected directly from Python backend if desired
    dynamic_style = (
        'background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);'
        if prediction == 1
        else 'background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);'
    )

    return render_template(
        'index.html',
        prediction_text=f'Prediction: {result_text}',
        confidence=f'Confidence: {probability:.2f}%',
        dynamic_style=dynamic_style,
    )

  except Exception as e:
    return render_template(
        'index.html', prediction_text=f'Error in processing: {str(e)}'
    )


if __name__ == '__main__':
  app.run(debug=True)
