import pickle
from flask import Flask, render_template_string, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the GaussianNB model trained with scikit-learn 1.9.1
with open('naive.pkl', 'rb') as f:
  model = pickle.load(f)

# HTML Layout and CSS Style embedded directly inside app.py
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Retention Predictor</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .card {
            border: none;
            border-radius: 1rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        }
        .btn-primary {
            background-color: #4e73df;
            border: none;
            transition: background-color 0.2s;
        }
        .btn-primary:hover {
            background-color: #2e59d9;
        }
        .form-label {
            color: #495057;
        }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-7">
                <div class="card p-4 p-md-5 bg-white">
                    <div class="text-center mb-4">
                        <h2 class="fw-bold text-primary">Employee Retention Predictor</h2>
                        <p class="text-muted">Enter employee details below to analyze professional outlook.</p>
                    </div>

                    <form action="/predict" method="POST">
                        <div class="row g-3">
                            <!-- Education -->
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Education</label>
                                <select class="form-select" name="Education" required>
                                    <option value="Bachelors">Bachelors</option>
                                    <option value="Masters">Masters</option>
                                    <option value="PHD">PHD</option>
                                </select>
                            </div>

                            <!-- Joining Year -->
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Joining Year</label>
                                <input type="number" class="form-control" name="JoiningYear" placeholder="e.g. 2018" required>
                            </div>

                            <!-- City -->
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">City</label>
                                <select class="form-select" name="City" required>
                                    <option value="Bangalore">Bangalore</option>
                                    <option value="New Delhi">New Delhi</option>
                                    <option value="Pune">Pune</option>
                                </select>
                            </div>

                            <!-- Payment Tier -->
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Payment Tier</label>
                                <select class="form-select" name="PaymentTier" required>
                                    <option value="1">Tier 1</option>
                                    <option value="2">Tier 2</option>
                                    <option value="3">Tier 3</option>
                                </select>
                            </div>

                            <!-- Age -->
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Age</label>
                                <input type="number" class="form-control" name="Age" placeholder="e.g. 28" required>
                            </div>

                            <!-- Gender -->
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Gender</label>
                                <select class="form-select" name="Gender" required>
                                    <option value="Male">Male</option>
                                    <option value="Female">Female</option>
                                </select>
                            </div>

                            <!-- Ever Benched -->
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Ever Benched?</label>
                                <select class="form-select" name="EverBenched" required>
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                </select>
                            </div>

                            <!-- Experience In Current Domain -->
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Experience in Current Domain (Years)</label>
                                <input type="number" step="0.1" class="form-control" name="ExperienceInCurrentDomain" placeholder="e.g. 3" required>
                            </div>
                        </div>

                        <div class="d-grid mt-4">
                            <button type="submit" class="btn btn-primary btn-lg shadow-sm">Predict Result</button>
                        </div>
                    </form>

                    {% if prediction_text %}
                    <div class="alert alert-info mt-4 text-center shadow-sm" role="alert">
                        <h4 class="alert-heading mb-1 fw-bold">{{ prediction_text }}</h4>
                        <p class="mb-0 text-secondary">{{ confidence }}</p>
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""


@app.route('/')
def home():
  return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
  try:
    # 1. Retrieve the 8 required features from form input[cite: 3]
    education_str = request.form['Education']
    joining_year = float(request.form['JoiningYear'])
    city_str = request.form['City']
    payment_tier = int(request.form['PaymentTier'])
    age = float(request.form['Age'])
    gender_str = request.form['Gender']
    ever_benched_str = request.form['EverBenched']
    experience = float(request.form['ExperienceInCurrentDomain'])

    # 2. Map categorical fields to numerical equivalents
    education_map = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}
    city_map = {'Bangalore': 0, 'New Delhi': 1, 'Pune': 2}
    gender_map = {'Male': 0, 'Female': 1}
    benched_map = {'No': 0, 'Yes': 1}

    education = education_map.get(education_str, 0)
    city = city_map.get(city_str, 0)
    gender = gender_map.get(gender_str, 0)
    ever_benched = benched_map.get(ever_benched_str, 0)

    # 3. Create feature vector matching model's expected input structure[cite: 3]
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

    # 4. Predict using the model
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max() * 100

    result_text = (
        'Likely to Leave / Change Job'
        if prediction == 1
        else 'Likely to Stay'
    )

    return render_template_string(
        HTML_TEMPLATE,
        prediction_text=f'Prediction: {result_text}',
        confidence=f'Confidence: {probability:.2f}%',
    )

  except Exception as e:
    return render_template_string(
        HTML_TEMPLATE, prediction_text=f'Error in processing: {str(e)}'
    )


if __name__ == '__main__':
  app.run(debug=True)
