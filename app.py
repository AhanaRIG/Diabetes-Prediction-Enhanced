from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the RandomForest model
with open('models/RandomForest.pkl', 'rb') as file:
    model = pickle.load(file)

# Step 2: Load the saved StandardScaler
with open('models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
    
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            input_values = [
                int(request.form['Pregnancies']),
                float(request.form['Glucose']),
                float(request.form['BloodPressure']),
                float(request.form['SkinThickness']),
                float(request.form['Insulin']),
                float(request.form['BMI']),
                float(request.form['DiabetesPedigreeFunction']),
                int(request.form['Age'])
            ]
            # input_array = np.array([input_values])
            # result = model.predict(input_array)[0]

            # Step 3: Convert to array and scale
            input_array = np.array([input_values])
            input_scaled = scaler.transform(input_array)

            # Step 3: Predict using scaled input
            result = model.predict(input_scaled)[0]

            prediction = "Diabetic" if result == 1 else "Not Diabetic"
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
