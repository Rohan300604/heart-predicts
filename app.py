from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the logistic regression model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Predict route accessed")  # Check if the route is accessed

        # Get form data
        male = int(request.form['male'])
        age = int(request.form['age'])
        currentSmoker = int(request.form['currentSmoker'])
        cigsPerDay = int(request.form['cigsPerDay'])
        BPMeds = int(request.form['BPMeds'])
        prevalentStroke = int(request.form['prevalentStroke'])
        prevalentHyp = int(request.form['prevalentHyp'])
        diabetes = int(request.form['diabetes'])
        totChol = float(request.form['totChol'])
        sysBP = float(request.form['sysBP'])
        diaBP = float(request.form['diaBP'])
        BMI = float(request.form['BMI'])
        heartRate = int(request.form['heartRate'])
        glucose = float(request.form['glucose'])

        # Create a feature array for prediction
        features = np.array([[male, age, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, 
                              prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose]])

        print("Features array:", features)  # Check if features are correctly created

        # Predict using the loaded model
        prediction = model.predict(features)[0]
        print("Prediction successful:", prediction)  # Check if prediction was made

        # Render the result template with the prediction
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        print("Error:", e)  # Log any errors to the console
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
