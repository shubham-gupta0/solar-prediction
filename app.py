from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import time

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('best_gbr.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Get user input from the form
    wind_speed = float(request.form['wind_speed'])
    sunshine = float(request.form['sunshine'])
    air_pressure = float(request.form['air_pressure'])
    radiation = float(request.form['radiation'])
    air_temperature = float(request.form['air_temperature'])
    relative_humidity = float(request.form['relative_humidity'])
    
    # Preprocess the input data if necessary
    input_data = np.array([[wind_speed, sunshine, air_pressure, radiation, air_temperature, relative_humidity]])
    
    # Make predictions using the model
    prediction = model.predict(input_data)

    # Return the prediction to the user
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
