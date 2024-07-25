from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
# Load the scikit-learn model
model = joblib.load('./dementialModel.joblib')

def convert_to_standard_types(value):
    if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    elif isinstance(value, (np.float64, np.float32)):
        return float(value)
    else:
        return value

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the POST request
        input_data = request.json

        # Convert input data to a format compatible with the model
        input_array = [[
            input_data['M/F'],
            input_data['Age'],
            input_data['EDUC'],
            input_data['SES'],
            input_data['MMSE'],
            input_data['eTIV'],
            input_data['nWBV'],
            input_data['ASF']
        ]]

        # Make predictions using the loaded model
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[:, 1][0]

        # Convert to standard types
        prediction = convert_to_standard_types(prediction)
        probability = convert_to_standard_types(probability)

        # Return the results as JSON
        result = {'prediction': prediction, 'probability': probability}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()

# python -m flask --app ./app.py run
# python -m venv alzh
