import json
import numpy as np
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify

app = Flask(__name__)

def predict_pdu_data(data):
    """
    Processes PDU data, predicts future values, and returns the prediction.

    Args:
        data (dict): The PDU data in a dictionary format.

    Returns:
        dict: A dictionary containing the predicted labels and values, or an error message.
    """
    time_unit = data.get("timeUnit")
    buckets = data.get("buckets", [])

    if not time_unit or not buckets:
        return {"error": "JSON data must include 'timeUnit' and 'buckets'."}

    pdu_fields = [
        "entityStatePduCount", "fireEventPduCount", "collisionPduCount",
        "detonationPduCount", "dataPduCount", "actionRequestPduCount",
        "startResumePduCount", "setDataPduCount", "designatorPduCount",
        "electromagneticEmissionsPduCount"
    ]

    historical_y_values = []
    historical_x_labels = []

    for bucket in buckets:
        total_pdus = sum(bucket.get(field, 0) for field in pdu_fields)
        historical_y_values.append(total_pdus)
        if time_unit == 'hour':
            historical_x_labels.append(bucket.get('hour'))
        elif time_unit == 'day':
            historical_x_labels.append(bucket.get('date'))
        elif time_unit == 'week':
            week_str = bucket.get('week', '')
            if week_str:
                historical_x_labels.append(week_str.split(' ')[0] + " " + week_str.split(' ')[1])

    if not historical_x_labels:
        return {"error": "No data points were found to analyze."}

    num_historical_points = len(historical_x_labels)
    num_to_predict = num_historical_points

    # --- Prediction using Linear Regression ---
    X_historical = np.arange(num_historical_points).reshape(-1, 1)
    y_historical = np.array(historical_y_values)

    model = LinearRegression()
    model.fit(X_historical, y_historical)

    future_x_indices = np.arange(num_historical_points, num_historical_points + num_to_predict).reshape(-1, 1)
    predicted_y = model.predict(future_x_indices)
    
    predicted_x_labels = [f"Pred. Point {i+1}" for i in range(num_to_predict)]

    return {
        "predicted_labels": predicted_x_labels,
        "predicted_values": predicted_y.tolist()
    }

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for Kubernetes and Docker.
    """
    return jsonify({"status": "healthy", "service": "cap-pdu-prediction"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive PDU JSON data and return predictions.
    """
    try:
        # Try to get JSON data, this will raise an exception for invalid JSON
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "Request must be JSON"}), 400
        prediction_result = predict_pdu_data(data)
        if "error" in prediction_result:
            return jsonify(prediction_result), 400
        return jsonify(prediction_result)
    except (ValueError, TypeError, UnicodeDecodeError) as e:
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Example of how to run the Flask app.
    # In a production environment, you would use a proper WSGI server.
    app.run(debug=True, port=5001)
