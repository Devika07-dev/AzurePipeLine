from flask import Flask, request, jsonify
from flask.logging import create_logger
import logging

import pandas as pd
# from sklearn.externals import joblib
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
LOG = create_logger(app)
LOG.setLevel(logging.INFO)

def scale(payload):
    """Scales Payload"""

    LOG.info("Scaling Payload: %s payload")
    scaler = StandardScaler().fit(payload)
    scaled_adhoc_predict = scaler.transform(payload)
    return scaled_adhoc_predict

@app.route("/")
def home():
    html = "<h3>Sklearn Prediction Home</h3>"
    return html.format(format)

# TO DO:  Log out the prediction value
@app.route("/predict", methods=["POST"])
def predict():
    try:
        import os

        MODEL_PATH = os.path.join(
            os.path.dirname(_file_),
            "Housing_price_model",
            "GradientBoostingRegressor.joblib"
        )

        clf = joblib.load(MODEL_PATH)

        json_payload = request.get_json(force=True)
        inference_payload = pd.DataFrame(json_payload)

        prediction = clf.predict(inference_payload)
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500        
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
