import requests
from flask import Flask, jsonify, send_from_directory, request

app = Flask(__name__, static_folder="static")

BACKEND_URL = "http://localhost:6000/predict"

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.json.get("text", "")
    try:
        response = requests.post(BACKEND_URL, json={"text": user_input})
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=3020, debug=True)
