from flask import Flask, json, jsonify, request
from classifier import get_pred

app = Flask(__name__)

@app.route("/pred-digit", methods = ["POST"])
def pred_data():
    image = request.files.get("digit")
    prediction = get_pred(image)
    return jsonify({
        "prediction" : prediction
    }, 200)

if __name__ == "__main__":
    app.run(debug=True)