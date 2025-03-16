from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/run-python', methods=['POST'])
def run_python():
    data = request.get_json()  # Get JSON data from the request
    user_text = data.get('text', '')  # Extract the 'text' field
    return jsonify({"message": user_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
