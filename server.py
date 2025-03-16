from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/run-python', methods=['GET'])
def run_python():
    return jsonify({"message": "Python script executed successfully!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)