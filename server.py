from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/run-python', methods=['GET'])
def run_python():
    return jsonify({"message": "Python script executed successfully!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)