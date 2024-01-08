from flask import Flask, jsonify, request

from lstm import run_predict
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return jsonify("Hello world")

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    print("payload: ",payload)
    trainData = payload['trainData']
    myAvgScores =  payload['myAvgScores']
    if not trainData or not myAvgScores:
        return jsonify({'error': 'Không có dữ liệu được gửi lên!'}), 400
    try:
       redicted_jobs = run_predict(trainData,myAvgScores)
       return jsonify(redicted_jobs)
    except Exception as e:
        print("Error: ",e)
        return jsonify({'error': 'Lỗi dự đoán!'}), 400

if __name__ == '__main__':
    app.run(debug=True)
