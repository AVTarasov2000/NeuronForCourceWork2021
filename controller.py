import os

from flask import Flask, request, send_file
from utils import make_resp
from flask_cors import CORS
from predict_service import predict

app = Flask(__name__)
app.config.from_object(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

@app.route('/')
def index():
    return 'hello'


@app.route('/model', methods=['POST'])
def post_couriers():
    # data = request
    print(len(request.files))
    file = request.files
    if file:
        # filename = secure_filename(file.filename)
        file['model.json'].save(os.path.join(app.config['UPLOAD_FOLDER'], "filename"))
        file['model.weights.bin'].save(os.path.join(app.config['UPLOAD_FOLDER'], "filename.bin"))
    return make_resp({},200)


@app.route('/predict', methods=['POST'])
def post_prediction():
    data = request.json
    x = predict(data['photo'])
    return {'prediction': str(x)}


@app.route('/model')
def model():
    with open("./model_2/", "r") as res:
        return send_file(res)


@app.route('/weigts')
def weights():
    with open("./model_2/variables", "r") as res:
        return send_file(res)

if __name__ == '__main__':
    app.run()