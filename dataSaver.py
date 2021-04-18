import os
from datetime import datetime
from waitress import serve

from flask import Flask, request
from flask_cors import CORS
from utils import make_resp

app = Flask(__name__)
app.config.from_object(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app)

@app.route('/')
def index():
    return '<a href = "http://abdyabdya.duckdns.org:27500/first">fist</a>'


@app.route('/first')
def firstTemp():
    return '''<!DOCTYPE html>
<head>
    <meta charset="utf-8">
</head>
<html>
<body>
<video id="webcam" width="720" height="400" autoplay></video>
<canvas id="overlay" hidden></canvas>

<p id="explanation">
    Смотрите на курсор и нажмите пробел для отправки фото и координат на сервер.
    Сделайте несколько снимков, с курсором мыши в разных местах экрана
</p>
<style>
    #explanation {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 24pt;
    }
</style>


<script src="https://code.jquery.com/jquery-3.3.1.min.js"></script>

<script>

        const video = $('#webcam')[0];

        const canvas = $('#overlay')[0];

        function onStreaming(stream) {
            video.srcObject = stream;
        }



        navigator.mediaDevices
            .getUserMedia({
                video: true,
            })
            .then(onStreaming);

        function captureExample(){
            const eyesCC = canvas.getContext('2d');
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight
            eyesCC.drawImage(video, 0, 0);
        }


        $('body').keyup(function (event) {
            if (event.keyCode == 32) {
                captureExample();
                sendModel()
                event.preventDefault();
                return false;
            }
        });

        const mouse = {
            x: 0,
            y: 0,

            handleMouseMove: function(event) {
                // Get the mouse position and normalize it to [-1, 1]
                mouse.x = (event.clientX / $(window).width()) * 2 - 1;
                mouse.y = (event.clientY / $(window).height()) * 2 - 1;
            },
        };

        document.onmousemove = mouse.handleMouseMove;



        function sendModel() {
            var arr = {photo: canvas.toDataURL("image/jpeg") , x: mouse.x, y:mouse.y};
            $.ajax({
                url: 'http://127.0.0.1:5000/myNeuron',
                type: 'POST',
                data: JSON.stringify(arr),
                contentType: "application/json; charset=utf-8",
                dataType: "json",
                success: function (msg) {
                    alert(msg);
                }
            });
            // currentModel.save("http://127.0.0.1:5000/model")
        }

</script>

<!--<script src="node_modules/jszip/dist/jszip.min.js"></script>-->
</body>
</html>
'''


@app.route('/myNeuron', methods=['POST'])
def post_couriers():
    file = request.json
    if file:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], str(datetime.now())+".txt"), 'w') as inp:
            inp.write(str(file))

    return make_resp(file,200)


if __name__ == '__main__':
    for i, j in [1,2,3]+[4,5,6]:
        print(f"{i},{j}")
    # app.run()
    # serve(app, host="0.0.0.0", port="8080")