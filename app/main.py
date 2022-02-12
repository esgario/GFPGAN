import os
import cv2
import numpy as np

from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from app.inference import run_inference
from app.utils import get_base64_image, reshape_image

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
limiter = Limiter(app, key_func=get_remote_address)


@app.route('/')  # Homepage
def home():
    return render_template('index.html', data={}, error_msg='')


@app.route("/predict", methods=["POST"])
@limiter.limit("3/minute")
@cross_origin()
def predict():
    try:
        filestr = request.files['file'].read()
        assert filestr, "No image selected"
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        image = reshape_image(image)

        bgenh = request.form.getlist('bgenh')
        enable_realesrgan = any([i == 'on' for i in bgenh])

    except Exception as e:
        return render_template('index.html', data={}, error_msg=e.__str__())

    restored_img, cropped_faces, restored_faces = run_inference(image, enable_realesrgan)

    data = {
        "original_img": get_base64_image(image),
        "restored_img": get_base64_image(restored_img),
        "faces": [(get_base64_image(x), get_base64_image(y)) for x, y in zip(cropped_faces, restored_faces)]
    }

    return render_template('index.html', data=data, error_msg='')  # rendering the predicted result


def page_too_many_requests(e):
    msg = 'Too many requests per minute. Wait a while and try again.'
    return render_template('index.html', data={}, error_msg=msg)


def init():
    app.register_error_handler(429, page_too_many_requests)


init()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))