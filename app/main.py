import os
import cv2
import numpy as np

from flask import Flask, request, jsonify, make_response, render_template
from flask_cors import CORS, cross_origin

from app.inference import run_inference
from app.utils import get_base64_image, reshape_image

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app = Flask(__name__)


@app.route('/')  # Homepage
def home():
    return render_template('index.html', data={}, error_msg='')


@app.route("/predict", methods=["POST"])
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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))