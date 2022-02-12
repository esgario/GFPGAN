FROM python:3.7.12-slim-buster

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN apt-get update
RUN apt-get install wget -y
RUN mkdir -p experiments/pretrained_models
RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth -O experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth

RUN python -m pip install --upgrade pip
RUN python -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install basicsr
RUN pip install facexlib
RUN pip install -r requirements.txt
RUN pip install realesrgan
RUN python setup.py develop

RUN apt-get install ffmpeg libsm6 libxext6  -y

# Force download pre-trained model.
RUN python -c "from facexlib.utils.face_restoration_helper import FaceRestoreHelper; FaceRestoreHelper(1)"
RUN mkdir -p /usr/local/lib/python3.7/site-packages/realesrgan/weights
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -O /usr/local/lib/python3.7/site-packages/realesrgan/weights/RealESRGAN_x2plus.pth

EXPOSE 8080

ENTRYPOINT python app/main.py
