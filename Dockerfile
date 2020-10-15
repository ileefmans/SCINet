FROM python:3.7

COPY utils/datahelper.py /
COPY utils/preprocess.py /
COPY utils/facealign.py /
COPY utils/testSCINet10.py /
COPY requirements.txt /
COPY utils/pickle/simple_train_dict.pkl /
COPY utils/shape_predictor_68_face_landmarks.dat /
COPY utils/mmod_human_face_detector.dat /

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
RUN apt-get update && apt-get -y install cmake protobuf-compiler
RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "testSCINet10.py"]
