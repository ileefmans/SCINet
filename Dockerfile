FROM python:3.7

COPY skinConditionDetect/datahelper.py /
COPY skinConditionDetect/preprocess.py /
COPY skinConditionDetect/facealign.py /
COPY skinConditionDetect/run_match.py /
COPY requirements.txt /
COPY skinConditionDetect/pickle/simple_train_dict.pkl /
COPY skinConditionDetect/shape_predictor_68_face_landmarks.dat /
COPY skinConditionDetect/mmod_human_face_detector.dat /

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
RUN apt-get update && apt-get -y install cmake protobuf-compiler
RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "run_match.py"]
