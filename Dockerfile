FROM python:3.7

COPY skinConditionDetect/datahelper2.py /
COPY skinConditionDetect/preprocess.py /
COPY skinConditionDetect/dockerexample.py /
COPY requirements.txt /
COPY skinConditionDetect/pickle/simple_train_dict.pkl /

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y
RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "dockerexample.py"]


