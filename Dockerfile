FROM jjanzic/docker-python3-opencv:opencv-4.0.0
COPY ./requirements.txt /tmp/
RUN pip install -U pip && pip install -r /tmp/requirements.txt
COPY . ./app
WORKDIR app
ENTRYPOINT python app.py