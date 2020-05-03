FROM tensorflow/tensorflow:2.2.0rc4
WORKDIR /app
ADD requirements.txt .
RUN pip install -r requirements.txt
ADD src src/.
ADD train.sh .
ADD config.gin .
CMD [ "bash" ]