#
FROM python:3.8

#
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt
ENV PORT=$port

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN python -m nltk.downloader punkt
EXPOSE $PORT

#
COPY ./app /code/app

#

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
#CMD uvicorn app.main:app --host 0.0.0.0 --port 80
#CMD ["uvicorn app.main:app --host 0.0.0.0 -p\ :$PORT"]
