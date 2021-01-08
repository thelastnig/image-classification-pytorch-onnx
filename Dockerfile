FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN python -m pip install -U pip setuptools \
 && pip install python_pachyderm Pillow tqdm mlflow

COPY . /app
ENV PYTHONPATH "${PYTHONPATH}:/app"
WORKDIR /app
ENTRYPOINT ["python", "/app/train.py"]
