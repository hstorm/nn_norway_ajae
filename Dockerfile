# This is tensorflow version used by Alex for training the model
# FROM tensorflow/tensorflow:2.3.0-gpu
# FROM tensorflow/tensorflow:latest-gpu

# FROM nvcr.io/nvidia/cuda:10.1-cudnn8-runtime-ubi7
FROM nvcr.io/nvidia/cuda:10.1-cudnn7-runtime-ubi7

RUN yum install -y rh-python36-python-pip.noarch

# RUN scl enable rh-python36 bash
RUN scl enable rh-python36 -- pip3 install --upgrade pip

# RUN mkdir src
# WORKDIR src/
# COPY . .

RUN mkdir nn_norway
COPY . nn_norway/
WORKDIR nn_norway/

# RUN pip3 install jupyter
# RUN pip3 install -r requirements_docker.txt
RUN scl enable rh-python36 -- pip3 install --no-cache-dir -r ./requirements_docker.txt

# RUN pip3 install -r requirements_docker.txt


# scl enable rh-python36 -- python
# scl enable rh-python36 -- python src/models/nn_model.py
# import tensorflow as tf
# tf.config.list_physical_devices('GPU')

# # This is the docker file used by Alex for training the model
# FROM tensorflow/tensorflow:2.2.0-gpu
# # CMD [ "mkdir", "MODEL" ]
# # Copy ./MODEL /MODEL
# # WORKDIR /MODEL

# # Set the folder in which venv creates the environment
# ENV WORKON_HOME /.venvs

# # Install & use pipenv
# # COPY Pipfile Pipfile.lock ./
# COPY Pipfile  ./
# RUN python -m pip install --upgrade pip
# RUN pip install pipenv 
# RUN pipenv install --dev

# Install application into container
# COPY . .
# WORKDIR /app
# COPY . /app
