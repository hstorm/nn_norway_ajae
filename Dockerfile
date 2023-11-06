
FROM nvcr.io/nvidia/cuda:10.1-cudnn7-runtime-ubi7

RUN yum install -y rh-python36-python-pip.noarch

RUN scl enable rh-python36 -- pip3 install --upgrade pip

RUN mkdir nn_norway
COPY . nn_norway/
WORKDIR nn_norway/

RUN scl enable rh-python36 -- pip3 install --no-cache-dir -r ./requirements_docker.txt

