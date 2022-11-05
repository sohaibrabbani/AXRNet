FROM tensorflow/tensorflow:1.5.0-gpu
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt update \
    && apt install -y htop python3-dev wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir -p root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n ml python=3.6

COPY . src/
RUN /bin/bash -c "cd src \
    && source activate ml \
    && apt update && apt install -y libsm6 libxext6 \
    && apt-get install -y libxrender-dev \
    && pip install -r requirements.txt"