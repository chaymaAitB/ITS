FROM tensorflow/tensorflow:2.10.0-gpu

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    cmake \
    build-essential \
    python3-dev \
    libboost-all-dev \
    nano \
    sumo \
    sumo-tools \
    sumo-doc && \
    rm -rf /var/lib/apt/lists/*

# SUMO environment
ENV SUMO_HOME=/usr/share/sumo
ENV PYTHONPATH=/usr/share/sumo/tools

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Python ML stack
RUN python3 -m pip install \
    numpy \
    cython \
    matplotlib \
    pandas \
    seaborn \
    scipy \
    tqdm \
    gym \
    torch

# CityFlow
RUN git clone https://github.com/cityflow-project/CityFlow.git /CityFlow

WORKDIR /CityFlow

RUN python3 -m pip install .

WORKDIR /workspace

CMD ["/bin/bash"]
