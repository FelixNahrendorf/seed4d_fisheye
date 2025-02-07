# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

FROM carlasim/carla:0.9.14

USER root

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update \
    && apt install -y software-properties-common \
    && apt update && apt install -y python3.8 python3.8-distutils python3-pip apt-utils git wget psmisc tmux vulkan-utils xdg-user-dirs unzip zip

RUN python3.8 -m easy_install pip
RUN pip install --upgrade pip

RUN wget https://files.pythonhosted.org/packages/c1/a2/6e172f2cc17e6ad9f9853f18dd4f99c5d05d5a241ce2ba4a2daa73eff695/carla-0.9.14-cp38-cp38-manylinux_2_27_x86_64.whl 
RUN python3.8 -m pip install carla-0.9.14-cp38-cp38-manylinux_2_27_x86_64.whl
RUN python3.8 -m pip install jupyterlab matplotlib opencv-python scikit-image pandas tqdm pypng open3d tabulate

USER carla

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=3333","--allow-root","--no-browser"]

EXPOSE 5894

# BUILD: docker build -t seed4d .
# RUN: docker run --privileged --name seed4d --gpus all --net=host --ipc=host --shm-size=20g -v /tmp/.X11-unix:/tmp/.X11-unix:rw seed4d sleep infinity