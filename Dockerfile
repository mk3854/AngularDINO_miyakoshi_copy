#
# NVIDIA NGC コンテナを使う
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-05.yhtml
# 動作確認されているので、独自にコンテナをビルドするより手間が少ないです
#
FROM nvcr.io/nvidia/pytorch:23.12-py3

#
# settings
#
RUN apt-get update && apt-get -y upgrade && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-dev tzdata fonts-ipafont htop \
    && ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/local/src/*
ENV TZ=Asia/Tokyo

#
# ClearML
#
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir \
     clearml==1.13.1 \
     boto3

#
# opencv...
#
RUN pip3 install --no-cache-dir \
    "opencv-python-headless<4.3" \
    pyyaml \
    scipy \
    tensorboard==1.14.0 \
    termcolor \
    timm==0.4.12 \
    yacs