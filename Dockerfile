FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    unzip \
    build-essential \
    patchelf \
    ffmpeg \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglfw3 \
    libosmesa6-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN mkdir -p /root/.mujoco && \
    cd /root/.mujoco && \
    wget https://www.roboti.us/download/mujoco200_linux.zip && \
    wget https://www.roboti.us/file/mjkey.txt && \
    unzip mujoco200_linux.zip && \
    mv mujoco200_linux mujoco200 && \
    rm mujoco200_linux.zip

ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco200
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}

RUN pip install --upgrade \
    pip==24.0 \
    setuptools==63.2.0 \
    wheel==0.38.4 \
    cython==0.29.36

# diffuser reqs

RUN pip install \
    -f https://download.pytorch.org/whl/torch_stable.html \
    torch==1.9.1+cu111

RUN pip install \
    numpy \
    gym==0.18.0 \
    mujoco-py==2.0.2.8 \
    matplotlib==3.3.4 \
    typed-argument-parser \
    scikit-image==0.17.2 \
    scikit-video==1.1.11 \
    gitpython \
    einops \
    pillow \
    tqdm \
    pandas \
    wandb \
    "flax>=0.3.5" \
    "jax<=0.2.21" \
    ray==2.0.0 \
    crcmod \
    google-api-python-client \
    cryptography \
    gdown>=4.6.0 \ 
    bullet_safety_gym

RUN pip install \
    git+https://github.com/Farama-Foundation/d4rl@f2a05c0d66722499bf8031b094d9af3aea7c372b#egg=d4rl \
    git+https://github.com/JannerM/doodad.git@janner



COPY . /workspace/

CMD ["/bin/bash"]
