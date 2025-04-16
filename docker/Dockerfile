FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
ENV https_proxy http://proxy.ethz.ch:3128/
ENV http_proxy http://proxy.ethz.ch:3128/

RUN groupadd -g 1009 hongyuan && \
    useradd -m -s /bin/bash -u 1009 -g 1009 hongyuan && \
    chown -R hongyuan:hongyuan /home/hongyuan

RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    zip \
    unzip \
    libgl1-mesa-glx \
    git

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip \
    && pip install tqdm openpyxl pandas opencv-python scikit-image albumentations more-itertools segment-anything-hq\
    && pip install ftfy regex imgviz mahotas pycocotools matplotlib onnxruntime onnx timm pyyaml cellpose\
    && pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
    && pip install deepspeed fairscale hdbscan nose transformers\
    && pip install -U openmim  --upgrade ipykernel

RUN mim install mmengine mmdet mmsegmentation mmpretrain

# jupyter notebook
#RUN pip install jupyter
#CMD jupyter notebook --ip 0.0.0.0 --port 7777 --no-browser --allow-rootd

# jupyter lab
RUN pip install jupyterlab

USER hongyuan
CMD jupyter-lab --ip 0.0.0.0 --port 7788 --no-browser --allow-root




