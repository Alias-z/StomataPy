Installation
============

The installation scenarios depending on your setup.

Use Case 1: Google Colab (No Local GPU)
----------------------------------------

For users who want to try StomataPy without installing anything locally:

.. raw:: html

   <a target="_blank" href="https://colab.research.google.com/github/Alias-z/StomataPy/blob/master/StomataPy_demo.ipynb">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

Use Case 2: Windows with NVIDIA GPU
------------------------------------

For local Windows installation with GPU acceleration:

Install with `UV <https://docs.astral.sh/uv/getting-started/installation/>`_:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/Alias-z/StomataPy.git
   cd StomataPy

   # Install basic dependencies
   uv venv venv --python 3.10
   uv sync --link-mode=copy

   # Install PyTorch with CUDA
   uv pip install torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

   # Install OpenMMLab dependencies
   uv run -m mim install mmengine==0.10.4 mmcv==2.1.0 mmpretrain --trusted-host download.openmmlab.com --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

   # Install submodules
   uv pip install -e ./sam-hq -e ./mmsegmentation -e ./mmdetection --no-build-isolation

   # Install xformers for inference acceleration
   uv pip install xformers==0.0.23 


Use Case 3: Linux Server with NVIDIA GPU
-----------------------------------------

For headless server deployment:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/Alias-z/StomataPy.git
   cd StomataPy

   # Build and run with Docker (choose more GPUs for training)
   docker build -t stomatapy .
   docker run --gpus '"device=0,1,2,3"' -v $(pwd)/data:/workspace/data stomatapy

