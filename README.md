# Env config

```
mamba create -n stomatapy python=3.10 -y
mamba activate stomatapy
mamba install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install fairscale openmim==0.3.8 xformers==0.0.23 
mim install mmengine mmdet mmsegmentation mmpretrain
mamba install jupyter more-itertools openpyxl scikit-image ftfy imgviz mahotas pyqt pyyaml hdbscan nose  -y
mamba install segment-anything-hq onnxruntime onnx timm transformers sahi -y
   
mamba install roifile fastremap llvmlite -y
pip install cellpose[gui]
```
