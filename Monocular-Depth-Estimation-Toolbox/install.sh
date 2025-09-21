conda create -n vitsplit-mde python=3.7
conda activate vitsplit-mde

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12

pip install future tensorboard
pip install yapf==0.40.1
pip install setuptools==59.5.0
pip install ipdb