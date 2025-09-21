conda create -n vitsplit-llava python=3.10 -y
conda activate vitsplit-llava
pip install --upgrade pip
pip install -e .

pip install -e ".[train]"
pip install flash-attn==2.6.3
