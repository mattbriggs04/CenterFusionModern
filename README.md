# CenterFusionModern
This project is a modern, Python 3.12.11 + PyTorch 2.8.0 reproduction of [CenterFusion](https://github.com/mrnabati/CenterFusion) which implements [CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection](https://arxiv.org/abs/2011.04841)


## Installation Instructions
To use python 3.12.11 with pyenv, run in the root directory
```
pyenv install 3.12.11
pyenv local 3.12.11
```

Create a virtual environment
```
python3 -m venv .venv
```
Activate the virtual environment (MacOS / Linux)
```
source .venv/bin/activate
```
On Windows:
```
.\.venv\Scripts\activate
```

Install dependencies
```
cd CenterFusionModern
pip install -r requirements.txt
```

## Training a model
The jupyter notebook file located in `experiments/` is intended for use on Google Colab using a GPU. Using an A100 if available is suggested. This notebook contains how to train, test, and run 3D box decoding.

## Pretrained Models
The original CenterFusion pretrained models are rotten (404 errors). My own pretrained models may be coming soon.

## Modernization Simplifications
The main major simplification CenterFusionModern makes over the original repository is the reduction of using a Deformable Convolutional Network (DCN) as the fully convolutional backbone. This method is still supported, and if desired a DCN implementation can be cloned (clone to `src/lib/model/networks`) and used.

> **dla.py** in `src/lib/model/networks/` will attempt to import `from .DCNv2.dcn_v2 import DCN`. If this fails, it falls back to a purely convolutional network and assumes the --dla_node argument is set to 'conv'.
