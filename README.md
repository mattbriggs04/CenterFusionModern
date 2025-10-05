# CenterFusionModern
This project is a modern, Python 3.12.11 + PyTorch 2.8.0 reproduction of [https://github.com/mrnabati/CenterFusion](CenterFusion) which implements [https://arxiv.org/abs/2011.04841](CenterFusion: Center-based Radar and Camera Fusion for 3D Object Detection).


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
## Pretrained Models
The original CenterFusion pretrained models are rotten (404 errors). Coming soon: my own pretrained models using Google Colab cloud computing.
