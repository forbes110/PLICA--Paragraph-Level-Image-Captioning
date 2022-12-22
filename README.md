# ADL-Final-Project

## Environment Setup(default in download.sh)
```shell
(optional) conda create -n <name> python=3.9
pip install -r requirements.txt
```

# Run Code From Scratch
## 1. Dataset loading, preprocessing and environment setup
```shell
bash scripts/download.sh
```
## 2. Training(or if you only need to eval/predict, skip this step)
```shell
## tune config in scripts/train.sh
bash scripts/train.sh
```
## Inferencing

