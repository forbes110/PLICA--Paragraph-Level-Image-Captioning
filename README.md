# PLEDGE: Paragraph LEvel image Description GEneration


Apply an end-to-end model structure (ViT + GPT) to describe images in more detail, rather than traditional image captioning that only provides object detections or a few simple sentences.

detail: https://www.dropbox.com/scl/fi/ybvzpnkkcy7lnn9jbkkl1/report.pdf?rlkey=grunab2372z90x0uj429x2n5o&dl=1

## Environment Setup(default in download.sh)
```shell
(optional) conda create -n <name> python=3.9
pip install -r requirements.txt
```

# Run Code From Scratch
## 1. Dataset loading, preprocessing and environment setup
before the following scripts, install java by yourself.
```shell
bash scripts/download.sh
```
## 2. Training(or if you only need to eval/predict, skip this step)
```shell
## tune config in scripts/train.sh
bash scripts/train.sh $output_model_path
```
## 3. Inference and evaluate
```shell
## tune config in scripts/predict.sh
bash scripts/predict.sh
```

