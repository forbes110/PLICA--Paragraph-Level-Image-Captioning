# ADL-Final-Project

## Environment Setup
```shell
(optional) conda create -n <name> python=3.9
pip install -r requirements.txt
```
## Dataset loading & preprocessing

```shell
bash ./scripts/download.sh
```
## Training
```shell
// for test only
python train_imacap.py --dataset_name maderix/flickr_bw_rgb
```
## Inferencing

