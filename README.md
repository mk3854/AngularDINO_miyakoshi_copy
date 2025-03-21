# AngularDINO: Semi-Supervised Anomaly Detection via Self-Distillation with Dual Angular Margins
This is a pytorch implementation of the following paper [[arXiv]](https://arxiv.org/abs/2405.18929):
```
@misc{takahashi2024deep,
      title={Deep Positive-Unlabeled Anomaly Detection for Contaminated Unlabeled Data}, 
      author={Hiroshi Takahashi and Tomoharu Iwata and Atsutoshi Kumagai and Yuuki Yamanaka},
      year={2024},
      eprint={2405.18929},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
Please read [LICENCE.md](LICENCE.md) before reading or using the files.

## Prerequisites
- Please install `python>=3.10`, `numpy`, `scipy`, `torch`, `torchvision`, `scikit_learn`, and `matplotlib`
- Please also see `requirements.txt`
- Note that we use the nvidia pytorch docker image (`23.12-py3`)
  -  https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags


## Datasets
All datasets will be downloaded when first used.

## Example
All settings should be in config.yml
```
python angular_dino_train.py --config config.yml
```
