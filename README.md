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

## Abstract

本研究では、DINOによる自己教師あり学習を用いた異常検知手法を提案する。ラベル情報を用いないクラスタリングにより、PUデータセットの限られたラベル情報を活用し、正常データおよび異常データの代表的なプロトタイプを学習することで、未知データの異常検知を行う。  

データセットは **unlabeled** データと **positive** データから構成されており、データの不均衡による影響を軽減するために以下の手法を導入した。  
- **DataLoaderの工夫**: `torch.utils.data.WeightedRandomSampler` を用い、positiveサンプルを重複許容でサンプリング。  
- **損失の重み付け**: unlabeledサンプルの損失を小さく、positiveサンプルの損失を大きくなるよう調整。  

また、DINOの埋め込み表現に対して、新たなレイヤーを追加した。このレイヤーはHeadの最終層と同じ形を持ち、Headと同様にStudentとTeacherで交差エントロピー損失を計算する。  

最終的な損失関数は以下のように表される.
$$\mathcal{L} = \mathcal{L}_{\text{DINO}} + \lambda \mathcal{L}_{\text{ANG}}$$
ここで、  
- **$\mathcal{L}_{\text{DINO}}$**: DINO本来の損失  
- **$\mathcal{L}_{\text{ANG}}$**: 追加レイヤーによる損失  
- **$\lambda$**: 両者のバランスを調整する係数（学習の進行に伴い増加）  

学習が進むにつれ、追加レイヤーの重みベクトルは、訓練データの各パターンを表す埋め込み表現と類似し、それぞれのパターンを代表するベクトルへと収束する。  
例えば、訓練用データセットに「犬」「猫」「馬」などの異なるパターンがある場合、重みベクトルの集合の中にそれぞれ対応する $\bm{v}_{犬}, \bm{v}_{猫}, \bm{v}_{馬}, \dots$ が形成される。  

ここで、もし **犬が正常クラス** で、それ以外（猫や馬など）が異常クラスであるとすると、unlabeledデータには犬のサンプルが多く含まれる。そのため、unlabeledデータの埋め込み表現と類似する重みベクトル$\bm{v}_{犬}$が **正常プロトタイプ** となる。  
一方、positiveデータ（異常クラス）の埋め込み表現に類似する重みベクトルは$\bm{v}_{猫}, \bm{v}_{馬}, \dots$ となり、これらが **異常プロトタイプ** として機能する。  


さらに、**50エポック目以降** は 推定した正常・異常プロトタイプに基づいた**角度マージン戦略** を導入し、以下の二つの手法を採用する。  
- **加法的角度マージン (Additive Angular Margin)**:  
  - Studentの埋め込み表現と追加レイヤーの重みベクトルの間の角度に正のマージンを与えることで、クラス内分散を小さくし、クラス間分散を大きくする（e.g. ArcFace）。  
- **減法的角度マージン (Subtractive Angular Margin)**:  
  - Teacherの埋め込み表現と追加レイヤーの重みベクトルの間の角度に**負のマージン**を与え、適切な教師ラベルを出力させる条件を緩和する。  
