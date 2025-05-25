"""
angular_dino_train.py

DINO（自己教師あり学習）を用いた異常検知の学習スクリプト。
設定ファイル（config.yml）を読み込み、データセット準備、モデル構築、学習ループ、結果保存までを一括で実行します。

主な流れ：
1. 引数・設定ファイルの読み込み
2. データセット・データ拡張の準備
3. モデル（ResNet/VisionTransformer等）の構築
4. 学習ループの実行
5. ログ・モデルの保存
"""

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import yaml
import shutil
from collections import Counter
import warnings
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')
warnings.filterwarnings('ignore', message='.*torch.nn.utils.weight_norm.*')
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.GradScaler.*')

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from  torch.utils.data import DataLoader, WeightedRandomSampler
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torchvision.models.resnet import BasicBlock, Bottleneck
from clearml import Task
from functools import partial
import timm

import utils
import vision_transformer as vits
from vision_transformer import DINOHead, VisionTransformer

from models.model_utils import myResNet, ReturnEmbWrapper
from models.losses import DINOLoss, PrototypeLoss
from augmentations import DataAugmentationDINOMNIST, DataAugmentationDINOCIFAR10, DataAugmentationDINOFashionMNIST, DataAugmentationDINOSVHN, Weak_DataAugmentationDINOCIFAR10, Smallscale_DataAugmentationDINOSVHN
from train_utils import my_train_one_epoch, validation, converge_cosine_scheduler
from eval_utils import plot_embed, test_anomaly_detection, get_prototype_vec_ids_from_loader, calc_y_score
from utils_extra import split, print_f
if "../" not in sys.path:
    sys.path.append("../")
from dataloaders import load

def get_args_parser():
    """
    コマンドライン引数パーサを作成
    --config: 設定ファイルのパス
    """
    parser = argparse.ArgumentParser('DINO', add_help=False)
    parser.add_argument('--config', type=str, help='Path to the config file.')
    return parser

def train_dino(args):
    """
    DINOによる異常検知モデルの学習メイン関数。
    設定に従いデータセット・モデル・損失関数・最適化手法を構築し、学習ループを回す。
    """
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(0)
    print("git:\n  {}\n".format(utils.get_sha()))
    # print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.use_clearml:
        if utils.is_main_process():
            task = get_task(args.project_name, args.task_name)
            # task.set_comment(args.description)
            logger = task.get_logger()
            print(f"clearml reporting gpu_{args.gpu} process" )
            # if args.log_file:
            #     write_log(args.log_file, logger)
        else:
            logger=None 
            print("clarml not use")
    else:
        logger=None 
        print("clarml not use")

    torchvision_archs = sorted(name for name in torchvision_models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(torchvision_models.__dict__[name]))
    
    # ============ データセットとデータ拡張の準備 ============
    if "MNIST" in args.dataset:
        aug = DataAugmentationDINOMNIST
        ch = 1
    elif "CIFAR10" in args.dataset:
        if args.transform == "week":
            aug = Weak_DataAugmentationDINOCIFAR10
        else:
            aug = DataAugmentationDINOCIFAR10
        ch = 3
    elif "FashionMNIST" in args.dataset:
        aug = DataAugmentationDINOFashionMNIST
        ch = 1
    elif "SVHN" in args.dataset:
        if "vit_small_scale" == args.arch:
            aug = Smallscale_DataAugmentationDINOSVHN
            print("small scale data aug")
        else:
            aug = DataAugmentationDINOSVHN
        ch = 3
    else:
        raise NameError("Invalid dataset name")

    # データ拡張インスタンス生成
    transform = aug(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        # resize=args.resize
    )

    # データセットのロード
    if args.dataset == "MNIST":
        dataset = datasets.MNIST(
                    root="../datasets", download=False, train=True, transform=transform
                )
        print(f"dataset loaded MNIST")
    elif "PU" in args.dataset:
        # PU（Positive-Unlabeled）データセットのロード
        # dataset: 訓練用データローダーのためのデータセット
        # pos_train_subset: *訓練データ*の異常サブセット
        # unl_normalize_subset: 訓練用のラベル無しデータの正規化されたサブセット
        # pos_normalize_subset: 訓練用の異常データの正規化されたサブセット
        # val_subset: 検証用データのサブセット
        dataset, pos_train_subset, unl_normalize_subset, pos_normalize_subset, val_subset = load(
            name=args.dataset.replace("PU", ""),
            normal_class= args.normal_class,
            unseen_anomaly= args.unseen_anomaly_class,
            labeled_anomaly_class=False, 
            n_unlabeled_normal = 4500,
            n_unlabeled_anomaly = 250,
            n_labeled_anomaly = 250,
            train_ratio=0.9, # train=4500, val=500
            return_type="train",
            transform = transform,
            seed=0,
            return_id=True,
        )

        test_dataset = load(
            name=args.dataset.replace("PU", ""), normal_class= args.normal_class, unseen_anomaly= args.unseen_anomaly_class, labeled_anomaly_class=False, 
            n_test = 2000,
            return_type="test", 
            transform = None, seed=0,
            return_id=False,
        )
        print(f"dataset loaded {args.dataset}")
    else:
        raise NameError("Invalid dataset name")

    # クラス不均衡を補正するための重みを計算
    ## データセット内のラベル付き異常サンプル数とラベルなしサンプル数を計算
    class_counts = {
        "unlabeled": len(dataset) - len(pos_train_subset),  # ラベルなしサンプル数
        "positive": len(pos_train_subset)  # ラベル付き異常サンプル数
    }
    print(f"Dataset statistics - Unlabeled samples: {class_counts['unlabeled']}, Positive samples: {class_counts['positive']}")

    ## 各クラスの重みは、(全サンプル数)/(2*クラスサンプル数)で計算
    unl_pos_weight = {
        "unl": (class_counts["unlabeled"] + class_counts["positive"]) / (2 * class_counts["unlabeled"]),
        "pos": (class_counts["unlabeled"] + class_counts["positive"]) / (2 * class_counts["positive"])
    }
    print(f"Class weights for balanced sampling: {unl_pos_weight}")

    ## 各サンプルに重みを割り当て
    ## datasetはラベルなしデータの後ろに異常データが連結している形になっているため、
    ## sample_weightsは[unl_weight,..., unl_weight, pos_weight, ...]のようになる
    sample_weights = [list(unl_pos_weight.values())[sample[1][0].item()] for sample in dataset]
    
    ## positive sampleを重複を許してたくさんサンプリングするSamplerを定義
    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=weighted_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ モデル構築 ============
    # 学習するネットワーク（student, teacher）とプロトタイプベクトルの定義
    # ResNet/VisionTransformer/その他アーキテクチャに対応
    class Normalize_Module(nn.Module):
        """
        特徴ベクトルをL2ノルムで正規化するためのモジュール
        """
        def __init__(self, p=2, dim=1):
            super(Normalize_Module, self).__init__()
            self.p = p
            self.dim = dim
        def forward(self, x):
            return torch.nn.functional.normalize(x, p=self.p, dim=self.dim)

    # --- モデル本体の構築 ---
    if args.arch == "resnet18":
        embed_dim = args.emb_dim
        student = myResNet(BasicBlock, [2,2,2,2], num_classes=embed_dim, normalize=True)
        student.conv1 = nn.Conv2d(ch, 64, kernel_size=7, stride=1, padding=3,bias=False)
        student.fc = nn.Linear(student.fc.in_features, out_features=embed_dim, bias=True)
        teacher = myResNet(BasicBlock, [2,2,2,2], num_classes=embed_dim, normalize=True)
        teacher.conv1 = nn.Conv2d(ch, 64, kernel_size=7, stride=1, padding=3,bias=False)
        teacher.fc = nn.Linear(teacher.fc.in_features, out_features=embed_dim, bias=True)
    elif args.arch == "resnet50":
        embed_dim = args.emb_dim
        student = myResNet(Bottleneck, [3,4,6,3], num_classes=embed_dim, normalize=True)
        student.conv1 = nn.Conv2d(ch, 64, kernel_size=7, stride=1, padding=3,bias=False)
        student.fc = nn.Linear(student.fc.in_features, out_features=embed_dim, bias=True)
        teacher = myResNet(Bottleneck, [3,4,6,3], num_classes=embed_dim, normalize=True)
        teacher.conv1 = nn.Conv2d(ch, 64, kernel_size=7, stride=1, padding=3,bias=False)
        teacher.fc = nn.Linear(teacher.fc.in_features, out_features=embed_dim, bias=True)
    elif args.arch == "resnet50_drop":
        student = timm.models.resnet.ResNet(
            timm.models.resnet.Bottleneck, [3,4,6,3], num_classes=args.emb_dim, 
            drop_rate=args.drop_rate, drop_path_rate=args.drop_path_rate, drop_block_rate=args.drop_block_rate)
        student.conv1 = nn.Conv2d(ch, 64, kernel_size=7, stride=1, padding=3,bias=False)
        student.fc = nn.Sequential(
            student.fc,
            Normalize_Module(p=2, dim=1)
        )
        teacher = timm.models.resnet.ResNet(
            timm.models.resnet.Bottleneck, [3,4,6,3], num_classes=args.emb_dim, 
            drop_rate=args.drop_rate, drop_path_rate=args.drop_path_rate, drop_block_rate=args.drop_block_rate)
        teacher.conv1 = nn.Conv2d(ch, 64, kernel_size=7, stride=1, padding=3,bias=False)
        teacher.fc = nn.Sequential(
            teacher.fc,
            Normalize_Module(p=2, dim=1)
        )
        embed_dim = args.emb_dim
    elif args.arch == "efficientnetv2":
        embed_dim = args.emb_dim
        student = timm.create_model("efficientnetv2_s", pretrained=False, num_classes=embed_dim) 
        student.classifier = nn.Sequential(
            student.classifier,
            Normalize_Module(p=2, dim=1)
        )
        teacher = timm.create_model("efficientnetv2_s", pretrained=False, num_classes=embed_dim) 
        teacher.classifier = nn.Sequential(
            teacher.classifier,
            Normalize_Module(p=2, dim=1)
        )
    elif "vit_small_scale":
        # VisionTransformer（ViT）小型スケール
        student = VisionTransformer(img_size=[32],
            patch_size=args.patch_size,
            in_chans=3,
            embed_dim=192,
            depth=9,
            num_heads=12,
            mlp_ratio=2,
            qkv_bias=True,
            # drop_rate=args.drop_rate, #0,
            drop_path_rate=args.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            normalize=True,
            )
        teacher = VisionTransformer(img_size=[32],
            patch_size=args.patch_size,
            in_chans=3,
            embed_dim=192,
            depth=9,
            num_heads=12,
            mlp_ratio=2,
            qkv_bias=True,
            # drop_rate=args.drop_rate, #0,
            drop_path_rate=args.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            normalize=True,
            )
        embed_dim = student.embed_dim
        print("custom vit build")
    elif "vit" in args.arch:
        # VisionTransformer（ViT）
        student = vits.__dict__[args.arch](
                patch_size=args.patch_size,
                drop_path_rate=args.drop_path_rate,  # stochastic depth
                normalize=True
            )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size, normalize=True)
        embed_dim = student.embed_dim
    else:
        raise NameError(f"invalid args.arch {args.arch}")
    
    print(f"model arch: {args.arch}")

    # --- プロトタイプベクトルの定義 ---
    prototype_num = args.prototype_num # プロトタイプの数
    # student側のプロトタイプベクトルの初期化
    student_prototype_weight = nn.utils.weight_norm(nn.Linear(embed_dim, prototype_num, bias=False)).cuda()
    student_prototype_weight.weight_g.data.fill_(1)
    student_prototype_weight.weight_g.requires_grad = False
    student_prototype_weight = nn.parallel.DistributedDataParallel(student_prototype_weight, device_ids=[args.gpu], broadcast_buffers=False)

    # teacher側のプロトタイプベクトルの初期化
    teacher_prototype_weight = nn.utils.weight_norm(nn.Linear(embed_dim, prototype_num, bias=False)).cuda()
    teacher_prototype_weight.weight_g.data.fill_(1)
    teacher_prototype_weight.weight_g.requires_grad = False
    # studentから重みをコピーし、勾配の計算を行わないようにする
    teacher_prototype_weight.load_state_dict(student_prototype_weight.module.state_dict())
    for p in teacher_prototype_weight.parameters():
        p.requires_grad = False

    # --- DINOヘッドの設定 ---
    head_arcface_conf = {'name': None}  # Arcfaceはheadでは使わない

    # backbone(プロトタイプベクトルは含まない)とDINOヘッドを結合し、DINOヘッドの出力、およびbackboneの出力を返すラッパーを定義
    student = ReturnEmbWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
        arcface_family_conf=head_arcface_conf,
        nlayers=args.nlayers, #3,
        hidden_dim=args.hidden_dim, #512
        bottleneck_dim=args.bottleneck_dim #128,
    ))
    teacher = ReturnEmbWrapper(
        teacher,
        DINOHead(
            embed_dim, 
            args.out_dim,
            args.use_bn_in_head,
            arcface_family_conf=head_arcface_conf,
            nlayers=args.nlayers, #3,
            hidden_dim=args.hidden_dim, #512
            bottleneck_dim=args.bottleneck_dim #128,
            ),
    )
    
    student, teacher = student.cuda(), teacher.cuda()
    # BatchNormの同期化（必要な場合のみ）
    if utils.has_batchnorms(student):
        print("convert_sync_batchnorm")
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], broadcast_buffers=False)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False)
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both vit_small network.")

    # ============ 損失関数・スケジューラ・最適化手法の準備 ============
    # studentとteacherのヘッドの出力からDINOの損失を計算するクラス
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = local_crops_num + 2global crops
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs, 
        args=args,
        logger=logger,
        student_temp=args.student_tmp, 
        unl_pos_weight=unl_pos_weight
    ).cuda()

    # プロトタイプレイヤーで用いるソフトマックス温度パラメータのスケジューラ
    # base_value=1, final_value=0, epoch=10 warmup_epochs=3なら, 3エポックまで0から1まで線形に増加し、そこからコサイン関数で0に収束する
    temp_schedule = converge_cosine_scheduler(base_value=args.prototype_student_temp, final_value=args.final_emb_student_temp,
                                               epochs=args.epochs, niter_per_ep=len(data_loader),
                                               start_warmup_value=args.prototype_student_temp,
                                               warmup_epochs=0,
                                               converge_epoch= args.epochs//2) # これだとすべて0.1固定
    
    # プロトタイプレイヤーで用いるマージンの設定
    # 実際のマージンの値は'm'*'warmup'で与えられる
    margin_settings = {'name': 'arcface', 's':30.0, 'm':args.margin, 'easy_margin':False, 
                        'warmup': [1 if i>=50*len(data_loader) else 0 for i in range(args.epochs*len(data_loader))] # 50epからmarginを有効にする
                        }
    
    # プロトタイプレイヤーの出力で計算する損失関数
    prototype_loss = PrototypeLoss(
        prototype_num, # プロトタイプとして選択するベクトルの数
        args.local_crops_number + 2, 
        args.emb_warmup_teacher_temp,
        args.emb_teacher_temp,
        args.emb_warmup_teacher_temp_epochs,
        args.epochs, 
        args=args,
        logger=logger,
        unl_pos_weight=unl_pos_weight,
        # student_temp=args.prototype_student_temp,
        student_temp_schedule = temp_schedule, 
        margin_settings = margin_settings
        ).cuda()

    # --- オプティマイザの準備 ---
    student_prototype_groups = utils.get_params_groups(student_prototype_weight)
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        s_prototype_optimizer = torch.optim.AdamW(student_prototype_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.precision in ['fp16', 'bf16']:
        fp16_scaler = torch.cuda.amp.GradScaler()
    elif args.precision != 'fp32':
        raise ValueError(f"Invalid precision setting: {args.precision}. Choose from 'fp32', 'fp16', or 'bf16'.")

    # --- スケジューラの準備 ---
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        float(args.min_lr),
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    # DINO損失とプロトタイプ損失の合計する際に、プロトタイプ損失に掛ける係数
    lambda_schedule = converge_cosine_scheduler(0, args.lambda_end,
                                               args.epochs, len(data_loader),
                                               converge_epoch= args.lambda_converge)
    print(f"Loss, optimizer and schedulers ready.")

    # ============ チェックポイントからの再開 ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    # ============ 学習前の初期化 ============
    best_valid = 1e5
    # 正常プロトタイプ
    normal_prototype_dict = None
    # 異常プロトタイプ
    positive_prototype_dict = None
    start_time = time.time()
    print("Starting DINO training !")

    # ============ 学習ループ ============
    for epoch in range(start_epoch, args.epochs):
        # 1エポック分の学習
        train_stats, epoch_valid, normal_prototype_dict, positive_prototype_dict = my_train_one_epoch(
            student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, 
            args=args, logger=logger, 
            unl_normalize_subset=unl_normalize_subset, pos_normalize_subset=pos_normalize_subset,
            lambda_schedule=lambda_schedule,
            student_prototype_weight=student_prototype_weight, teacher_prototype_weight=teacher_prototype_weight, 
            s_prototype_optimizer=s_prototype_optimizer, 
            prototype_loss=prototype_loss, test_dataset=test_dataset, val_dataset=val_subset, normal_class=args.normal_class, 
            normal_prototype_dict=normal_prototype_dict, positive_prototype_dict=positive_prototype_dict)

        # ============ モデル・ログの保存 ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
            's_prototype_vec': student_prototype_weight.state_dict(),
            't_prototype_vec': teacher_prototype_weight.state_dict()
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoints/checkpoint.pth'))
        print(f"output_path: {args.output_dir}")
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoints/checkpoint{epoch:04}.pth'))
        if best_valid > epoch_valid:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoints/bestcheckpoint.pth'))
            print(f"save bestcheckpoint.pth")
            best_valid=epoch_valid
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        # 100エポックごと、または最終エポックで埋め込み可視化
        if utils.is_main_process() and (epoch%100 == 0 or epoch==args.epochs-1):
            print("plot embedding")
            plot_embed(teacher_without_ddp, epoch, output_path=args.output_dir, logger=logger, dataset=args.dataset.replace("PU", ""))
            print("save embedding")
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def load_config(config_file):
    with open(config_file, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def get_task(project_name, task_name):
    task = Task.init(
        project_name=project_name, 
        task_name=task_name, 
        # output_uri='s3://133.8.202.70:9000/clearml-default',
        # output_uri=True  # IMPORTANT: setting this to True will upload the model
        # If not set the local path of the model will be saved instead!
        auto_connect_frameworks={'pytorch': False, 'matplotlib': False}
    )
    return task

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    config = load_config(args.config)

    # コマンドライン引数を更新
    for key, value in config.items():
        parser.add_argument(f'--{key}', default=value)

    # コマンドライン引数を再度パース
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.output_dir+"/description.txt", mode="w") as f:
        f.write(args.description)
    
    shutil.copyfile(args.config, args.output_dir+"/config.yml")

    # gpu_num = str(args.gpu_num)
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    if isinstance(args.gpu_num, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    elif isinstance(args.gpu_num, (list, tuple)):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu_num))
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print
        raise ValueError("gpu_num should be an integer or a list/tuple of integers")
    train_dino(args)