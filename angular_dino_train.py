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

from angular_dino.angular_dino.my_utils import plot_embed, DINOLoss, converge_cosine_scheduler, \
    myResNet, my_train_one_epoch, ReturnEmbWrapper, backbone_emb_Loss, \
    DataAugmentationDINOMNIST, DataAugmentationDINOCIFAR10, DataAugmentationDINOFashionMNIST, DataAugmentationDINOSVHN, \
    Weak_DataAugmentationDINOCIFAR10, Smallscale_DataAugmentationDINOSVHN
if "../" not in sys.path:
    sys.path.append("../")
from dataloaders import load

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)
    parser.add_argument('--config', type=str, help='Path to the config file.')
    return parser

def train_dino(args):
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
            print("not clearml", args.gpu)
    else:
        logger=None 
        print("not clearml", args.gpu)

    torchvision_archs = sorted(name for name in torchvision_models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(torchvision_models.__dict__[name]))
    
    # ============ preparing dataset ... ============
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
    transform = aug(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        # resize=args.resize
    )

    if args.dataset == "MNIST":
        dataset = datasets.MNIST(
                    root="../datasets", download=False, train=True, transform=transform
                )
        print(f"dataset loaded MNIST")
    #PU dataset
    #datasetはtrainのpositive+unlabeled
    #pos_train_subsetはtrainのうちのpositive
    #unl_normalize_subsetはtrainのunlabeledを,dino_transformではなく単にnormalizeしたもの. プロトタイプの推定に用いる.
    #pos_normalize_subsetはtrainのpositiveを,dino_transformではなく単にnormalizeしたもの. プロトタイプの推定に用いる.
    #val_subsetは検証用データセット
    elif "PU" in args.dataset:
        dataset, pos_train_subset, unl_normalize_subset, pos_normalize_subset, val_subset = load(
        name=args.dataset.replace("PU", ""),
        batch_size=False,
        normal_class= args.normal_class, #[1],
        unseen_anomaly= args.unseen_anomaly_class, #[0],
        labeled_anomaly_class=False, 
        n_train = 4500,
        n_valid = 500,
        # n_test = 2000,
        n_unlabeled_normal = 4500, #4500
        n_unlabeled_anomaly = 250, #250
        n_labeled_anomaly = 250, #250
        return_pu_pos_tra_valtrans_val_subset = True, 
        transform = transform,
        seed=0,
        return_id=True, #False: 1->0, 0,2~9->1, True: 0->0, ... 9->9, only use visualise result
        )

        test_dataset = load(
        name=args.dataset.replace("PU", ""), batch_size=128, normal_class= args.normal_class, unseen_anomaly= args.unseen_anomaly_class, labeled_anomaly_class=False, 
        # n_train = 4500, n_valid = 500, 
        n_test = 2000,
        # n_unlabeled_normal = 4500, n_unlabeled_anomaly = 250, n_labeled_anomaly = 250,
        return_test_subset=True, 
        transform = None, seed=0,
        return_id=False, #False: 1->0, 0,2~9->1, True: 0->0, ... 9->9, only use visualise result
        )
        print(f"dataset loaded {args.dataset}")

    else: raise NameError("Invalid dataset name")

    unl_pos = {"unl":len(dataset)-len(pos_train_subset), "pos":len(pos_train_subset)} #訓練用データセットにおけるunlabeled, positiveそれぞれの数
    print(f"dataset unl_num:{unl_pos['unl']}, pos_num:{unl_pos['pos']}")

    # unl_pos_weight={"unl":unl_pos["pos"]/(unl_pos["unl"]+unl_pos["pos"]), 
    #                 "pos":unl_pos["unl"]/(unl_pos["unl"]+unl_pos["pos"])} #({1/a} / {1/a+1/b} = b/a+b)
    unl_pos_weight={"unl":(unl_pos["unl"]+unl_pos["pos"])/(2*unl_pos["unl"]), 
                    "pos":(unl_pos["unl"]+unl_pos["pos"])/(2*unl_pos["pos"])} #unlとposのバランスをとる係数. aはunl, bはposの数に対応. (a+b)/2a << (a+b)/2b
    print(f"unl_pos_weight: {unl_pos_weight}") #この重みは, samplerとlossの計算に使用され, unlとposの数の不均衡に対処するために用いられる.
    sample_weight = [list(unl_pos_weight.values())[i[1][0].item()] for i in dataset] #data_loaderは上記の理由から, positive sampleを重複を許したくさんサンプリングする. 
    sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(dataset), replacement=True)
        
    # sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        # shuffle=True,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============

    class Normalize_Module(nn.Module): #backboneの最後に取り付ける, normalizeのためのモジュール
        def __init__(self, p=2, dim=1):
            super(Normalize_Module, self).__init__()
            self.p = p
            self.dim = dim
        def forward(self, x):
            return torch.nn.functional.normalize(x, p=self.p, dim=self.dim)

    # student = torchvision_models.__dict__["resnet18"]()
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

    ### Prototype vector
    prototype_num = args.prototype_num
    student_prototype_weight = nn.utils.weight_norm(nn.Linear(embed_dim, prototype_num, bias=False)).cuda()
    student_prototype_weight.weight_g.data.fill_(1)
    student_prototype_weight.weight_g.requires_grad = False
    student_prototype_weight = nn.parallel.DistributedDataParallel(student_prototype_weight, device_ids=[args.gpu], broadcast_buffers=False)

    teacher_prototype_weight = nn.utils.weight_norm(nn.Linear(embed_dim, prototype_num, bias=False)).cuda()
    teacher_prototype_weight.weight_g.data.fill_(1)
    teacher_prototype_weight.weight_g.requires_grad = False
    teacher_prototype_weight.load_state_dict(student_prototype_weight.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher_prototype_weight.parameters():
        p.requires_grad = False
    #####

    # head_arcface_conf = {'name': 'arcface', 's':30.0, 'm':0.30, 'easy_margin':False, 'warmup':0}
    head_arcface_conf = {'name': None}  # Do not use arcface for head

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
    # synchronize batch norms (if any)
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

    dino_loss = DINOLoss( #to compute ordinary dino loss
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs, 
        args=args,
        logger=logger,
        student_temp=args.student_tmp, 
        unl_pos_weight=unl_pos_weight
    ).cuda()

    temp_schedule = converge_cosine_scheduler(args.prototype_student_temp, args.final_emb_student_temp,
                                               args.epochs, len(data_loader),
                                               start_warmup_value=args.prototype_student_temp,
                                               warmup_epochs=0,
                                               converge_epoch= args.epochs//2)
    
    margin_settings = {'name': 'arcface', 's':30.0, 'm':args.margin, 'easy_margin':False, 
                        'warmup': [1 if i>=0 else 0 for i in converge_cosine_scheduler(1, 1,
                                               args.epochs, len(data_loader),
                                               start_warmup_value=-1,
                                               warmup_epochs=100,)]}
    emb_Loss = backbone_emb_Loss( #to compute prototype vector loss
        prototype_num, 
        args.local_crops_number + 2, 
        args.emb_warmup_teacher_temp,
        args.emb_teacher_temp,
        args.emb_warmup_teacher_temp_epochs,
        args.epochs, 
        args=args,
        logger=logger,
        student_temp=args.prototype_student_temp,
        unl_pos_weight=unl_pos_weight,
        student_temp_schedule = temp_schedule, 
        margin_settings = margin_settings
        ).cuda()


    student_prototype_groups = utils.get_params_groups(student_prototype_weight) #to optimize prototype vector
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

    # ============ init schedulers ... ============
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
    # lambda parameter is increased and set to 1 after converge_epoch
    lambda_schedule = converge_cosine_scheduler(0, args.lambda_end,
                                               args.epochs, len(data_loader),
                                               converge_epoch= args.lambda_converge)
    # lambda_schedule = converge_cosine_scheduler(args.lambda_end, args.lambda_end,
    #                                            args.epochs, len(data_loader),
    #                                            start_warmup_value=args.lambda_end,
    #                                            warmup_epochs=args.lambda_converge
    #                                            )
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
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

    best_valid = 1e5
    pu_normal_prototype_dict = None
    pu_positive_prototype_dict = None
    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        # data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats, epoch_valid, pu_normal_prototype_dict, pu_positive_prototype_dict = my_train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args=args, logger=logger, 
            unl_normalize_subset=unl_normalize_subset, pos_normalize_subset=pos_normalize_subset,
            lambda_schedule=lambda_schedule,
            student_prototype_weight=student_prototype_weight, teacher_prototype_weight=teacher_prototype_weight, s_prototype_optimizer=s_prototype_optimizer, 
            emb_loss=emb_Loss, test_dataset=test_dataset, val_dataset=val_subset, normal_class=args.normal_class, 
            pu_normal_prototype_dict=pu_normal_prototype_dict, pu_positive_prototype_dict=pu_positive_prototype_dict)

        # ============ writing logs ... ============
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
        if utils.is_main_process() and (epoch%100 == 0 or epoch==args.epochs-1):#(epoch%10 == 0 or epoch==99):
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
    with open(config_file, 'r') as file:
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