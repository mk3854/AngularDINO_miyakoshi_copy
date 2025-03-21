import argparse
import itertools
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import yaml
import random
from collections import defaultdict
import re
# import pypdf
from collections import defaultdict, Counter

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import torch.nn as nn
from  torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, roc_auc_score
import umap

import utils

class MLPNet (nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28 * 1, 512)   
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)
        
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return F.relu(self.fc3(x))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class ResNetMNIST(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(ResNetMNIST, self).__init__()
        self.in_channels = 64
        
        # 最初の畳み込み層を小さくする（7x7→3x3）
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # プーリング層を削除（画像サイズが小さいため）
        
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)    # 28x28 -> 28x28
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)   # 28x28 -> 28x28
        x = self.layer2(x)   # 28x28 -> 14x14
        x = self.layer3(x)   # 14x14 -> 7x7
        x = self.layer4(x)   # 7x7 -> 4x4
        
        x = self.avgpool(x)  # 4x4 -> 1x1
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class myResNet(torchvision_models.ResNet):
    def __init__(self, block, layers, num_classes = 1000, zero_init_residual = False, groups = 1, width_per_group = 64, replace_stride_with_dilation = None, norm_layer = None, normalize=False):
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.normaize = normalize
    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.normaize:
            x = torch.nn.functional.normalize(x, p=2, dim=1)

        return x


class CustomWrapper(utils.MultiCropWrapper):
    def __init__(self, backbone, head):
        super(utils.MultiCropWrapper, self).__init__()
        self.backbone = backbone
        self.head = head

class ReturnEmbWrapper(nn.Module):
    """
    Wrapper to return the output of the backbone and the head.
    """

    def __init__(self, backbone, head):
        super(ReturnEmbWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        self.backbone = backbone
        self.head = head

    def forward(self, x, module_type=False, max_id=False, epoch=False, return_before_arcface=False, return_emb=False):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        # if return_emb: breakpoint()
        return self.head(output, module_type, max_id, epoch, return_before_arcface, return_emb), output


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, args=None, logger=None,
                 unl_pos_weight=None):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.batch_size = args.batch_size_per_gpu
        self.logger = logger
        self.unl_pos_weight = unl_pos_weight

    def forward(self, student_output, teacher_output, epoch, iteration=0, 
                class_ids=None, student_raw_out=None, pos_output=False, pos_batch=False, use_weight=False):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        if use_weight and torch.is_tensor(class_ids):# or isinstance(class_ids, list): #unlとposの不均衡を解消するため, lossに重みをつける
            unl_pos = class_ids[:,0] #batch_size  [0,0,0,1,0,0,0,...]
            batch_weight = (1-unl_pos)*self.unl_pos_weight["unl"] + unl_pos*self.unl_pos_weight["pos"]
            batch_weight = batch_weight.to("cuda")
        else:
            batch_weight = None

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                if batch_weight != None:
                    loss = loss * batch_weight
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        #####
        if self.logger and (iteration%500==0 or iteration==0):
            self.visualize_output(student_out, student_raw_out, teacher_out, iteration, temp)
        #####
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def visualize_output(self, student_sharpend_out, student_raw_out, teacher_sf_out, iteration, temp):
        ### plotting 2 teacher softmax outputs
        plot_minibatch = np.random.randint(0, self.batch_size)
        plot_teacher_0=teacher_sf_out[0][plot_minibatch].detach().cpu().numpy() #emb_dim
        plot_teacher_1=teacher_sf_out[1][plot_minibatch].detach().cpu().numpy() #emb_dim
        top_100_indices_from0 = np.argsort(plot_teacher_0)[::-1][:100]
        top_100_indices_from1 = np.argsort(plot_teacher_1)[::-1][:100]
        top_100_indices = []
        ind_0, ind_1 = 0, 0
        for i in range(50):
            while True:
                if top_100_indices_from0[ind_0] not in top_100_indices:
                    top_100_indices.append(top_100_indices_from0[ind_0])
                    ind_0+=1
                    break
                else: ind_0+=1
            while True:
                if top_100_indices_from1[ind_1] not in top_100_indices:
                    top_100_indices.append(top_100_indices_from1[ind_1])
                    ind_1+=1
                    break
                else: ind_1+=1
        top_100_indices = np.array(top_100_indices)
        top_100_indices = np.sort(top_100_indices)
        plot_teacher_0 = plot_teacher_0[top_100_indices]
        plot_teacher_1 = plot_teacher_1[top_100_indices]
        string_top_100_indices = [str(i) for i in top_100_indices]
        self.logger.report_histogram(
            f"iteration:{iteration} teacher_out 0 S:{temp.item():.2f}, C:{self.center.mean().item():.2f}",
            "value",
            # iteration=epoch,
            values=plot_teacher_0,
            # xlabels=string_top_100_indices,
            xaxis="title x",
            yaxis="title y",
        )
        self.logger.report_histogram(
            f"iteration:{iteration} teacher_out 1 S:{temp.item():.2f}, C:{self.center.mean().item():.2f}",
            "value",
            # iteration=epoch,
            values=plot_teacher_1,
            # xlabels=string_top_100_indices,
            xaxis="title x",
            yaxis="title y",
        )

        ### plotting student arcface softmax output
        plot_view = np.random.randint(0, self.ncrops)
        plot_student=(F.softmax(student_sharpend_out[plot_view][plot_minibatch], dim=-1)).detach().cpu().float().numpy()
        plot_student = plot_student[top_100_indices]
        self.logger.report_histogram(
            f"iteration:{iteration} student_out {plot_view} w/ arcface,softmax S:{self.student_temp:.2f}",
            "value",
            # iteration=epoch,
            values=plot_student,
            xaxis="title x",
            yaxis="title y",
        )

        ### plotting student raw softmax output
        if student_raw_out is not None:
            student_raw_out = student_raw_out.chunk(self.ncrops)
            # plot_view = np.random.randint(0, self.ncrops)
            plot_student=F.softmax(student_raw_out[plot_view][plot_minibatch], dim=-1)
            plot_student = plot_student[top_100_indices]
            self.logger.report_histogram(
                f"iteration:{iteration} student_out {plot_view} w/o arcface w/ softmax S:{self.student_temp:.2f}",
                "value",
                # iteration=epoch,
                values=plot_student,
                xaxis="title x",
                yaxis="title y",
            )

        ### plotting student arcface output
        # plot_view = np.random.randint(0, self.ncrops)
        plot_student=(student_sharpend_out[plot_view][plot_minibatch]*self.student_temp).detach().cpu().float().numpy()
        plot_student = plot_student[top_100_indices]
        self.logger.report_histogram(
            f"iteration:{iteration} student_out {plot_view} w/ arcface w/o softmax",
            "value",
            # iteration=epoch,
            values=plot_student,
            xaxis="title x",
            yaxis="title y",
        )

        if student_raw_out is not None:
            ### plotting student raw output
            plot_student=student_raw_out[plot_view][plot_minibatch]
            plot_student = plot_student[top_100_indices]
            self.logger.report_histogram(
                f"iteration:{iteration} student_out {plot_view} w/o arcface,softmax",
                "value",
                # iteration=epoch,
                values=plot_student,
                xaxis="title x",
                yaxis="title y",
            )

class sub_dataset_sampler(DistributedSampler):

    """
    Iterates two sets of indices, combining them into batches.

    This sampler is designed to work in distributed training environments, ensuring
    that each process receives a balanced subset of the data.
    """

    def __init__(self, main_dataset, main_batch_size, sub_dataset, sub_batch_size):
        super().__init__(sub_dataset)
        self.main_indices = list(range(len(main_dataset)))
        self.main_batch_size = main_batch_size

        self.indices = list(range(len(sub_dataset)))
        self.batch_size = sub_batch_size
        if (len(self.indices)//self.batch_size) % self.num_replicas != 0:
            self.num_samples = math.ceil( ((len(self.main_indices)//self.main_batch_size) - self.num_replicas) / self.num_replicas )
        else: 
            self.num_samples = math.ceil( ((len(self.main_indices)//self.main_batch_size)) / self.num_replicas )

        self.total_size = self.num_samples * self.num_replicas
        print(f"num replica {self.num_replicas}, rank {self.rank}, total_size {self.total_size}")

    def __iter__(self):
        # breakpoint()
        main_iter = iterate_once(self.main_indices)
        eternal_iter = iterate_eternally(self.indices)

        indices = [batch for (_, batch) in 
        zip(grouper(main_iter, self.main_batch_size), grouper(eternal_iter, self.batch_size))]

        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples #len(self.primary_indices) // self.primary_batch_size // self.num_replicas


def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class eval_transform:
    def __init__(self, stats):
        self.Nomarize=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ])
        
    def __call__(self, img):
        return self.Nomarize(img)


def plot_embed(teacher_without_ddp, epoch, output_path, logger, dataset):
    teacher_without_ddp.eval()
    print(os.getcwd())
    if "FashionMNIST" in dataset:
        stats = ((0.2860,), (0.3530,))
        eval_dataset = datasets.FashionMNIST(root="/workspace/angular_dino/datasets", download=False, train=False, transform=eval_transform(stats))
    elif "MNIST" in dataset:
        stats = ((0.1307,), (0.3081,))
        eval_dataset = datasets.MNIST(root="/workspace/angular_dino/datasets", download=False, train=False, transform=eval_transform(stats))
    elif "CIFAR10" in dataset:
        stats = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        eval_dataset = datasets.CIFAR10(root="/workspace/angular_dino/datasets/CIFAR10", download=False, train=False, transform=eval_transform(stats))
    elif "SVHN" in dataset:
        stats = ((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197))
        eval_dataset = datasets.SVHN(root="/workspace/angular_dino/datasets/SVHN", download=False, split="test", transform=eval_transform(stats))
    eval_loader = DataLoader(eval_dataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)
    digit_out = defaultdict(list)

    for batch in eval_loader:
        img, label = batch
        img = img.to("cuda")
        with torch.no_grad():
            output = teacher_without_ddp.backbone(img)
        for out, l in zip(output, label):
            digit_out[l.item()].append(out.cpu().numpy())
    for i in digit_out.keys():
        digit_out[i] = np.vstack(digit_out[i])
    digit_out = {k: v for k, v in sorted(digit_out.items())}

    colors = [
        "#1f77b4",
        "#d62728",
        "#ff7f0e",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    plt.figure()
    plot_num = 10

    ########
    emb_dim = out.shape[-1]
    targets_indices = []
    pre_len = 0
    for i in digit_out:
        cur_len = len(digit_out[i])
        cur_tar = np.array(random.choices(list(range(pre_len, cur_len+pre_len)), k=plot_num))
        targets_indices.append(cur_tar)
        pre_len = cur_len + pre_len
    conc_out = np.vstack(list(digit_out.values()))
    if emb_dim != 2:
        # tsne = TSNE(n_components=2, random_state = 0, perplexity = 10, n_iter = 1000)
        _umap = umap.UMAP(n_components=2)
        conc_out = _umap.fit_transform(conc_out)
    for n, i in enumerate(targets_indices):
        plt.scatter(*conc_out[i][0], label=n, color=colors[n])
        for k in range(1, plot_num):
            plt.scatter(*conc_out[i][k], color=colors[n])
    ########

    # for key in digit_out:
    #     x = random.choices(digit_out[key], k=plot_num)
    #     class_id = key
    #     plt.scatter(*x[0], label=class_id, color=colors[class_id])
    #     for i in range(1,plot_num):
    #         plt.scatter(*x[i], color=colors[class_id])

    plt.title(f"dino embedding ep:{epoch}")
    plt.legend(loc='upper center', bbox_to_anchor=(.5, -0.0), ncol=5, fontsize=7)
    # if logger:
    #     print("logging plt")
    #     # logger.report_matplotlib_figure(title="backbone emb", series="MNIST", iteration=0, figure=plt, report_image=True)
    # logger.report_image(title="image", series="series", iteration=epoch, local_path="/workspace/angular_dinoNIR_FP0001_1_DOWN_a.jpg")
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    plt.savefig(os.path.join(output_path, f"images/dino_emb_ep{epoch}.pdf"), bbox_inches='tight')
    plt.close()
    
    # plt.savefig(f"./images/dino_emb_ep{epoch}.pdf", bbox_inches='tight')
    teacher_without_ddp.train()


def load_pretrained_backbone_head(model, pretrained_weights, checkpoint_key):
    state_dict = torch.load(pretrained_weights, map_location="cpu")["teacher"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    backbone_state = {k.replace("backbone.", ""):v for k,v in state_dict.items() if k.startswith("backbone")}
    head_state = {k.replace("head.", ""):v for k,v in state_dict.items() if k.startswith("head")}
    model.backbone.load_state_dict(backbone_state, strict=False)
    model.head.load_state_dict(head_state, strict=False)
    del model.head.last_layer


def converge_cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, converge_epoch=None):
    """
    Cosine scheduler with early convergence.

    Args:
        base_value: 初期学習率。
        final_value: 最終学習率。
        epochs: 全エポック数。
        niter_per_ep: 各エポックあたりのイテレーション数。
        warmup_epochs: ウォームアップエポック数。
        start_warmup_value: ウォームアップ時の初期学習率。
        converge_epoch: 学習率を最終値に収束させるエポック。

    Returns:
        np.ndarray: 学習率スケジュール。
    """

    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    total_iters = epochs * niter_per_ep
    if converge_epoch is None:
        converge_epoch = epochs
    converge_iter = converge_epoch * niter_per_ep

    cosine_iters = converge_iter - warmup_iters
    cosine_schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * np.arange(cosine_iters) / cosine_iters))

    # converge_iter以降はfinal_valueを保持
    remaining_iters = total_iters - converge_iter
    constant_schedule = np.full(remaining_iters, final_value)

    schedule = np.concatenate((warmup_schedule, cosine_schedule, constant_schedule))
    assert len(schedule) == total_iters
    return schedule


class Random_delate(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p
        self.kernel = np.ones((3, 3), np.uint8)

    def __call__(self, img):
        if random.random() < self.p:
            return Image.fromarray(cv2.dilate(np.array(img), self.kernel, iterations=1))
        else:
            return img


class DataAugmentationDINOMNIST(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        # MNISTに適した基本的な変換
        basic_transform = transforms.Compose([
            transforms.RandomRotation(degrees=15),  # ±15度の回転
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
            Random_delate(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=global_crops_scale, interpolation=Image.BICUBIC),
            basic_transform,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=global_crops_scale, interpolation=Image.BICUBIC),
            basic_transform,
            utils.GaussianBlur(0.1),
            normalize,
        ])
        
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(21, scale=local_crops_scale, interpolation=Image.BICUBIC),
            basic_transform,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        # image = image.resize((28, 28))
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

class DataAugmentationDINOFashionMNIST(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        # MNISTに適した基本的な変換
        flip_and_color_jitter = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=15), 
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            normalize,
        ])
        
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(21, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        # image = image.resize((28, 28))
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

class DataAugmentationDINOCIFAR10(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        # CIFAR10に適した基本的な変換
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=15), 
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(24, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        # image = image.resize((28, 28))
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        # crops.append(image)
        return crops

class Weak_DataAugmentationDINOCIFAR10(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        # CIFAR10に適した基本的な変換
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=15), 
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=[0.8, 1.0], interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            # utils.GaussianBlur(1.0),
            normalize,
        ])
        
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=[0.8, 1.0], interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            # utils.GaussianBlur(0.1),
            # utils.Solarization(0.2),
            normalize,
        ])
        
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(24, scale=[0.6, 0.8], interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            # utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        # image = image.resize((28, 28))
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        # crops.append(image)
        return crops

class DataAugmentationDINOSVHN(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, resize=32, invert=False):
        # CIFAR10に適した基本的な変換
        invert = 0.2 if invert==True else 0.0
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=15), 
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            # transforms.RandomInvert(p=invert),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(resize, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(resize, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(int(resize*(0.75)), scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        # image = image.resize((28, 28))
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        # crops.append(image)
        return crops

class Smallscale_DataAugmentationDINOSVHN(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, resize=32):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(resize, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(resize, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(resize//2, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        # image = image.resize((28, 28))
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        # crops.append(image)
        return crops


def my_train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args, logger, unl_normalize_subset, pos_normalize_subset, lambda_schedule,
                    student_prototype_weight, teacher_prototype_weight, s_prototype_optimizer, emb_loss, test_dataset, val_dataset,
                    normal_class, pu_normal_prototype_dict, pu_positive_vec_dict):
    torch.autograd.set_detect_anomaly(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    epoch_total_valid = []
    for it, (images, class_ids) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        for i, param_group2 in enumerate(s_prototype_optimizer.param_groups):
            param_group2["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group2["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]# + [im.cuda(non_blocking=True) for im in pos_tra_samples]

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(enabled=args.precision != 'fp32', dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float16):
            (teacher_output, max_id), t_emb = teacher(images[:2], "teacher", epoch=epoch)  # only the 2 global views pass through the teacher
            (student_output, _), s_emb = student(images, "student", max_id=None, epoch=epoch)
            loss_dino = dino_loss(student_output, teacher_output, epoch, it, 
                                                class_ids=class_ids, pos_output=False, pos_batch=False, use_weight=args.use_weight)
            
            s_dot = student_prototype_weight(s_emb)
            t_dot = teacher_prototype_weight(t_emb)
            
            loss_emb = emb_loss(s_dot, t_dot, epoch, it, 
                                class_ids, pos_output=False, pos_batch=False, use_weight=args.use_weight, margin_type=args.margin_type, 
                                pu_normal_prototype_dict=pu_normal_prototype_dict, pu_positive_vec_dict=pu_positive_vec_dict, pt_num=args.pt_num)
            report_margin = emb_loss.margin * emb_loss.warmup[it]
 
            loss = loss_dino + lambda_schedule[it]*loss_emb # + pos_batch_emb_loss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        s_prototype_optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward(retain_graph=True)
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            # fp16_scaler.update()
        ######
        #update extra optimizer
        if args.clip_grad:
            fp16_scaler.unscale_(s_prototype_optimizer)  # unscale the gradients of optimizer's assigned params in-place
            param_norms = utils.clip_gradients(student_prototype_weight, args.clip_grad)
        if epoch < args.freeze_last_layer:
            for n, p in student_prototype_weight.named_parameters():
                    p.grad = None
        fp16_scaler.step(s_prototype_optimizer)
        fp16_scaler.update()
        ######

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(student_prototype_weight.module.parameters(), teacher_prototype_weight.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        #####Validation#####
        # valid_loss, val_loss_dino, val_loss_emb = validation(val_dataset, teacher, student, teacher_prototype_weight, student_prototype_weight, dino_loss, emb_loss, epoch)
        # epoch_total_valid.append(valid_loss)
        # logging
        torch.cuda.synchronize()
        # breakpoint()
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(loss_dino=loss_dino.item())
        metric_logger.update(loss_lambda_emb=lambda_schedule[it]*loss_emb.item())
        metric_logger.update(loss_emb=loss_emb.item())
        metric_logger.update(emb_s_temp=emb_loss.student_temp)
        metric_logger.update(margin=report_margin)

        # metric_logger.update(valid_loss=valid_loss)
        # metric_logger.update(val_loss_emb=val_loss_dino)
        # metric_logger.update(val_loss_emb=val_loss_emb)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(lambda_=lambda_schedule[it])

        
        if utils.is_main_process() and logger:
            logger.report_scalar("train", "total_loss", iteration=it, value=loss.item())
            logger.report_scalar("train", "dino_loss", iteration=it, value=loss_dino.item())
            logger.report_scalar("train", "lambda_emb_loss", iteration=it, value=lambda_schedule[it]*loss_emb.item())
            logger.report_scalar("train", "emb_loss", iteration=it, value=loss_emb.item())

            # logger.report_scalar("train", "valid_loss", iteration=it, value=valid_loss)
            # logger.report_scalar("train", "val_loss_dino", iteration=it, value=val_loss_dino)
            # logger.report_scalar("train", "val_loss_emb", iteration=it, value=val_loss_emb)

            logger.report_scalar("margin", "margin", iteration=it, value=report_margin)
            logger.report_scalar("temp", "emb_student", iteration=it, value=emb_loss.student_temp)
            logger.report_scalar("Learning_Rate", "lr", iteration=it, value=optimizer.param_groups[0]["lr"])
            logger.report_scalar("Weight decay", "wb", iteration=it, value=optimizer.param_groups[0]["weight_decay"])
            logger.report_scalar("lambda", "lb", iteration=it, value=lambda_schedule[it])
        
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # print(f"normal_prototypevec_ids:{normal_prototype_vec_id}, positive_prototypevec_ids{postive_prototype_vec_ids}")
    # print(f"detached_push_loss: \n{detached_push_loss}")
    auroc, pu_normal_prototype_dict, pu_positive_vec_dict = test_anomaly_detection(teacher_without_ddp, unl_normalize_subset, pos_normalize_subset, test_dataset, teacher_prototype_weight, args.output_dir, epoch, args.pt_num)
    if utils.is_main_process() and logger:
            logger.report_scalar("test", "auroc", iteration=epoch, value=auroc)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, 1e6, pu_normal_prototype_dict, pu_positive_vec_dict #np.mean(epoch_total_valid)

def validation(val_datset, teacher, student, teacher_prototype_weight, student_prototype_weight, dino_loss, emb_loss, epoch):
    # breakpoint()
    teacher.eval()
    student.eval()
    teacher_prototype_weight.eval()
    student_prototype_weight.eval()
    dino_loss.eval()
    emb_loss.eval()
    test_loader = DataLoader(val_datset, batch_size=100, shuffle=True)
    for samples, gts in test_loader:
        images = [im.cuda(non_blocking=True) for im in samples]
        with torch.no_grad():
            (teacher_output, max_id), t_emb = teacher(images[:2], "teacher", epoch=epoch)  # only the 2 global views pass through the teacher
            (student_output, _), s_emb = student(images, "student", max_id=None, epoch=epoch)
            loss_dino, *_ = dino_loss(student_output, teacher_output, epoch, iteration=-1)
            loss_dino = loss_dino.cpu().item()

            s_dot = student_prototype_weight(s_emb)
            t_dot = teacher_prototype_weight(t_emb)
            loss_emb, *_ = emb_loss(s_dot, t_dot, epoch, iteration=-1, class_ids=None)
            loss_emb = loss_emb.cpu().item()
            total_loss = loss_dino + loss_emb
            total_loss = total_loss
    teacher.train()
    student.train()
    teacher_prototype_weight.train()
    student_prototype_weight.train()
    dino_loss.train()
    emb_loss.train()
    return total_loss, loss_dino, loss_emb


class backbone_emb_Loss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, args=None, logger=None, unl_pos_weight=None, student_temp_schedule=None, 
                 margin_settings=None):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.batch_size = args.batch_size_per_gpu
        self.logger = logger
        self.unl_pos_weight = unl_pos_weight
        self.student_temp_schedule = student_temp_schedule
        self.register_buffer('m', torch.zeros(args.batch_size_per_gpu, args.prototype_num))

        if margin_settings is not None:
            self.use_margin = True
            self.margin = margin_settings["m"]
            self.easy_margin = margin_settings["easy_margin"]
            self.warmup = margin_settings["warmup"]
        else:
            self.use_margin = False
            self.margin = 0.0

    def forward(self, student_output, teacher_output, epoch, iteration=0, 
                class_ids=None, student_raw_out=None, pos_output=False, pos_batch=False, 
                use_weight=False, margin_type=False, pu_normal_prototype_dict=None, pu_positive_vec_dict=None, pt_num=10):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        if self.student_temp_schedule is not None:
            self.student_temp = self.student_temp_schedule[iteration]
        # student_out = student_output / self.student_temp
        # student_out = student_out.chunk(self.ncrops)
        #without temp scaling
        student_out = student_output.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]

        # teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        #without teacher softmax
        # teacher_out = (teacher_output - self.center) / temp
        #without teacher softmax and temp
        teacher_out = (teacher_output - self.center)
        teacher_out = teacher_out.detach().chunk(2)

        if use_weight and torch.is_tensor(class_ids):# or isinstance(class_ids, list):
            unl_pos = class_ids[:,0] #batch_size  [0,0,0,1,0,0,0,...]
            batch_weight = (1-unl_pos)*self.unl_pos_weight["unl"] + unl_pos*self.unl_pos_weight["pos"]
            batch_weight = batch_weight.to("cuda")
        else:
            batch_weight = None

        ##########################################
        freeze_ep=50
        if epoch>freeze_ep and margin_type != False and self.margin > 0.0:
            unl_pos = class_ids[:,0] #batch_size  [0,0,0,1,0,0,0,...] pos=1
            norm_margin_index = torch.tensor(list(pu_normal_prototype_dict.keys())[:pt_num]) #5
            pos_margin_index = torch.tensor(list(pu_positive_vec_dict.keys())[:pt_num]) #5
            margin=True
        else:
            margin = None
        ##########################################

        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                if margin == None:
                    # loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1) 
                    loss = torch.sum(-F.softmax(q/temp, dim=-1) * F.log_softmax(student_out[v]/self.student_temp, dim=-1), dim=-1)              
                elif margin_type.lower() == 'student':
                    loss = torch.sum(-F.softmax(q/temp, dim=-1) * F.log_softmax(self.additive_margin(student_out[v], unl_pos, norm_margin_index, pos_margin_index, iteration)/self.student_temp, dim=-1), dim=-1)
                elif margin_type.lower() == 'teacher':
                    loss = torch.sum(-F.softmax(self.additive_margin(q, unl_pos, norm_margin_index, pos_margin_index, iteration, sub=True)/temp, dim=-1) * F.log_softmax(student_out[v]/self.student_temp, dim=-1), dim=-1)
                elif margin_type.lower() == 'tea-stu' or margin_type.lower() == 'stu-tea':
                    loss = torch.sum(-F.softmax(self.additive_margin(q, unl_pos, norm_margin_index, pos_margin_index, iteration, sub=True)/temp, dim=-1) * F.log_softmax(self.additive_margin(student_out[v], unl_pos, norm_margin_index, pos_margin_index, iteration)/self.student_temp, dim=-1), dim=-1)
                else:
                    raise NotImplementedError

                if batch_weight != None:
                    loss = loss * batch_weight
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        #####
        if self.logger and (iteration%100==0 or iteration==0):
            if class_ids is not None:
                self.visualize_output(student_output.detach().clone().chunk(self.ncrops), self.student_temp, teacher_output.detach().clone().chunk(2), self.center, temp, iteration, class_ids[:,1])
        #####
        return total_loss
        
    def additive_margin(self, cosine, unl_pos, norm_m_ids, pos_m_ids, it, sub=False):
        # breakpoint()
        dtype = cosine.dtype
        margin =  self.margin * self.warmup[it]
        cos_m = math.cos(margin)
        sin_m = math.sin(margin)

        ## Additive
        if not sub:
            th = math.cos(math.pi - margin)
            mm = math.sin(math.pi - margin) * margin

            ### Additive
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-7, 1))
            phi = cosine * cos_m - sine * sin_m
            phi = phi.to(dtype)
            phi = torch.clamp(phi, -1 + 1e-7, 1 - 1e-7)
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > th, phi, cosine - mm)
        
        ### Subtractive
        else:
            s_th = math.cos(margin)
            s_mm = math.sin(margin) * margin

            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-7, 1))
            phi = cosine * cos_m + sine * sin_m
            phi = phi.to(dtype)
            phi = torch.clamp(phi, -1 + 1e-7, 1 - 1e-7)
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine < s_th, phi, cosine + s_mm)

        pos_row = unl_pos.nonzero().squeeze()
        unl_row = (1-unl_pos).nonzero().squeeze()

        norm_margin_pos_ids = torch.cartesian_prod(pos_row, norm_m_ids)
        norm_margin_unl_ids = torch.cartesian_prod(unl_row, norm_m_ids)
        pos_margin_pos_ids = torch.cartesian_prod(pos_row, pos_m_ids)
        pos_margin_unl_ids = torch.cartesian_prod(unl_row, pos_m_ids)

        one_hot = torch.zeros(cosine.size(), device='cuda') #1000*65536 (10*B)*65536
        for carten in [norm_margin_unl_ids, pos_margin_pos_ids]:
            one_hot[carten[:,0], carten[:,1]] = 1

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4

        return output

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    def visualize_output(self, student_output, student_temp, teacher_output, center, teacher_temp, iteration, class_ids):
        # breakpoint()
        unique_indices = defaultdict(list)
        class_ids = class_ids.tolist()
        for index, value in enumerate(class_ids):
            unique_indices[value].append(index)
        unique_indices = dict(sorted(unique_indices.items(), key=lambda x:x[0]))
        selected_indices = {k:random.choice(list(v)) for k,v in unique_indices.items()}

        ### plotting each class teacher softmax emb
        for class_id, index in selected_indices.items():
            plot_teacher_0=teacher_output[0][index].cpu().float().numpy() #emb_dim
            self.logger.report_histogram(
            f"iteration:{iteration} class_id:{class_id} teacher_emb view:0 w/o softmax",
            "value",
            values=plot_teacher_0,
            xaxis="title x", yaxis="title y",
            )

            plot_teacher_0=teacher_output[0][index].cpu().float() #emb_dim
            plot_teacher_0 = (F.softmax((plot_teacher_0 - center.cpu())/teacher_temp, dim=-1)).numpy() #emb_dim
            self.logger.report_histogram(
            f"iteration:{iteration} class_id:{class_id} teacher_emb view:0 softmax S:{teacher_temp.item():.2f}, C:{center.mean().item():.2f}",
            "value",
            values=plot_teacher_0,
            xaxis="title x", yaxis="title y",
            )

            plot_teacher_1=teacher_output[1][index].cpu().float().numpy() #emb_dim
            self.logger.report_histogram(
            f"iteration:{iteration} class_id:{class_id} teacher_emb view:1 w/o softmax",
            "value",
            values=plot_teacher_1,
            xaxis="title x", yaxis="title y",
            )

            plot_teacher_1=teacher_output[1][index].cpu().float() #emb_dim
            plot_teacher_1 = (F.softmax((plot_teacher_1 - center.cpu())/teacher_temp, dim=-1)).numpy()#emb_dim
            self.logger.report_histogram(
            f"iteration:{iteration} class_id:{class_id} teacher_emb view:1 softmax S:{teacher_temp.item():.2f}, C:{center.mean().item():.2f}",
            "value",
            values=plot_teacher_1,
            xaxis="title x", yaxis="title y",
            )
            ### plotting student emb
            plot_view = np.random.randint(0, self.ncrops)
            plot_student=(student_output[plot_view][index]).cpu().float().numpy()
            self.logger.report_histogram(
                f"iteration:{iteration} class_id:{class_id} student_emb view:{plot_view} w/o sharpning, softmax",
                "value",
                # iteration=epoch,
                values=plot_student,
                xaxis="title x", yaxis="title y",
            )
            ### plotting student emb softmax
            plot_view = np.random.randint(0, self.ncrops)
            plot_student=(F.softmax(student_output[plot_view][index]/student_temp, dim=-1)).cpu().float().numpy()
            self.logger.report_histogram(
                f"iteration:{iteration} class_id:{class_id} student_emb view:{plot_view} w/ sharpning{student_temp}, softmax",
                "value",
                # iteration=epoch,
                values=plot_student,
                xaxis="title x", yaxis="title y",
            )
       

def split(a, b):
    # 連結
    a = np.array(a)
    b = np.array(b)
    combined = np.concatenate((a, b))
    print(len(combined))

    np.random.shuffle(combined)

    # 分割
    split_index = int(len(combined) * 0.9)
    train_indices, val_indices = np.split(combined, [split_index])
    print(len(train_indices))
    train_unlabel = []
    train_positive = []
    for i in train_indices:
        if i in a: train_unlabel.append(i)
        elif i in b: train_positive.append(i)
        else:
            print(i, a, b)
            return

    return train_unlabel, train_positive, val_indices


def get_prototype_vec_ids_from_loader(model, unl_normalize_subset, pos_normalize_subset, prototype_vec_weight=None):
    num_unlabel = 0
    num_positive = 0

    num_unlabel = len(unl_normalize_subset)
    num_positive = len(pos_normalize_subset)

    pu_unlabel_vec_dict = defaultdict(float)
    pu_positive_vec_dict = defaultdict(float)
    # breakpoint()
    #####
    unl_normalize_loader = DataLoader(unl_normalize_subset, batch_size=200, shuffle=True, num_workers=8, pin_memory=True)
    pos_normalize_loader = DataLoader(pos_normalize_subset, batch_size=50, shuffle=True, num_workers=8, pin_memory=True)


    for images, unl_pos in unl_normalize_loader:
        images = images.cuda(non_blocking=True)
        if not prototype_vec_weight:
            with torch.no_grad():
                output = model(images)[0][0]
        else:
            with torch.no_grad():
                output = prototype_vec_weight(model.backbone(images))
        output =  output.cpu().detach()
        sorted_out, sorted_arg = torch.sort(output, dim=1, descending=True)
        sorted_out, sorted_arg = sorted_out.numpy()[:, :5], sorted_arg.numpy()[:, :5]
        for n, (out, arg) in enumerate(zip(sorted_out, sorted_arg)):
            for o, a in zip(out, arg):
                pu_unlabel_vec_dict[a] += o/num_unlabel
            # if unl_pos[n] == 0:
            #     for o, a in zip(out, arg):
            #         pu_unlabel_vec_dict[a] += o/num_unlabel
            # elif unl_pos[n] == 1:
            #     for o, a in zip(out, arg):
            #         pu_positive_vec_dict[a] += o/num_positive
            # else: raise NotImplementedError
    
    for images, unl_pos in pos_normalize_loader:
        images = images.cuda(non_blocking=True)
        if not prototype_vec_weight:
            with torch.no_grad():
                output = model(images)[0][0]
        else:
            with torch.no_grad():
                output = prototype_vec_weight(model.backbone(images))
        output =  output.cpu().detach()
        sorted_out, sorted_arg = torch.sort(output, dim=1, descending=True)
        sorted_out, sorted_arg = sorted_out.numpy()[:, :5], sorted_arg.numpy()[:, :5]
        for n, (out, arg) in enumerate(zip(sorted_out, sorted_arg)):
            for o, a in zip(out, arg):
                pu_positive_vec_dict[a] += o/num_positive
            # if unl_pos[n] == 0:
            #     for o, a in zip(out, arg):
            #         pu_unlabel_vec_dict[a] += o/num_unlabel
            # elif unl_pos[n] == 1:
            #     for o, a in zip(out, arg):
            #         pu_positive_vec_dict[a] += o/num_positive
            # else: raise NotImplementedError

    pu_unlabel_vec_dict = dict(sorted(pu_unlabel_vec_dict.items(), key=lambda x: x[1], reverse=True))
    pu_positive_vec_dict = dict(sorted(pu_positive_vec_dict.items(), key=lambda x: x[1], reverse=True))
    pu_normal_prototype_dict = pu_unlabel_vec_dict.copy()

    for k in pu_positive_vec_dict:
        if k in pu_normal_prototype_dict.keys():
            pu_normal_prototype_dict[k] -= pu_positive_vec_dict[k]

    pu_normal_prototype_dict = dict(sorted(pu_normal_prototype_dict.items(), key=lambda x: x[1], reverse=True))
    pu_normal_prototype_dict_id = list(pu_normal_prototype_dict.keys())[0]
    return  pu_unlabel_vec_dict, pu_positive_vec_dict, pu_normal_prototype_dict, pu_normal_prototype_dict_id

def calc_y_score(model, test_dataset, pu_normal_prototype_dict_id, pu_unlabel_vec_dict, pu_positive_vec_dict, pu_normal_prototype_dict, prototype_vec_weight=None, output=None, pt_num=10):
    # normal_prototype_vec = head_state["last_layer.weight_v"][pu_normal_prototype_dict_id,:]
    # breakpoint()


    if not prototype_vec_weight:
        normal_prototype_vec = model.head.last_layer.weight_v.detach().cpu().clone()[pu_normal_prototype_dict_id,:]
        normal_prototype_vec /= torch.norm(normal_prototype_vec, p=2)
    else:
        # normal_prototype_vec = prototype_vec_weight.weight.data.detach().cpu().clone()[pu_normal_prototype_dict_id,:]
        normal_prototype_vec = prototype_vec_weight.weight.data.detach().cpu().clone()
    # print_f(f"正常代表ベクトルのshape: {normal_prototype_vec.shape}, norm: {normal_prototype_vec.norm(p=2)}", output+"/result.txt")

    ################
    top5_normal_prototype_ids = list(pu_normal_prototype_dict.keys())[:pt_num]
    top5_positive_prototype_ids = list(pu_positive_vec_dict.keys())[:pt_num]
    # top5_normal_prototype_vecs= head_state["last_layer.weight_v"][top5_normal_prototype_ids,:]
    if not prototype_vec_weight:
        top5_normal_prototype_vecs= model.head.last_layer.weight_v.detach().cpu().clone()[top5_normal_prototype_ids,:]
        top5_normal_prototype_vecs= nn.functional.normalize(top5_normal_prototype_vecs, dim=1)
    else:
        top5_normal_prototype_vecs = normal_prototype_vec[top5_normal_prototype_ids,:]
        top5_positive_prototype_vecs = normal_prototype_vec[top5_positive_prototype_ids,:]
    print_f(f"top5正常代表ベクトルのindex: {top5_normal_prototype_ids}", output+"/result.txt")
    print_f(f"top5異常代表ベクトルのindex: {top5_positive_prototype_ids}", output+"/result.txt")
    # print_f(f"top5正常代表ベクトルのshape: {top5_normal_prototype_vecs.shape}, norm: {top5_normal_prototype_vecs.norm(dim=1, p=2)}", output+"/result.txt")
    top5_normal_prototype_wights = list(pu_normal_prototype_dict.values())[:pt_num]
    top5_positive_prototype_wights = list(pu_positive_vec_dict.values())[:pt_num]
    print_f(f"top5_normal_prototype_weights: {[round(i,4) for i in top5_normal_prototype_wights]}", output+"/result.txt")
    print_f(f"top5_positive_prototype_weights: {[round(i,4) for i in top5_positive_prototype_wights]}", output+"/result.txt")
    # top5_normal_prototype_wights = nn.functional.softmax(torch.tensor(top5_normal_prototype_wights)/0.1, dim=0, dtype=torch.float32)
    # print_f(f"top5_normal_prototype_softmax_weights: {top5_normal_prototype_wights}", output+"/result.txt")
    ################

    random.seed(42)
    #正常データのindex 
    # try: indices = val_dataset.targets
    # except: indices = val_dataset.labels
    

    # if not torch.is_tensor(indices):
    #     indices = torch.tensor(indices)
    # normal_indices = (indices==1).nonzero().flatten()
    # normal_indices = random.sample(normal_indices.tolist(), 1000)
    # #異常データのindex
    # anomaly_indices = (indices!=1).nonzero().flatten()
    # anomaly_indices = random.sample(anomaly_indices.tolist(), 1000)

    #結果の配列
    y_true = []
    y_score = []
    y_each_class = defaultdict(list)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=8, pin_memory=True)
    for samples, gts in test_loader:
        # sample, gt = val_dataset.__getitem__(i)
        # sample = sample.unsqueeze(0)
        y_true.extend(gts.tolist())
        with torch.no_grad():
            backbone_emb = model.backbone(samples.cuda(non_blocking=True)).cpu() #b * 128
            # score = (backbone_emb @ top5_normal_prototype_vecs.T).mean(dim=1).tolist() #meanではなくmaxも試す 

            score = (backbone_emb @ top5_normal_prototype_vecs.T).mean(dim=1) - \
                    (backbone_emb @ top5_positive_prototype_vecs.T).mean(dim=1)
            score = score.squeeze().tolist()
            y_score.extend(score)

    y_score = np.array(y_score)
    y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
    y_score = 1 - y_score
    # y_each_class = dict(sorted(y_each_class.items(), key=lambda x: x[0]))
    return y_true, y_score#, y_each_class

def test_anomaly_detection(teacher, unl_normalize_subset, pos_normalize_subset, test_dataset, teacher_prototype_weight, output, epoch, pt_num=10):
    teacher.eval()
    teacher_prototype_weight.eval()
    # breakpoint()
    print_f(f"--------epoch: {epoch}--------", output+"/result.txt")

    pu_unlabel_vec_dict, pu_positive_vec_dict, pu_normal_prototype_dict, pu_normal_prototype_dict_id = \
        get_prototype_vec_ids_from_loader(teacher, unl_normalize_subset, pos_normalize_subset, teacher_prototype_weight)
    print_f(f"unlabel_prototype_vec_ids = {list(pu_unlabel_vec_dict.keys())[:10]}, total dot = {list(map(lambda x: round(x, 3), pu_unlabel_vec_dict.values()))[:10]}", output+"/result.txt")
    print_f(f"positive_prototype_vec_ids = {list(pu_positive_vec_dict.keys())[:10]}, total dot = {list(map(lambda x: round(x, 3), pu_positive_vec_dict.values()))[:10]}", output+"/result.txt")
    print_f(f"noraml_prototype_vec_ids = {list(pu_normal_prototype_dict.keys())[:10]}, total dot = {list(map(lambda x: round(x, 3), pu_normal_prototype_dict.values()))[:10]}", output+"/result.txt")
    print_f(f"noraml_prototype_vec_ids = {pu_normal_prototype_dict_id}", output+"/result.txt")

    print("calculating score")
    y_true, y_score = calc_y_score(teacher, test_dataset, pu_normal_prototype_dict_id, pu_unlabel_vec_dict, pu_positive_vec_dict, pu_normal_prototype_dict, teacher_prototype_weight, output, pt_num)
    # breakpoint()
    auroc = roc_auc_score(y_true, y_score)
    print_f(f"AUROC: {auroc}", output+"/result.txt")
    teacher.train()
    teacher_prototype_weight.train()
    return auroc, pu_normal_prototype_dict, pu_positive_vec_dict

def print_f(text, file):
    print(text)
    with open(file, mode="a") as f:
        f.write(text+"\n")