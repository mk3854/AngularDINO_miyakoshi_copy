"""
可視化・プロトタイプ推定・評価系の関数
"""
import os
import random
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import umap
from torchvision import datasets, transforms

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
        eval_dataset = datasets.FashionMNIST(root="./datasets", download=False, train=False, transform=eval_transform(stats))
    elif "MNIST" in dataset:
        stats = ((0.1307,), (0.3081,))
        eval_dataset = datasets.MNIST(root="./datasets", download=False, train=False, transform=eval_transform(stats))
    elif "CIFAR10" in dataset:
        stats = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        eval_dataset = datasets.CIFAR10(root="./datasets/CIFAR10", download=False, train=False, transform=eval_transform(stats))
    elif "SVHN" in dataset:
        stats = ((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197))
        eval_dataset = datasets.SVHN(root="./datasets/SVHN", download=False, split="test", transform=eval_transform(stats))

    # 各クラスから100サンプルずつ選択
    samples_per_class = 100
    selected_indices = []
    for class_idx in range(len(eval_dataset.classes)):
        class_indices = [i for i, (_, label) in enumerate(eval_dataset) if label == class_idx]
        selected_indices.extend(np.random.choice(class_indices, samples_per_class, replace=False))
    
    # 選択したサンプルでサブセットを作成
    subset_dataset = torch.utils.data.Subset(eval_dataset, selected_indices)
    eval_loader = DataLoader(subset_dataset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

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
        "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    plt.figure()
    plot_num = 10
    emb_dim = list(digit_out.values())[0].shape[-1]
    targets_indices = []
    pre_len = 0
    for i in digit_out:
        cur_len = len(digit_out[i])
        cur_tar = np.array(random.choices(list(range(pre_len, cur_len+pre_len)), k=plot_num))
        targets_indices.append(cur_tar)
        pre_len = cur_len + pre_len
    
    conc_out = np.vstack(list(digit_out.values()))
    
    if emb_dim != 2:
        _umap = umap.UMAP(n_components=2, n_jobs=1)
        conc_out = _umap.fit_transform(conc_out)
    
    for n, i in enumerate(targets_indices):
        plt.scatter(*conc_out[i][0], label=n, color=colors[n])
        for k in range(1, plot_num):
            plt.scatter(*conc_out[i][k], color=colors[n])
            
    plt.title(f"dino embedding ep:{epoch}")
    plt.legend(loc='upper center', bbox_to_anchor=(.5, -0.0), ncol=5, fontsize=7)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    plt.savefig(os.path.join(output_path, f"images/dino_emb_ep{epoch}.pdf"), bbox_inches='tight')
    plt.close()
    teacher_without_ddp.train()

def test_anomaly_detection(teacher, unl_normalize_subset, pos_normalize_subset, test_dataset, teacher_prototype_weight, output, epoch, pt_num=10):
    """
    異常検出の評価を行う関数
    """
    import torch
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score
    from utils_extra import print_f

    # モデルを評価モードに設定
    teacher.eval()
    teacher_prototype_weight.eval()

    # プロトタイプベクトルのIDと重みを取得
    positive_sim_dict, normal_sim_dict = \
        get_prototype_vec_ids_from_loader(teacher, unl_normalize_subset, pos_normalize_subset, teacher_prototype_weight)

    # 結果を出力
    print_f(f"--------epoch: {epoch}--------", output+"/result.txt")
    # print_f(f"unlabel_prototype_vec_ids = {list(pu_unlabel_vec_dict.keys())[:10]}, total dot = {list(map(lambda x: round(x, 3), pu_unlabel_vec_dict.values()))[:10]}", output+"/result.txt")
    # 学習データから推定されたプロトタイプのIDと、埋め込みとプロトタイプの類似度の合計
    print_f(f"positive_prototype_ids = {list(positive_sim_dict.keys())[:pt_num]}", output+"/result.txt")
    print_f(f"total dot = {list(map(lambda x: round(x, 3), positive_sim_dict.values()))[:pt_num]}", output+"/result.txt")
    print_f(f"normal_prototype_ids = {list(normal_sim_dict.keys())[:pt_num]}", output+"/result.txt")
    print_f(f"total dot = {list(map(lambda x: round(x, 3), normal_sim_dict.values()))[:pt_num]}", output+"/result.txt")

    # スコアを計算
    print("calculating score")
    y_true, y_score = calc_y_score(teacher, test_dataset, positive_sim_dict, normal_sim_dict, teacher_prototype_weight, output, pt_num)
    auroc = roc_auc_score(y_true, y_score)  # AUROCを計算
    print_f(f"AUROC: {auroc}", output+"/result.txt")

    # モデルを学習モードに戻す
    teacher.train()
    teacher_prototype_weight.train()
    return auroc, normal_sim_dict, positive_sim_dict

def get_prototype_vec_ids_from_loader(model, unl_normalize_subset, pos_normalize_subset, prototype_vec_weight):
    """
    教師なしデータと正例データからプロトタイプベクトルのIDと重みを計算する関数
    """
    from torch.utils.data import DataLoader
    import torch
    import numpy as np
    from collections import defaultdict

    # データセットのサイズを取得
    num_unlabel = len(unl_normalize_subset)  # 教師なしデータの数
    num_positive = len(pos_normalize_subset)  # 正例データの数

    # プロトタイプベクトルの重みを格納する辞書
    unlabel_sim_dict = defaultdict(float)  # 教師なしデータのプロトタイプベクトル重み
    positive_sim_dict = defaultdict(float)  # 正例データのプロトタイプベクトル重み

    # データローダーの設定
    unl_normalize_loader = DataLoader(unl_normalize_subset, batch_size=200, shuffle=True, num_workers=8, pin_memory=True)
    pos_normalize_loader = DataLoader(pos_normalize_subset, batch_size=50, shuffle=True, num_workers=8, pin_memory=True)

    # 教師なしデータの処理
    for images, unl_pos in unl_normalize_loader:
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            prototype_emb_sim = prototype_vec_weight(model.backbone(images))  # プロトタイプ重みを適用 (200, 64) * (64,100)
        prototype_emb_sim = prototype_emb_sim.cpu().detach() # 各サンプルにおける埋め込みとプロトタイプの類似度 size = batch_size, all_prototype_num (200, 100)
        sorted_sim, sorted_arg = torch.sort(prototype_emb_sim, dim=1, descending=True)  # 類似度を降順にソート
        sorted_sim, sorted_arg = sorted_sim.numpy()[:, :5], sorted_arg.numpy()[:, :5]  # 上位5つの類似度とそのインデックスを取得
        for n, (sim, arg) in enumerate(zip(sorted_sim, sorted_arg)): #辞書に追加　key: プロトタイプベクトルのインデックス, value: 類似度の合計
            for s, a in zip(sim, arg):
                unlabel_sim_dict[a] += s/num_unlabel  # 重みを正規化して加算

    # 正例データの処理（教師なしデータと同様）
    for images, unl_pos in pos_normalize_loader:
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            prototype_emb_sim = prototype_vec_weight(model.backbone(images))
        prototype_emb_sim = prototype_emb_sim.cpu().detach()
        sorted_sim, sorted_arg = torch.sort(prototype_emb_sim, dim=1, descending=True)
        sorted_sim, sorted_arg = sorted_sim.numpy()[:, :5], sorted_arg.numpy()[:, :5]
        for n, (sim, arg) in enumerate(zip(sorted_sim, sorted_arg)):
            for s, a in zip(sim, arg):
                positive_sim_dict[a] += s/num_positive

    # 辞書を重みの降順にソート
    unlabel_sim_dict = dict(sorted(unlabel_sim_dict.items(), key=lambda x: x[1], reverse=True))
    positive_sim_dict = dict(sorted(positive_sim_dict.items(), key=lambda x: x[1], reverse=True))

    # 正常プロトタイプの類似度を推定（教師なしデータの重みから正例データの重みを引く）
    normal_sim_dict = unlabel_sim_dict.copy()
    for k in positive_sim_dict:
        if k in normal_sim_dict.keys():
            normal_sim_dict[k] -= positive_sim_dict[k]
    normal_sim_dict = dict(sorted(normal_sim_dict.items(), key=lambda x: x[1], reverse=True))

    return positive_sim_dict, normal_sim_dict

def calc_y_score(model, test_dataset, positive_sim_dict, normal_sim_dict, prototype_vec_weight, output=None, pt_num=10):
    """
    異常スコアを計算する関数
    """
    import torch
    import numpy as np
    import random
    import torch.nn as nn
    from collections import defaultdict
    from torch.utils.data import DataLoader

    # プロトタイプベクトルの取得
    prototype_vec = prototype_vec_weight.weight.data.detach().cpu().clone()

    # トップKのプロトタイプIDを取得
    top5_normal_prototype_ids = list(normal_sim_dict.keys())[:pt_num]  # 正常プロトタイプの上位K個
    top5_positive_prototype_ids = list(positive_sim_dict.keys())[:pt_num]  # 正例プロトタイプの上位K個

    # 正常・異常プロトタイプベクトルの取得
    top5_normal_prototype_vecs = prototype_vec[top5_normal_prototype_ids,:]
    top5_positive_prototype_vecs = prototype_vec[top5_positive_prototype_ids,:]

    # スコア計算の準備
    random.seed(42)  # 乱数シードを固定
    y_true = []  # 正解ラベル
    y_score = []  # 異常スコア
    y_each_class = defaultdict(list)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=8, pin_memory=True)

    # バッチごとにスコアを計算 正常度=(正常プロトタイプベクトルとの平均類似度 - 異常プロトタイプベクトルとの平均類似度)
    for samples, gts in test_loader:
        y_true.extend(gts.tolist())
        with torch.no_grad():
            backbone_emb = model.backbone(samples.cuda(non_blocking=True)).cpu()
            score = (backbone_emb @ top5_normal_prototype_vecs.T).mean(dim=1) - \
                    (backbone_emb @ top5_positive_prototype_vecs.T).mean(dim=1)
            score = score.squeeze().tolist()
            y_score.extend(score)

    # スコアの正規化
    y_score = np.array(y_score)
    y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
    y_score = 1 - y_score

    return y_true, y_score
