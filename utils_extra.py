"""
小ユーティリティ関数群
"""

def split(a, b, train_ratio=0.9):
    """
    2つのインデックスリストを結合し、訓練データと検証データに分割する関数
    
    Args:
        a (list): ラベルなしデータのインデックスリスト
        b (list): ラベルありデータのインデックスリスト
        train_ratio (float): 訓練データの割合 (0.0 ~ 1.0)
        
    Returns:
        tuple: (訓練用ラベルなしデータのインデックスリスト, 
               訓練用ラベルありデータのインデックスリスト,
               検証用データのインデックスリスト)
    """
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    combined = np.concatenate((a, b))
    print(len(combined))
    np.random.shuffle(combined)
    split_index = int(len(combined) * train_ratio)
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

def print_f(text, file):
    print(text)
    with open(file, mode="a") as f:
        f.write(text+"\n")
