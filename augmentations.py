"""
データ拡張・前処理系のクラス・関数
"""
import random
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import utils

class Random_delate(object):
    """
    PIL画像にランダムな膨張処理を適用
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
