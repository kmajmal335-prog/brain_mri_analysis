import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch


class BrainMRIAugmentation:
    def __init__(self, mode='train', img_size=(224, 224)):
        self.mode = mode
        self.img_size = img_size

        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                    p=0.3
                ),
                A.GridDistortion(p=0.3),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

    def __call__(self, image, mask=None):
        if mask is not None:
            augmented = self.transform(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        else:
            augmented = self.transform(image=image)
            return augmented['image']