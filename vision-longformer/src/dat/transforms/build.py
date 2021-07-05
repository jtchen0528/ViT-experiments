# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from timm.data import create_transform
from PIL import ImageFilter
import logging
import random

import torchvision.transforms as T


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_resolution(original_resolution):
    """Takes (H,W) and returns (precrop, crop)."""
    area = original_resolution[0] * original_resolution[1]
    return (160, 128) if area < 96*96 else (512, 480)


def build_transforms(cfg, is_train=True):
    if cfg.MODEL.ARCH.startswith('inception'):
        assert cfg.INPUT.IMAGE_SIZE == 299, "Invalid image size for Inception models!"
    if cfg.AUG.TIMM_AUG.USE_TRANSFORM and is_train:
        logging.info('=> use timm transform for training')
        timm_cfg = cfg.AUG.TIMM_AUG
        transforms = create_transform(
            input_size=cfg.INPUT.IMAGE_SIZE,
            is_training=True,
            use_prefetcher=False,
            no_aug=False,
            re_prob=timm_cfg.RE_PROB,
            re_mode=timm_cfg.RE_MODE,
            re_count=timm_cfg.RE_COUNT,
            scale=cfg.AUG.SCALE,
            ratio=cfg.AUG.RATIO,
            hflip=timm_cfg.HFLIP,
            vflip=timm_cfg.VFLIP,
            color_jitter=timm_cfg.COLOR_JITTER,
            auto_augment=timm_cfg.AUTO_AUGMENT,
            interpolation='bicubic' if cfg.INPUT.INTERPOLATION==3 else 'bilinear',
            mean=cfg.INPUT.MEAN,
            std=cfg.INPUT.STD,
        )

        return transforms

    # assert isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)), 'DATASET.OUTPUT_SIZE should be list or tuple'
    normalize = T.Normalize(mean=cfg.INPUT.MEAN, std=cfg.INPUT.STD)

    transforms = None
    if is_train:
        if cfg.FINETUNE.FINETUNE and not cfg.FINETUNE.USE_TRAIN_AUG:
            # precrop, crop = get_resolution(cfg.INPUT.IMAGE_SIZE)
            crop = cfg.INPUT.IMAGE_SIZE
            precrop = int(crop / cfg.INPUT.CROP_PCT)
            transforms = T.Compose([
                T.Resize(precrop,
                    interpolation=cfg.INPUT.INTERPOLATION
                ),
                T.RandomCrop((crop, crop)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ])
        else:
            aug = cfg.AUG
            scale = aug.SCALE
            ratio = aug.RATIO
            ts = [
                T.RandomResizedCrop(
                    cfg.INPUT.IMAGE_SIZE, scale=scale, ratio=ratio, interpolation=cfg.INPUT.INTERPOLATION
                ),
                T.RandomHorizontalFlip(),
            ]

            cj = aug.COLOR_JITTER
            if cj[-1] > 0.0:
                ts.append(T.RandomApply([T.ColorJitter(*cj[:-1])], p=cj[-1]))

            gs = aug.GRAY_SCALE
            if gs > 0.0:
                ts.append(T.RandomGrayscale(gs))

            gb = aug.GAUSSIAN_BLUR
            if gb > 0.0:
                ts.append(T.RandomApply([GaussianBlur([.1, 2.])], p=gb))

            ts.append(T.ToTensor())
            ts.append(normalize)

            transforms = T.Compose(ts)
    else:
        transforms = T.Compose([
            T.Resize(int(cfg.INPUT.IMAGE_SIZE / cfg.INPUT.CROP_PCT), interpolation=cfg.INPUT.INTERPOLATION),
            T.CenterCrop(cfg.INPUT.IMAGE_SIZE),
            T.ToTensor(),
            normalize,
        ])

    return transforms
