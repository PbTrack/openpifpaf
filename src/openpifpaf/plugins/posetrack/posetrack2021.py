import argparse

import PIL
import torch

import openpifpaf

from . import datasets
from .normalize_transform import NormalizePosetrack

from .constants import (
    KEYPOINTS,
    SIGMAS,
    UPRIGHT_POSE,
    SKELETON,
    DENSER_CONNECTIONS,
)


class Posetrack2021(openpifpaf.datasets.DataModule):
    """Posetrack 2021 dataset as a multi person pose estimation dataset."""

    debug = False
    pin_memory = False

    # cli configurable
    train_annotations = "data-posetrack2021/annotations/train/*.json"
    val_annotations = "data-posetrack2021/annotations/val/*.json"
    eval_annotations = val_annotations
    data_root = "data-posetrack2021"

    square_edge = 385
    with_dense = False
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    bmin = 0.1
    image_aug = 0.0

    def __init__(self):
        super().__init__()

        cif = openpifpaf.headmeta.Cif(
            "cif",
            "posetrack2021",
            keypoints=KEYPOINTS,
            sigmas=SIGMAS,
            pose=UPRIGHT_POSE,
            draw_skeleton=SKELETON,
        )
        caf = openpifpaf.headmeta.Caf(
            "caf",
            "posetrack2021",
            keypoints=KEYPOINTS,
            sigmas=SIGMAS,
            pose=UPRIGHT_POSE,
            skeleton=SKELETON,
        )
        dcaf = openpifpaf.headmeta.Caf(
            "dcaf",
            "posetrack2021",
            keypoints=KEYPOINTS,
            sigmas=SIGMAS,
            pose=UPRIGHT_POSE,
            skeleton=DENSER_CONNECTIONS,
            sparse_skeleton=SKELETON,
            only_in_field_of_view=True,
        )

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        dcaf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf, dcaf] if self.with_dense else [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group2021 = parser.add_argument_group("data module Posetrack2021")
        group2021.add_argument(
            "--posetrack2021-train-annotations",
            default=cls.train_annotations,
            help="train annotations",
        )
        group2021.add_argument(
            "--posetrack2021-val-annotations",
            default=cls.val_annotations,
            help="val annotations",
        )
        group2021.add_argument(
            "--posetrack2021-eval-annotations",
            default=cls.eval_annotations,
            help="eval annotations",
        )
        group2021.add_argument(
            "--posetrack2021-data-root", default=cls.data_root, help="data root"
        )

        group = parser.add_argument_group("data module Posetrack2021")
        group.add_argument(
            "--posetrack2021-square-edge",
            default=cls.square_edge,
            type=int,
            help="square edge of input images",
        )
        assert not cls.with_dense
        group.add_argument(
            "--posetrack2021-with-dense",
            default=False,
            action="store_true",
            help="train with dense connections",
        )
        assert not cls.extended_scale
        group.add_argument(
            "--posetrack2021-extended-scale",
            default=False,
            action="store_true",
            help="augment with an extended scale range",
        )
        group.add_argument(
            "--posetrack2021-orientation-invariant",
            default=cls.orientation_invariant,
            type=float,
            help="augment with random orientations",
        )
        group.add_argument(
            "--posetrack2021-blur",
            default=cls.blur,
            type=float,
            help="augment with blur",
        )
        assert cls.augmentation
        group.add_argument(
            "--posetrack2021-no-augmentation",
            dest="posetrack2021_augmentation",
            default=True,
            action="store_false",
            help="do not apply data augmentation",
        )
        group.add_argument(
            "--posetrack2021-rescale-images",
            default=cls.rescale_images,
            type=float,
            help="overall rescale factor for images",
        )
        group.add_argument(
            "--posetrack2021-upsample",
            default=cls.upsample_stride,
            type=int,
            help="head upsample stride",
        )
        group.add_argument(
            "--posetrack2021-min-kp-anns",
            default=cls.min_kp_anns,
            type=int,
            help="filter images with fewer keypoint annotations",
        )
        group.add_argument(
            "--posetrack2021-bmin", default=cls.bmin, type=float, help="bmin"
        )
        group.add_argument(
            "--posetrack2021-image-augmentations",
            default=cls.image_aug,
            type=float,
            help="autocontrast, equalize, invert, solarize",
        )

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # posetrack2021 specific
        cls.train_annotations = args.posetrack2021_train_annotations
        cls.val_annotations = args.posetrack2021_val_annotations
        cls.eval_annotations = args.posetrack2021_eval_annotations
        cls.data_root = args.posetrack2021_data_root

        cls.square_edge = args.posetrack2021_square_edge
        cls.with_dense = args.posetrack2021_with_dense
        cls.extended_scale = args.posetrack2021_extended_scale
        cls.orientation_invariant = args.posetrack2021_orientation_invariant
        cls.blur = args.posetrack2021_blur
        cls.augmentation = args.posetrack2021_augmentation
        cls.rescale_images = args.posetrack2021_rescale_images
        cls.upsample_stride = args.posetrack2021_upsample
        cls.min_kp_anns = args.posetrack2021_min_kp_anns
        cls.bmin = args.posetrack2021_bmin
        cls.image_aug = args.posetrack2021_image_augmentations

    def _preprocess(self):
        encoders = [
            openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.bmin),
            openpifpaf.encoder.Caf(self.head_metas[1], bmin=self.bmin),
        ]
        if len(self.head_metas) > 2:
            encoders.append(openpifpaf.encoder.Caf(self.head_metas[2], bmin=self.bmin))

        if not self.augmentation:
            return openpifpaf.transforms.Compose(
                [
                    openpifpaf.transforms.NormalizeAnnotations(),
                    openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                    openpifpaf.transforms.CenterPad(self.square_edge),
                    openpifpaf.transforms.EVAL_TRANSFORM,
                    openpifpaf.transforms.Encoders(encoders),
                ]
            )

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.25 * self.rescale_images, 2.0 * self.rescale_images),
                power_law=True,
                stretch_range=(0.75, 1.33),
            )
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.4 * self.rescale_images, 2.0 * self.rescale_images),
                power_law=True,
                stretch_range=(0.75, 1.33),
            )

        return openpifpaf.transforms.Compose(
            [
                NormalizePosetrack(),
                openpifpaf.transforms.RandomApply(
                    openpifpaf.transforms.HFlip(
                        KEYPOINTS, openpifpaf.plugins.coco.constants.HFLIP
                    ),
                    0.5,
                ),
                rescale_t,
                openpifpaf.transforms.RandomApply(
                    openpifpaf.transforms.Blur(), self.blur
                ),
                openpifpaf.transforms.RandomChoice(
                    [
                        openpifpaf.transforms.RotateBy90(),
                        openpifpaf.transforms.RotateUniform(30.0),
                    ],
                    [self.orientation_invariant, 0.4],
                ),
                openpifpaf.transforms.Crop(
                    self.square_edge, use_area_of_interest=True
                ),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.RandomChoice(
                    [
                        openpifpaf.transforms.ImageTransform(PIL.ImageOps.autocontrast),
                        openpifpaf.transforms.ImageTransform(PIL.ImageOps.equalize),
                        openpifpaf.transforms.ImageTransform(PIL.ImageOps.invert),
                        openpifpaf.transforms.ImageTransform(PIL.ImageOps.solarize),
                    ],
                    [
                        self.image_aug / 4,
                        self.image_aug / 4,
                        self.image_aug / 4,
                        self.image_aug / 4,
                    ],
                ),
                openpifpaf.transforms.TRAIN_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ]
        )

    def train_loader(self):
        train_data = datasets.Posetrack2021(
            annotation_files=self.train_annotations,
            data_root=self.data_root,
            group=None,  # [(0, -12), (0, -8), (0, -4)],
            preprocess=self._preprocess(),
            only_annotated=True,
        )
        return torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory,
            num_workers=self.loader_workers,
            drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta,
        )

    def val_loader(self):
        val_data = datasets.Posetrack2021(
            annotation_files=self.val_annotations,
            data_root=self.data_root,
            group=None,  # [(0, -12), (0, -8), (0, -4)],
            preprocess=self._preprocess(),
            only_annotated=True,
        )
        return torch.utils.data.DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory,
            num_workers=self.loader_workers,
            drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta,
        )

    def eval_loader(self):
        raise NotImplementedError

    def metrics(self):
        raise NotImplementedError
