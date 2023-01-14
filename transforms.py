import random
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from PIL import ImageDraw
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Div255,
    Normalize,
    Permute,
    RandAugment,
    RandomResizedCrop,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import CenterCrop, Compose, RandomHorizontalFlip


# This file contains some stuff from pytorchvideo's trainer utilities.
# TODO - use some augly transforms to better match dataset distribution


_default_color_pallete = [
    (62, 251, 0),
    (251, 0, 62),
]


def rectangle_overlay(image, color_choices=None, num_rectangle_choices=None):
    # TODO - Jitter right/left randomly
    # TODO - change width, height randomly
    # TODO - more colors

    color_choices = color_choices or _default_color_pallete
    num_rectangle_choices = num_rectangle_choices or [4, 5]

    image = image.convert("RGB")
    w, h = image.size
    # Raw a rectangle centered in the middle of the image by using center_x and center_y
    draw = ImageDraw.Draw(image, "RGBA")

    center_x = w / 2
    center_y = h / 2

    # lets say rectangles are 20% of width
    rect_width = w * 0.3
    rect_height = h * 0.07

    num_boxes = random.choice(num_rectangle_choices)

    # From pretty transparent to solid. idk
    alpha = random.choice(list(range(105, 256, 50)))
    fill_color = random.choice(color_choices)
    fill_color = (*fill_color, alpha)
    print(fill_color)
    for y_pad in np.linspace(-0.3, 0.3, num_boxes):
        # Coords - [(x0, y0), (x1, y1)]
        shape = [
            (center_x - rect_width / 2, center_y - rect_height / 2 + h * y_pad),
            (center_x + rect_width / 2, center_y + rect_height / 2 + h * y_pad),
        ]
        draw.rectangle(shape, fill=fill_color)
    return image.convert("RGB")


class RandomRectangleOverlay:
    def __init__(self, p=0.5, color_choices=None, num_rectangle_choices=None):
        """Randomly apply vertical rectangle overlay to the image.

        See `rectangle_overlay` for more details.

        Args:
            p (float, optional, *defaults to 0.5*): Probability to apply this transform.
            color_choices (List[Tuple[int]], optional): List of RGB tuples.
            num_rectangle_choices (List[int], optional): Candidates for num of rectangles to overlay.
        """
        self.p = p
        self.color_choices = color_choices
        self.num_rectangle_choices = num_rectangle_choices

    def __call__(self, image):
        if random.random() < self.p:
            return rectangle_overlay(image, self.color_choices, self.num_rectangle_choices)
        return image


class RepeatandConverttoList:
    """
    An utility transform that repeats each value in a
    key, value-style minibatch and replaces it with a list of values.
    Useful for performing multiple augmentations.
    An example such usecase can be found in
    `pytorchvideo_trainer/conf/datamodule/transforms/kinetics_classification_mvit_16x4.yaml`
    Args:
        repead_num (int): Number of times to repeat each value.
    """

    def __init__(self, repeat_num: int) -> None:
        super().__init__()
        self.repeat_num = repeat_num

    # pyre-ignore[3]
    def __call__(self, sample_dict: Dict[str, Any]) -> Dict[str, List[Any]]:
        for k, v in sample_dict.items():
            sample_dict[k] = self.repeat_num * [v]
        return sample_dict


class ApplyTransformToKeyOnList:
    """
    Applies transform to key of dictionary input wherein input is a list
    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied
    Example:
         >>>  transforms.ApplyTransformToKeyOnList(
        >>>       key='video',
        >>>       transform=UniformTemporalSubsample(num_video_samples),
        >>>   )
    """

    def __init__(self, key: str, transform: Callable) -> None:  # pyre-ignore[24]
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        x[self._key] = [self._transform(a) for a in x[self._key]]
        return x


def get_transform(mode="val", num_clip_frames=16, crop_height=224, crop_width=224, short_side_size=256):

    # TODO - too much permuting happening here? Does this slow down training?
    default_train_transform = Compose(
        [
            # RepeatandConverttoList(repeat_num=2),
            ApplyTransformToKeyOnList(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_clip_frames),
                        Div255(),
                        Permute(dims=[1, 0, 2, 3]),
                        RandAugment(magnitude=7, num_layers=4),
                        Permute(dims=[1, 0, 2, 3]),
                        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                        RandomResizedCrop(
                            target_height=crop_height,
                            target_width=crop_width,
                            scale=[0.08, 1.0],
                            aspect_ratio=[0.75, 1.3333],
                        ),
                        RandomHorizontalFlip(p=0.5),
                        Permute(dims=[1, 0, 2, 3]),
                    ]
                ),
            ),
            RemoveKey(key="audio"),
        ]
    )
    default_val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_clip_frames),
                        Div255(),
                        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                        ShortSideScale(size=short_side_size),
                        CenterCrop(size=crop_height),
                        Permute(dims=[1, 0, 2, 3]),
                    ]
                ),
            ),
            RemoveKey(key="audio"),
        ]
    )
    return default_train_transform if mode == "train" else default_val_transform
