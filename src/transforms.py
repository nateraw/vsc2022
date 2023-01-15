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
from torchvision.transforms import CenterCrop, Compose, RandomHorizontalFlip, RandomVerticalFlip
import random
from typing import Any, Callable, Dict, List

import torch
import torchvision
from PIL import Image, ImageFilter
from torchvision.transforms import Compose

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


class RandomRectangleOverlay(torch.nn.Module):
    def __init__(self, p=0.5, color_choices=None, num_rectangle_choices=None):
        """Randomly apply vertical rectangle overlay to the image.

        See `rectangle_overlay` for more details.

        Args:
            p (float, optional, *defaults to 0.5*): Probability to apply this transform.
            color_choices (List[Tuple[int]], optional): List of RGB tuples.
            num_rectangle_choices (List[int], optional): Candidates for num of rectangles to overlay.
        """
        super().__init__()
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

class ColorJitterVideoSSl:
    """
    A custom sequence of transforms that randomly performs Color jitter,
    Gaussian Blur and Grayscaling on the given clip.
    Particularly useful for the SSL tasks like SimCLR, MoCoV2, BYOL, etc.
    Args:
        bri_con_sat (list[float]): A list of 3 floats reprsenting brightness,
        constrast and staturation coefficients to use for the
        `torchvision.transforms.ColorJitter` transform.
        hue (float): Heu value to use in the `torchvision.transforms.ColorJitter`
        transform.
        p_color_jitter (float): The probability with which the Color jitter transform
        is randomly applied on the given clip.
        p_convert_gray (float): The probability with which the given clip is randomly
        coverted into grayscale.
        p_gaussian_blur (float): The probability with which the Gaussian transform
        is randomly applied on the given clip.
        gaussian_blur_sigma (list[float]): A list of 2 floats with in which
        the blur radius is randomly sampled for Gaussian blur transform.
    """

    def __init__(
        self,
        bri_con_sat: List[float],
        hue: float,
        p_color_jitter: float,
        p_convert_gray: float,
        p_gaussian_blur: float = 0.5,
        gaussian_blur_sigma: List[float] = (0.1, 2.0),
    ) -> None:

        self.color_jitter = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomApply(
                    [
                        torchvision.transforms.ColorJitter(
                            bri_con_sat[0], bri_con_sat[1], bri_con_sat[2], hue
                        )
                    ],
                    p=p_color_jitter,
                ),
                torchvision.transforms.RandomGrayscale(p=p_convert_gray),
                torchvision.transforms.RandomApply(
                    [GaussianBlur(gaussian_blur_sigma)], p=p_gaussian_blur
                ),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
        Returns:
            frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
        """
        c, t, h, w = frames.shape
        frames = frames.view(c, t * h, w)
        frames = self.color_jitter(frames)  # pyre-ignore[6,9]
        frames = frames.view(c, t, h, w)

        return frames


class GaussianBlur(object):
    """
    A PIL image version of Gaussian blur augmentation as
    in SimCLR https://arxiv.org/abs/2002.05709
    Args:
        sigma (list[float]): A list of 2 floats with in which
        the blur radius is randomly sampled during each step.
    """

    def __init__(self, sigma: List[float] = (0.1, 2.0)) -> None:
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        img (Image): A PIL image with single or 3 color channels.
        """
        sigma = self.sigma[0]
        if len(self.sigma) == 2:
            sigma = random.uniform(self.sigma[0], self.sigma[1])

        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


def get_transform(mode="val", num_clip_frames=16, crop_height=224, crop_width=224, short_side_size=256):

    default_train_transform = Compose(
        [
            UniformTemporalSubsample(num_clip_frames),
            Div255(),
            ColorJitterVideoSSl(
                bri_con_sat=[0.6, 0.6, 0.6],
                hue=0.15,
                p_color_jitter=0.8,
                p_convert_gray=0.2,
            ),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomResizedCrop(
                target_height=crop_height,
                target_width=crop_width,
                scale=[0.2, 0.766],
                aspect_ratio=[0.75, 1.3333],
            ),
            RandomHorizontalFlip(p=0.5),
            Permute(dims=[1, 0, 2, 3]),
        ]
    )

    # TODO - too much permuting happening here? Does this slow down training?
    default_train_transform_old = Compose(
        [
            UniformTemporalSubsample(num_clip_frames),
            Div255(),
            Permute(dims=[1, 0, 2, 3]),
            RandAugment(magnitude=7, num_layers=4),
            Permute(dims=[1, 0, 2, 3]),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomResizedCrop(
                target_height=crop_height,
                target_width=crop_width,
                scale=[0.08, 1.0],
                aspect_ratio=[0.75, 1.3333],
            ),
            # TODO - PIL Transforms over temporal dim?
            # RandomRectangleOverlay(),
            RandomHorizontalFlip(p=0.5),
            Permute(dims=[1, 0, 2, 3]),
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
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ShortSideScale(size=short_side_size),
                        CenterCrop(size=crop_height),
                        Permute(dims=[1, 0, 2, 3]),
                    ]
                ),
            ),
            RemoveKey(key="audio"),
        ]
    )

    transform_dict = {
        "train": default_train_transform,
        "train_old": default_train_transform_old,
        "val": default_val_transform,
    }
    return transform_dict[mode]
