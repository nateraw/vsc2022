import itertools
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler

from .transforms import get_transform


class LimitDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(itertools.repeat(iter(dataset), 2))
        self.transform = transform

    def __getitem__(self, index):
        ex = next(self.dataset_iter)
        x = ex["video"]
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return self.dataset.num_videos


def get_dataset(
    video_dir: str,
    mode: str = "train",
    num_clip_frames=16,
    frame_sample_rate=4,
    fps=30,
    decoder="pyav",
    val_clips_per_video=3,
) -> Dataset:
    """Get dataset for training or testing.

    Args:
        video_dir (str): Path to the directory containing videos.
        mode (str, optional): Mode of the dataset. Defaults to 'train'.

    Returns:
        Dataset: Dataset for training or testing.
    """
    samples_per_clip = num_clip_frames * frame_sample_rate
    clip_duration = samples_per_clip / fps

    video_dir = Path(video_dir)
    video_file_paths = sorted(video_dir.glob("*.mp4"))

    video_sampler = RandomSampler if mode == "train" else SequentialSampler
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        logging.info("Distributed Environmnet detected, using DistributedSampler for dataset.")
        video_sampler = DistributedSampler

    ds = LimitDataset(
        LabeledVideoDataset(
            LabeledVideoPaths(
                [(str(video_path), {"label": -1}) for video_path in video_file_paths],
            ),
            clip_sampler=make_clip_sampler("random", clip_duration)
            if mode == "train"
            else make_clip_sampler("constant_clips_per_video", clip_duration, val_clips_per_video),
            video_sampler=video_sampler,
            transform=None,  # We apply these in the limit dataset
            decode_audio=False,
            decoder=decoder,
        ),
        transform=get_transform(mode),
    )
    return ds
