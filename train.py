# TODO - placeholder for now

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data import get_dataset
from lightning_simsiam import LitSimSiam
from transforms import get_transform


data_dir = "/home/paperspace/Documents/meta-video-similarity/data/train/reference"
transform = get_transform("train")
ds = get_dataset(data_dir, transform=transform)
loader = DataLoader(ds, batch_size=4, num_workers=4, pin_memory=True, shuffle=True)
model = LitSimSiam(
    model_name_or_path="MCG-NJU/videomae-base",
    dim=512,
    pred_dim=128,
    lr=0.05,
    momentum=0.9,
    weight_decay=1e-4,
    batch_size=4,
    warmup_steps=0,
)
trainer = pl.Trainer(devices=1, accelerator="gpu")
trainer.fit(model, loader)
