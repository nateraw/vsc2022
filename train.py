# TODO - placeholder for now

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data import get_dataset
from src.lightning_simsiam import LitSimSiam
from src.transforms import get_transform


def main(
    data_dir: str = "/home/paperspace/Documents/meta-video-similarity/data/train/reference",
    model_name_or_path: str = "MCG-NJU/videomae-base",
    dim: int = 512,
    pred_dim: int = 128,
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    batch_size: int = 8,
    warmup_steps: int = 0,
    num_workers: int = 8,
):
    ds = get_dataset(data_dir)

    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    model = LitSimSiam(
        model_name_or_path=model_name_or_path,
        dim=dim,
        pred_dim=pred_dim,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
    )
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision=16,
        log_every_n_steps=10,
    )
    trainer.fit(model, loader)


if __name__ == "__main__":
    main()
