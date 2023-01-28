import json

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data_pretrain import get_dataset
from src.lightning_pretrain_videomae import LitVideoMAEForPretraining
from src.transforms import get_transform

import torch

torch.set_float32_matmul_precision("medium")

def main(
    data_dir: str = "./data/train/reference",
    model_name_or_path: str = "MCG-NJU/videomae-base",
    lr: float = 1.5e-4,
    momentum: float = 0.9,
    weight_decay: float = 0.05,
    batch_size: int = 32,
    warmup_steps: int = 200,
    num_workers: int = 30,
    num_clip_frames=16,
    frame_sample_rate=4,
    use_wandb=False,
    wandb_project="vsc2022-videomae-pretrain",
    wandb_run_name=None,
    max_epochs=100,
    log_every_n_steps=5,
    mask_ratio=0.75,
    accumulate_grad_batches=1,
    precision='bf16', # 'bf16'
    devices=1,
):
    # print all the provided args
    cfg = locals()
    print('-' * 80)
    print(json.dumps(cfg, indent=2, sort_keys=False))
    print('-' * 80)

    ds = get_dataset(data_dir, num_clip_frames=num_clip_frames, frame_sample_rate=frame_sample_rate)
    model = LitVideoMAEForPretraining(**cfg)

    num_patches_per_frame = (model.model.config.image_size // model.model.config.patch_size) ** 2
    seq_length = (num_clip_frames // model.model.config.tubelet_size) * num_patches_per_frame
    num_masks = int(mask_ratio * seq_length)

    def collate_fn(batch):
        # tublet size = 2
        # patch_size = 16
        # image_size = 224

        # make bool_masked_pos from batch.
        pixel_values = torch.stack(batch)

        batch_size = pixel_values.size(0)
        mask = torch.ones((num_masks,))
        mask = torch.cat([mask, torch.zeros(seq_length - mask.size(0))])
        mask = mask[torch.randperm(mask.size(0))]
        bool_masked_pos = mask.expand(batch_size, -1).bool()

        return {'pixel_values': pixel_values, 'bool_masked_pos': bool_masked_pos}

    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True, collate_fn=collate_fn,)

    trainer = pl.Trainer(
        devices=devices,
        accelerator="gpu",
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        max_epochs=max_epochs,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint(dirpath='pretrain_model_outputs', save_top_k=4, monitor='loss', mode='min', save_last=True),
        ],
        logger=not use_wandb or pl.loggers.WandbLogger(project=wandb_project, log_model=False, save_code=False, name=wandb_run_name),
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer.fit(model, loader)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
