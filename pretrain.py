# TODO - placeholder for now

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data_pretrain import get_dataset
from src.lightning_pretrain_videomae import LitVideoMAEForPretraining
from src.transforms import get_transform

import torch
def main(
    data_dir: str = "./data/train/reference",
    model_name_or_path: str = "MCG-NJU/videomae-base",
    lr: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 1.5e-4,
    batch_size: int = 4,
    warmup_steps: int = 0,
    num_workers: int = 4,
    num_clip_frames=16,
    frame_sample_rate=4,
    use_wandb=False,
    wandb_project="vsc2022-videomae-pretrain",
):
    ds = get_dataset(data_dir, num_clip_frames=num_clip_frames, frame_sample_rate=frame_sample_rate)
    model = LitVideoMAEForPretraining(
        model_name_or_path=model_name_or_path,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
    )

    num_patches_per_frame = (model.model.config.image_size // model.model.config.patch_size) ** 2
    seq_length = (num_clip_frames // model.model.config.tubelet_size) * num_patches_per_frame
    mask_ratio = 0.9
    num_masks = int(mask_ratio * seq_length)
    print("num masks: ", num_masks)

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
        devices=1,
        accelerator="gpu",
        precision=16,
        log_every_n_steps=10,
        max_epochs=300,
        logger=not use_wandb or pl.loggers.WandbLogger(project=wandb_project, log_model=False, save_code=False, name=wandb_run_name),
    )
    trainer.fit(model, loader)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
