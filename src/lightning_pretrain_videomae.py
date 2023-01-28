import pytorch_lightning as pl
import torch
from transformers import get_cosine_schedule_with_warmup, VideoMAEForPreTraining


class LitVideoMAEForPretraining(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path="MCG-NJU/videomae-base",
        lr=0.05,
        momentum=0.9,
        weight_decay=1e-4,
        batch_size=512,
        warmup_steps=0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = VideoMAEForPreTraining.from_pretrained(self.hparams.model_name_or_path)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        # infer learning rate before changing batch size
        # TODO - check batch size is right here. Its supposed to be total batch size across all devices
        # so if you set it to 8 and have 2 devices, it should be 16 here. Not sure what gets surfaced up,
        # especially if you use different distribution strategies...
        init_lr = self.hparams.lr * self.hparams.batch_size / 256
        optimizer = torch.optim.SGD(
            self.parameters(), init_lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, self.hparams.warmup_steps, self.trainer.estimated_stepping_batches
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
