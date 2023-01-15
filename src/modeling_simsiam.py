# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from transformers import VideoMAEForVideoClassification


class SimSiam(nn.Module):
    def __init__(self, model_name_or_path="MCG-NJU/videomae-base", dim=2048, pred_dim=512):
        """Build a SimSiam model with a VideoMAE encoder.

        Args:
            model_name_or_path (str, optional, *defaults to "MCG-NJU/videomae-base"*):
                Local model or model on Hugging Face Hub.
            dim (int, optional): Output dimension of both encoder classifier and predictor.
            pred_dim (int, optional): Latent dim of predictor layer.
        """
        super(SimSiam, self).__init__()

        # create the encoder
        self.encoder = VideoMAEForVideoClassification.from_pretrained(model_name_or_path, num_labels=dim)

        # build a 3-layer projector
        prev_dim = self.encoder.classifier.weight.shape[1]  # 768
        self.encoder.classifier = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            self.encoder.classifier,
            nn.BatchNorm1d(dim, affine=False),
        )  # output layer
        self.encoder.classifier[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim),
        )  # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1)[0]  # NxC
        z2 = self.encoder(x2)[0]  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()
