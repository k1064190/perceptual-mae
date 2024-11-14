"""
This script contains the code for 2 Models:
Masked Vision Model, based on the following: https://arxiv.org/abs/2111.06377

And the base ViT model used for downstream classification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from pathlib import Path
import random
import timm
import timm.optim.optim_factory as optim_factory
from functools import partial
import os
from collections import OrderedDict
from omegaconf.errors import ConfigAttributeError
import transformers

from src.models.modules.pos_embeds import *
from src.models.base_model import BaseModel
from src.models.modules.image_encoder import *
from src.models.modules.operation_vision_layers import (
    OpAEEncoder,
    OpAEDecoder,
    OpT5AEDecoder,
)
from src.models.modules.discriminators import Discriminator, MSGDiscriminator
from src.models.modules.stylegan_layers import *
from src.common.registry import registry
from src.models.modules.layer_utils import *
from src.losses.image_reconstruction import MaskedImageLoss, OpImageLoss, scale_pyramid
from src.datasets.transforms.vision_transforms_utils import Normalise, UnNormalise
from src.common.constants import (
    TRACTABLE_CED_MEAN,
    TRACTABLE_CED_STD,
    IMAGE_COLOR_MEAN,
    IMAGE_COLOR_STD,
)
from torchmetrics import Accuracy, Precision, F1Score, Recall, AUROC


@registry.register_model("operation_image_autoencoder")
class OperationImageAutoEncoder(BaseModel):
    def __init__(self, config, local_experiment_data_dir):
        super().__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config = self.config.dataset_config
        self.user_config = self.config.user_config
        self.image_out_dir = os.path.join(local_experiment_data_dir, "opae_recon_out")
        if not os.path.exists(self.image_out_dir):
            os.makedirs(self.image_out_dir, exist_ok=True)
        # patch embed args;
        self.finetune_imagenet = self.model_config.finetune_imagenet
        self.num_samples_to_visualise = self.model_config.num_samples_to_visualise
        self.frequency_to_visualise = self.model_config.frequency_to_visualise
        self.operations = self.model_config.operations.keys()
        self.batch_size = self.config.training.batch_size
        self.gpu_device = self.config.trainer.params.devices
        if self.gpu_device == -1:
            self.device_count = torch.cuda.device_count()
        else:
            self.device_count = len(self.gpu_device)

        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        self.patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        embed_dim = self.model_config.image_encoder.embed_dim
        self.norm_layer_arg = self.model_config.norm_layer_arg

        if self.norm_layer_arg == "partial":
            self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
            print("using partial layer norm")
        else:
            self.norm_layer = nn.LayerNorm

        self.patch_embed = PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)

        # Encoder and Decoder setup (similar to MaskedImageAutoEncoder)
        self.encoder = OpAEEncoder(config, self.patch_embed, nn.LayerNorm)
        self.decoder = OpAEDecoder(config, self.patch_embed, nn.LayerNorm)

        # Loss function
        self.loss_fn = OpImageLoss(config, self.patch_embed)
        # if using the GAN loss; initiate the discriminator;
        if self.model_config.loss_type == "gan":
            raise Exception(
                "To use the GAN loss, use the following model: {} - TODO: not implemented. This model implementation does not support GAN loss".format(
                    "op_image_autoencoder_gan_loss"
                )
            )

        if self.model_config.normalisation_params == "imagenet":
            self.unnormalise = UnNormalise(IMAGE_COLOR_MEAN, IMAGE_COLOR_STD)
        else:
            self.unnormalise = UnNormalise(TRACTABLE_CED_MEAN, TRACTABLE_CED_STD)

        if self.finetune_imagenet != None:
            self.load_imagenet_weights()
            print(
                "og imagenet weights loaded from: {} \n to commence finetuning".format(
                    self.finetune_imagenet
                )
            )

    def load_imagenet_weights(self):
        # load the model dictionary
        # NOTE: ONLY encoder weights should be loaded; decoder has to be trained from scratch for the specific data;
        # otherwise no point in doing MAE since imagenet distribution can also reconstruct different image types.
        # pretrained_dict= torch.load(self.finetune_imagenet)
        pretrained_dict = torch.load(self.finetune_imagenet)
        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
        if "model" in pretrained_dict:
            pretrained_dict = pretrained_dict["model"]

        model_dict = self.encoder.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.encoder.load_state_dict(model_dict)
        # if patch embed value != pretrained dict; throw an error; to ensure patchembed is correct.

    def forward(self, x):
        latent, augmented_x, op_info = self.encoder(x)
        pred = self.decoder(latent, op_info)  # [N, L, p*p*3]

        return pred, augmented_x, op_info  # [N, op_info=("num_op", "info")]

    def training_step(self, batch, batch_idx):
        x = batch["pixel_values"]  # [bsz, channel, height, width]
        reconstructed, augmented_x, op_infos = self(x)

        loss = self.loss_fn(x, reconstructed)
        self.log(
            "train_loss",
            loss,
            rank_zero_only=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            rank_zero_only=True,
            prog_bar=True,
            logger=True,
        )

        # log images
        if dist.get_rank() == 0:
            if self.global_step % self.frequency_to_visualise == 0:
                original_imgs = x.clone().detach()
                preds = reconstructed.clone().detach()

                # Randomly select samples to visualize
                num_samples = min(self.num_samples_to_visualise, x.size(0))
                rand_indices = torch.randperm(x.size(0))[:num_samples]

                Images, Recons, Augs, Op_infos = [], [], [], []

                for idx in rand_indices:
                    orig_img = self.unnormalise(original_imgs[idx])
                    recon_img = self.visualise_sample(preds[idx].unsqueeze(0)).squeeze(
                        0
                    )
                    aug_imgs = self.unnormalise(augmented_x[idx])

                    # Assuming images are in range [0, 1]. If not, you might need to normalize them.
                    Images.append(orig_img)
                    Recons.append(recon_img)
                    Augs.append(aug_imgs)
                    Op_infos.append(op_infos[idx])

                Image_Recon = []
                for i in range(len(Images)):
                    image_recon = torch.cat(
                        (Images[i], Augs[i], Recons[i]), dim=1
                    )  # [3, 2 * H, W]
                    image_recon = (
                        image_recon.permute(1, 2, 0).cpu().numpy()
                    )  # [H, W, C]
                    Image_Recon.append(image_recon)

                # If you're using a logger (e.g., TensorBoard), you can log the images like this:
                self.logger.log_image(
                    "reconstructions",
                    images=Image_Recon,
                    caption=[
                        f"step_{self.global_step}, operation_{op_info}"
                        for op_info in Op_infos
                    ],
                )

        return loss

    def _eval_step(self, batch, batch_idx) -> float:
        x = batch["pixel_values"]  # [bsz, channel, height, width]
        reconstructed, _, _ = self(x)
        loss = self.loss_fn(x, reconstructed)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._eval_step(batch, batch_idx)

        # clone logits for metrics (don't want gradients to pass)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self._eval_step(batch, batch_idx)

        # clone logits for metrics (don't want gradients to pass)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"test_loss": loss}

    def exclude_from_wt_decay(self, skip_list=["bias", "LayerNorm.weight"]):
        params = []
        excluded_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": 0.05},
            {"params": excluded_params, "weight_decay": 0},
        ]

    def configure_optimizers(self):
        """
        Configure and load optimizers here.
        """
        weight_decay = 0.05
        lr = 3e-8
        blr = (
            lr
            * self.device_count
            * self.batch_size
            * self.trainer.accumulate_grad_batches
        )
        min_lr = 0.0
        warmup_steps = 1000
        betas = (0.9, 0.95)

        # following timm: set wd as 0 for bias and norm layers
        # param_groups = optim_factory.add_weight_decay(self, weight_decay) # weight_decay;
        param_groups = self.exclude_from_wt_decay(["bias", "LayerNorm.weight"])
        optimizer = torch.optim.AdamW(
            param_groups, lr=blr, betas=betas, weight_decay=weight_decay
        )
        total_steps = self.trainer.estimated_stepping_batches
        # warmup_epochs = 20
        # wramup_steps = warmup_epochs * (total_steps // self.trainer.max_epochs)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps=total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # or 'epoch'
                "frequency": 1,
            },
            "monitor": "val_loss",
        }

    def visualise_sample(self, pred):
        y = self.unpatchify(pred)
        y = self.unnormalise(y)

        return y[0]

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs


@registry.register_model("operation_t5_image_autoencoder")
class OperationT5ImageAutoEncoder(BaseModel):
    def __init__(self, config, local_experiment_data_dir):
        super().__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config = self.config.dataset_config
        self.user_config = self.config.user_config
        self.image_out_dir = os.path.join(local_experiment_data_dir, "opae_recon_out")
        if not os.path.exists(self.image_out_dir):
            os.makedirs(self.image_out_dir, exist_ok=True)
        # patch embed args;
        self.finetune_imagenet = self.model_config.finetune_imagenet
        self.num_samples_to_visualise = self.model_config.num_samples_to_visualise
        self.frequency_to_visualise = self.model_config.frequency_to_visualise
        self.operations = self.model_config.operations.keys()
        self.batch_size = self.config.training.batch_size
        self.gpu_device = self.config.trainer.params.devices
        if self.gpu_device == -1:
            self.device_count = torch.cuda.device_count()
        else:
            self.device_count = len(self.gpu_device)

        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        self.patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        embed_dim = self.model_config.image_encoder.embed_dim
        self.norm_layer_arg = self.model_config.norm_layer_arg

        if self.norm_layer_arg == "partial":
            self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
            print("using partial layer norm")
        else:
            self.norm_layer = nn.LayerNorm

        self.patch_embed = PatchEmbed(img_size, self.patch_size, in_channels, embed_dim)

        if self.model_config.normalisation_params == "imagenet":
            self.unnormalise = UnNormalise(IMAGE_COLOR_MEAN, IMAGE_COLOR_STD)
        else:
            self.unnormalise = UnNormalise(TRACTABLE_CED_MEAN, TRACTABLE_CED_STD)

        # Encoder and Decoder setup (similar to MaskedImageAutoEncoder)
        self.encoder = OpAEEncoder(config, self.patch_embed, nn.LayerNorm)
        self.decoder = OpT5AEDecoder(config, self.patch_embed, nn.LayerNorm)

        # Loss function
        self.loss_fn = OpImageLoss(config, self.patch_embed)
        # if using the GAN loss; initiate the discriminator;
        if self.model_config.loss_type == "gan":
            raise Exception(
                "To use the GAN loss, use the following model: {} - TODO: not implemented. This model implementation does not support GAN loss".format(
                    "op_image_autoencoder_gan_loss"
                )
            )

        if self.finetune_imagenet != None:
            self.load_imagenet_weights()
            print(
                "og imagenet weights loaded from: {} \n to commence finetuning".format(
                    self.finetune_imagenet
                )
            )

    def load_imagenet_weights(self):
        # load the model dictionary
        # NOTE: ONLY encoder weights should be loaded; decoder has to be trained from scratch for the specific data;
        # otherwise no point in doing MAE since imagenet distribution can also reconstruct different image types.
        # pretrained_dict= torch.load(self.finetune_imagenet)
        pretrained_dict = torch.load(self.finetune_imagenet)
        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
        if "model" in pretrained_dict:
            pretrained_dict = pretrained_dict["model"]

        model_dict = self.encoder.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.encoder.load_state_dict(model_dict)
        # if patch embed value != pretrained dict; throw an error; to ensure patchembed is correct.

    def forward(self, x):
        latent, augmented_x, op_info = self.encoder(x)
        pred = self.decoder(latent, op_info)  # [N, L, p*p*3]

        return pred, augmented_x, op_info  # [N, op_info=("num_op", "info")]

    def training_step(self, batch, batch_idx):
        x = batch["pixel_values"]  # [bsz, channel, height, width]
        reconstructed, augmented_x, op_infos = self(x)

        loss = self.loss_fn(x, reconstructed)
        self.log(
            "train_loss",
            loss,
            rank_zero_only=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            rank_zero_only=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "global_step",
            self.global_step,
            rank_zero_only=True,
            prog_bar=True,
            logger=True,
        )

        if dist.get_rank() == 0:
            if self.global_step % self.frequency_to_visualise == 0:
                original_imgs = x.clone().detach()
                preds = reconstructed.clone().detach()

                # Randomly select samples to visualize
                num_samples = min(self.num_samples_to_visualise, x.size(0))
                rand_indices = torch.randperm(x.size(0))[:num_samples]

                Images, Recons, Augs, Op_infos = [], [], [], []

                for idx in rand_indices:
                    orig_img = self.unnormalise(original_imgs[idx])
                    recon_img = self.visualise_sample(preds[idx].unsqueeze(0)).squeeze(
                        0
                    )
                    aug_imgs = self.unnormalise(augmented_x[idx])

                    # Assuming images are in range [0, 1]. If not, you might need to normalize them.
                    Images.append(orig_img)
                    Recons.append(recon_img)
                    Augs.append(aug_imgs)
                    Op_infos.append(op_infos[idx])

                Image_Recon = []
                for i in range(len(Images)):
                    image_recon = torch.cat(
                        (Images[i], Augs[i], Recons[i]), dim=1
                    )  # [3, 2 * H, W]
                    image_recon = (
                        image_recon.permute(1, 2, 0).cpu().numpy()
                    )  # [H, W, C]
                    Image_Recon.append(image_recon)

                # If you're using a logger (e.g., TensorBoard), you can log the images like this:
                self.logger.log_image(
                    "reconstructions",
                    images=Image_Recon,
                    caption=[
                        f"step_{self.global_step}, operation_{op_info}"
                        for op_info in Op_infos
                    ],
                )

        return loss

    def _eval_step(self, batch, batch_idx) -> float:
        x = batch["pixel_values"]  # [bsz, channel, height, width]
        reconstructed, _, _ = self(x)
        loss = self.loss_fn(x, reconstructed)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._eval_step(batch, batch_idx)

        # clone logits for metrics (don't want gradients to pass)
        self.log(
            "val_loss",
            loss,
            rank_zero_only=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self._eval_step(batch, batch_idx)

        # clone logits for metrics (don't want gradients to pass)
        self.log(
            "test_loss",
            loss,
            rank_zero_only=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return {"test_loss": loss}

    def exclude_from_wt_decay(self, skip_list=["bias", "LayerNorm.weight"]):
        params = []
        excluded_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": 0.05},
            {"params": excluded_params, "weight_decay": 0},
        ]

    def configure_optimizers(self):
        """
        Configure and load optimizers here.
        """
        weight_decay = 0.05
        lr = 5e-8
        # lr = device * batch * accumulations * lr
        blr = (
            lr
            * self.device_count
            * self.batch_size
            * self.trainer.accumulate_grad_batches
        )
        min_lr = 0.0
        warmup_steps = 1000
        betas = (0.9, 0.95)

        # following timm: set wd as 0 for bias and norm layers
        # param_groups = optim_factory.add_weight_decay(self, weight_decay) # weight_decay;
        param_groups = self.exclude_from_wt_decay(["bias", "LayerNorm.weight"])
        optimizer = torch.optim.AdamW(
            param_groups, lr=blr, betas=betas, weight_decay=weight_decay
        )
        total_steps = self.trainer.estimated_stepping_batches
        # warmup_epochs = 20
        # wramup_steps = warmup_epochs * (total_steps // self.trainer.max_epochs)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps=total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # or 'epoch'
                "frequency": 1,
            },
            "monitor": "val_loss",
        }

    def visualise_sample(self, pred):
        y = self.unpatchify(pred)
        y = self.unnormalise(y)

        return y[0]

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
