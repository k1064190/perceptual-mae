from dataclasses import dataclass
from functools import partial
import random
from typing import List, Optional, Tuple, Union

import omegaconf
import timm
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import albumentations as A
from timm.models.vision_transformer import Block, PatchEmbed

from src.models.modules.layer_utils import Flatten
from src.models.modules.pos_embeds import *
from src.models.modules.vision_transformer import OpT5DecoderBlock

from transformers import T5Tokenizer, T5EncoderModel
from collections import OrderedDict

from src.common.constants import (
    TRACTABLE_CED_MEAN,
    TRACTABLE_CED_STD,
    IMAGE_COLOR_MEAN,
    IMAGE_COLOR_STD,
)


"""
layers for: https://arxiv.org/abs/2111.06377
inspiration: https://github.com/facebookresearch/mae 
"""


@dataclass
class OperationOutput:
    """Operation의 출력을 표준화하기 위한 데이터 클래스"""

    transformed: torch.Tensor
    info: str
    mask: Optional[torch.Tensor] = None


class ImageOperations:
    """Image augmentation operations for MAE"""

    def __init__(self):
        self.mean = torch.tensor(IMAGE_COLOR_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGE_COLOR_STD).view(3, 1, 1)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Safely normalize the input tensor"""
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Safely denormalize the input tensor"""
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return x * std + mean

    def _get_random_value(self, param: Union[float, List[float]]) -> float:
        if isinstance(param, (list, omegaconf.ListConfig)):
            return random.uniform(param[0], param[1])
        return param

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array for albumentations"""
        arr = tensor.cpu().detach().numpy()
        return np.transpose(arr, (1, 2, 0))  # CHW -> HWC

    def _numpy_to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert numpy array back to torch tensor"""
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(arr)

    # 일반적인 geometric transforms는 normalisation 필요 없음
    def rotate_90(self, x: torch.Tensor) -> OperationOutput:
        return OperationOutput(F.rotate(x, 90), "rotate_90")

    def rotate_180(self, x: torch.Tensor) -> OperationOutput:
        return OperationOutput(F.rotate(x, 180), "rotate_180")

    def rotate_270(self, x: torch.Tensor) -> OperationOutput:
        return OperationOutput(F.rotate(x, 270), "rotate_270")

    def horizontal_flip(self, x: torch.Tensor) -> OperationOutput:
        return OperationOutput(F.hflip(x), "horizontal_flip")

    def vertical_flip(self, x: torch.Tensor) -> OperationOutput:
        return OperationOutput(F.vflip(x), "vertical_flip")

    def add_gaussian_noise(
        self,
        x: torch.Tensor,
        mean: float = 0.0,
        std: List[float] = [0.25, 0.75],
    ) -> OperationOutput:
        actual_std = self._get_random_value(std)
        noise = torch.randn_like(x) * actual_std + mean
        noisy_x = x + noise
        return OperationOutput(
            noisy_x, f"gaussian_noise, mean={mean:.3f}, std={actual_std:.3f}"
        )

    def gaussian_blur(
        self,
        x: torch.Tensor,
        kernel_size: Union[int, List[int]] = 5,
        sigma: List[float] = [1.0, 2.0],
    ) -> OperationOutput:
        actual_sigma = self._get_random_value(sigma)
        kernel_size = int(self._get_random_value(kernel_size))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        blurred = F.gaussian_blur(x, kernel_size, actual_sigma)
        return OperationOutput(
            blurred,
            f"gaussian_blur, kernel_size={kernel_size}, sigma={actual_sigma:.3f}",
        )

    def motion_blur(
        self,
        x: torch.Tensor,
        kernel_size: int = 5,
    ) -> OperationOutput:
        x_denorm = self._denormalize(x)

        transform = A.MotionBlur(blur_limit=kernel_size, allow_shifted=True, p=1.0)

        img = self._tensor_to_numpy(x_denorm)
        transformed = transform(image=img)["image"]
        transformed_tensor = self._numpy_to_tensor(transformed).to(x.device)

        x_norm = self._normalize(transformed_tensor)

        return OperationOutput(
            x_norm,
            f"motion_blur, kernel_size={kernel_size}",
        )

    def color_jitter(
        self,
        x: torch.Tensor,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.2,
    ) -> OperationOutput:
        x_denorm = self._denormalize(x)

        actual_brightness = self._get_random_value(brightness)
        actual_contrast = self._get_random_value(contrast)
        actual_saturation = self._get_random_value(saturation)
        actual_hue = self._get_random_value(hue)

        x_transformed = F.adjust_brightness(x_denorm, 1 + actual_brightness)
        x_transformed = F.adjust_contrast(x_transformed, 1 + actual_contrast)
        x_transformed = F.adjust_saturation(x_transformed, 1 + actual_saturation)
        x_transformed = F.adjust_hue(x_transformed, actual_hue)
        x_transformed = torch.clamp(x_transformed, 0, 1)

        x_norm = self._normalize(x_transformed)

        return OperationOutput(
            x_norm,
            f"color_jitter, brightness={actual_brightness:.3f}, contrast={actual_contrast:.3f}, "
            f"saturation={actual_saturation:.3f}, hue={actual_hue:.3f}",
        )

    def image_compression(
        self, x: torch.Tensor, quality: List[int] = [20, 70]
    ) -> OperationOutput:
        x_denorm = self._denormalize(x)
        actual_quality = int(self._get_random_value(quality))

        transform = A.ImageCompression(
            quality_lower=actual_quality, quality_upper=actual_quality, p=1.0
        )

        img = self._tensor_to_numpy(x_denorm)
        transformed = transform(image=img)["image"]
        transformed_tensor = self._numpy_to_tensor(transformed).to(x.device)

        x_norm = self._normalize(transformed_tensor)

        return OperationOutput(
            x_norm,
            f"image_compression, quality={actual_quality}",
        )

    def invert_colors(self, x: torch.Tensor) -> OperationOutput:
        return OperationOutput(torch.neg(x), "invert_colors")

    def random_masking(
        self, x: torch.Tensor, mask_ratio: float = 0.75, mask_token: torch.Tensor = None
    ) -> OperationOutput:
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        if mask_token is not None:
            mask_token = mask_token.expand(N, L, -1)
            x_masked = x * (1 - mask.unsqueeze(-1)) + mask_token * mask.unsqueeze(-1)
        else:
            x_masked = x * (1 - mask.unsqueeze(-1))

        return OperationOutput(
            x_masked, f"random_masking, ratio={mask_ratio:.3f}", mask
        )

    def center_crop(
        self, x: torch.Tensor, crop_ratio: float = 0.5, mask_token: torch.Tensor = None
    ) -> OperationOutput:
        N, L, D = x.shape
        H = W = int(L**0.5)

        actual_ratio = self._get_random_value(crop_ratio)
        crop_h = int(H * actual_ratio)
        crop_w = int(W * actual_ratio)

        # Ensure odd dimensions for symmetric cropping
        crop_h = crop_h if crop_h % 2 == 1 else crop_h + 1
        crop_w = crop_w if crop_w % 2 == 1 else crop_w + 1

        pad_h = (H - crop_h) // 2
        pad_w = (W - crop_w) // 2

        mask = torch.ones((N, H, W), device=x.device)
        mask[:, pad_h : pad_h + crop_h, pad_w : pad_w + crop_w] = 0
        mask = mask.reshape(N, -1)

        if mask_token is not None:
            mask_token = mask_token.expand(N, L, -1)
            x_masked = x * (1 - mask.unsqueeze(-1)) + mask_token * mask.unsqueeze(-1)
        else:
            x_masked = x * (1 - mask.unsqueeze(-1))

        return OperationOutput(
            x_masked,
            f"center_crop, ratio={actual_ratio:.3f}, kept_patches={crop_h}x{crop_w}",
            mask,
        )


class OpAEEncoder(nn.Module):
    def __init__(self, config, patch_embed, norm_layer):
        super(OpAEEncoder, self).__init__()

        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config = self.config.dataset_config

        # encoder args
        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        self.patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels
        self.embed_dim = self.model_config.image_encoder.embed_dim
        num_heads = self.model_config.image_encoder.num_heads
        mlp_ratio = self.model_config.image_encoder.mlp_ratio
        depth = self.model_config.image_encoder.depth

        self.patch_embed = patch_embed
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim), requires_grad=False
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.blocks = nn.ModuleList(
            [
                Block(
                    self.embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(self.embed_dim)

        self.initialize_weights()

        # Define operations
        self.operations_handler = ImageOperations()
        self.operations = []
        self.op_probs = []

        for op_name, op_config in config.model_config.operations.items():
            op_func = getattr(self.operations_handler, op_name)
            self.operations.append(partial(op_func, **op_config.params))
            self.op_probs.append(op_config.probability)

        # Normalize probabilities
        total_prob = sum(self.op_probs)
        self.op_probs = [p / total_prob for p in self.op_probs]

    def forward(self, x):
        # Apply operations
        x, augmented_x, op_info = self.apply_operations(x)

        # Add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]  # x: [N, L, D], pos_embed: [1, L+1, D]

        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, augmented_x, op_info

    def apply_operations(self, x):
        B, C, H, W = x.shape
        op_info = []

        augmented_x = torch.zeros_like(x)
        patched_op_x = torch.zeros(
            (B, self.num_patches, self.embed_dim), device=x.device
        )

        for i in range(B):
            op = np.random.choice(self.operations, p=self.op_probs)
            op_name = op.func.__name__

            if op_name in ["random_masking", "center_crop"]:
                x_patch = self.patch_embed(x[i].unsqueeze(0))
                result = op(x_patch, mask_token=self.mask_token)
                patched_op_x[i] = result.transformed.squeeze(0)

                if result.mask is not None:
                    mask_reshaped = result.mask.squeeze(0).reshape(
                        self.patch_embed.patch_shape
                    )
                    mask_upsampled = (
                        F.resize(
                            mask_reshaped.unsqueeze(0).unsqueeze(0).float(),
                            size=(H, W),
                            interpolation=F.InterpolationMode.NEAREST,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                    augmented_x[i] = x[i] * (1 - mask_upsampled).unsqueeze(0)
            else:
                result = op(x[i])
                augmented_x[i] = result.transformed
                patched_op_x[i] = self.patch_embed(
                    result.transformed.unsqueeze(0)
                ).squeeze(0)

            op_info.append(result.info)

        return patched_op_x, augmented_x, op_info

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # intialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class OpAEDecoder(nn.Module):
    def __init__(self, config, patch_embed, norm_layer):
        super(OpAEDecoder, self).__init__()

        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config = self.config.dataset_config

        # decoder args
        embed_dim = self.model_config.image_encoder.embed_dim
        decoder_embed_dim = self.model_config.image_decoder.decoder_embed_dim
        decoder_num_heads = self.model_config.image_decoder.decoder_num_heads
        decoder_depth = self.model_config.image_decoder.decoder_depth
        mlp_ratio = self.model_config.image_encoder.mlp_ratio

        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels

        self.patch_embed = patch_embed
        num_patches = self.patch_embed.num_patches

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_channels, bias=True
        )

        self.initialize_weights()

    def forward(self, x, op_info):
        # embed tokens
        x = self.decoder_embed(x)  # [N, L, D] -> [N, L, D']

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def initialize_weights(self):
        # initialization
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class OpT5AEDecoder(nn.Module):
    def __init__(self, config, patch_embed, norm_layer):
        super().__init__()
        self.config = config
        self.model_config = self.config.model_config
        self.dataset_config = self.config.dataset_config

        # Decoder args
        embed_dim = self.model_config.image_encoder.embed_dim
        decoder_embed_dim = self.model_config.image_decoder.decoder_embed_dim
        decoder_num_heads = self.model_config.image_decoder.decoder_num_heads
        decoder_depth = self.model_config.image_decoder.decoder_depth
        mlp_ratio = self.model_config.image_encoder.mlp_ratio

        img_size = self.dataset_config.preprocess.vision_transforms.params.Resize.size
        patch_size = self.model_config.image_encoder.patch_size
        in_channels = self.model_config.image_encoder.in_channels

        self.patch_embed = patch_embed
        num_patches = self.patch_embed.num_patches

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                OpT5DecoderBlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_channels, bias=True
        )

        # T5 setup for operation encoding
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.t5_encoder = T5EncoderModel.from_pretrained("t5-small")
        # Linear layer to project T5 hidden size to decoder_embed_dim
        self.t5_projection = nn.Linear(
            self.t5_encoder.config.hidden_size, decoder_embed_dim
        )

        # T5 parameters should not be updated
        for param in self.t5_encoder.parameters():
            param.requires_grad = False

        # self.operation_encodings = self.precompute_operation_encodings()

        self.initialize_weights()

    def precompute_operation_encodings(self):
        operations = list(self.model_config.operations.keys())
        encodings = {}

        for op in operations:
            inputs = self.t5_tokenizer(
                op, return_tensors="pt", padding=True, truncation=True
            )
            with torch.no_grad():
                outputs = self.t5_encoder(**inputs)
            encodings[op] = outputs.last_hidden_state.mean(dim=1)  # Average pooling

        # Free up memory
        del self.t5_tokenizer
        del self.t5_encoder
        torch.cuda.empty_cache()

        return encodings

    def get_operation_encoding(self, op_info):
        """실시간으로 operation과 파라미터 정보를 텍스트로 변환하여 인코딩"""
        # op_info is a string like "gaussian_blur, kernel_size=5, sigma=1.234"
        batch_texts = []
        for info in op_info:
            # 이미 문자열 형태로 받은 operation 정보를 그대로 사용
            batch_texts.append(info)

        # Tokenize all texts in batch
        inputs = self.t5_tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.t5_encoder.device)

        # Get T5 encodings
        with torch.no_grad():
            outputs = self.t5_encoder(**inputs)

        # Average pooling over sequence length
        encodings = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]

        # Project to decoder dimension
        projected_encodings = self.t5_projection(
            encodings
        )  # [batch_size, decoder_embed_dim]

        projected_encodings = projected_encodings.unsqueeze(1)

        return projected_encodings

    def forward(self, x, op_info):
        # Embed tokens
        x = self.decoder_embed(x)  # [N, L, D] -> [N, L, D']

        # # Get operation encodings and project
        # op_encodings = [self.operation_encodings[info] for info in op_info]
        # op_encodings = torch.stack(op_encodings).to(x.device)
        # op_encodings = self.t5_projection(op_encodings)  # [N, D']
        op_encodings = self.get_operation_encoding(op_info)  # [N, D']

        # Add position embeddings
        x = x + self.decoder_pos_embed  # [N, L, D']

        # Apply Transformer blocks with cross-attention
        for blk in self.decoder_blocks:
            x = blk(x, op_encodings)

        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        # Remove cls token
        x = x[:, 1:, :]

        return x

    def initialize_weights(self):
        # initialization
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
