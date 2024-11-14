from io import BytesIO
from pandas import DataFrame
import torch
from src.datasets.base_dataset_builder import BaseDatasetBuilder
import pytorch_lightning as pl
from src.common.registry import registry
import torchvision.datasets as datasets
from torch.utils.data import random_split
from datasets import load_dataset
import os
from PIL import Image
from src.datasets.transforms.vision_transforms_utils import Normalise


class RobustImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, huggingface_dataset, transform):
        self.dataset = huggingface_dataset
        self.transform = transform
        self.normalise = Normalise()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        while True:
            try:
                item = self.dataset[idx]
                image = item["image"].convert("RGB")
                image = self.transform(image)
                return {"pixel_values": image}
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                # # Create a random color image tensor (0, 1) torch.float32 (R, G, B) 각각 하나의 색
                # image = torch.rand(3)
                # image = self.normalise(image)
                # image = image.unsqueeze(1).unsqueeze(2)
                # image = image.repeat(1, 224, 224)
                return {"pixel_values": None}
        # while True:
        #     item = self.dataset[idx]
        #     image = item['image'].convert('RGB')
        #     image = self.transform(image)
        #     return {'pixel_values': image}


@registry.register_builder("imagenet1k")
class HuggingfaceImageNetDatasetModule(BaseDatasetBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = "imagenet1k"
        self.dataset_path = f"{os.getenv('WORKSPACE', '/workspace')}/datasets"
        self.transforms_name = self.config.dataset_config.preprocess.name
        self.min_batch_size = 3 * self.config.training.batch_size // 4

    def preprocess(self, split):
        data_transform_cls = registry.get_preprocessor_class(self.transforms_name)
        data_transforms_obj = data_transform_cls(self.config, split)
        return data_transforms_obj

    def data_setup(self, split):
        transform = self.preprocess(split)

        # Huggingface 데이터셋 로드
        dataset = load_dataset(f"{self.dataset_path}/{self.dataset_name}", split=split)
        return RobustImageNetDataset(dataset, transform)
        # dataset = dataset.with_transform(transform=transform)
        # return dataset

    def collate_fn(self, batch):
        # filter out None values
        images = [x["pixel_values"] for x in batch if x["pixel_values"] is not None]
        # if len(images) < self.min_batch_size:
        #     return None  # 최소 배치 크기보다 작으면 None 반환
        return {"pixel_values": torch.stack(images)}

    def train_dataloader(self):
        train_dataset = self.data_setup("train")
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            # pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        val_dataset = self.data_setup("validation")
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            # pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        test_dataset = self.data_setup("test")
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
