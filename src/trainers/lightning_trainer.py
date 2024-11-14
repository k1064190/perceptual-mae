# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from typing import Optional

import wandb

from src.common.registry import registry
from src.trainers.base_trainer import BaseTrainer
from src.utils import utils

# from src.utils.logger import TensorboardLogger, setup_output_folder
from omegaconf import OmegaConf
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.common.data_types import Datasets
from src.models.base_model import BaseModel

logger = logging.getLogger(__name__)


class SkipEmptyBatchCallback(Callback):
    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, *args, **kwargs
    ):
        if batch is None or len(batch["pixel_values"]) == 0:
            logger.warning(f"Empty batch {batch_idx} detected, skipping...")
            return -1  # 배치 스킵

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, *args, **kwargs
    ):
        if batch is None or len(batch["pixel_values"]) == 0:
            logger.warning(f"Empty batch {batch_idx} detected, skipping...")
            return -1  # 배치 스킵


@registry.register_trainer("lightning")
class LightningTrainer(BaseTrainer):
    def __init__(
        self,
        config,
        datasets: Datasets,
        model: BaseModel,
        *,
        local_experiment_data_dir,
        remote_experiment_data_dir,
        s3_client=None,
    ):
        super().__init__()

        self.config = config

        # define your loaders; data_module, train_loader, val_loader, test_loader
        # self.data_module = datasets.data_module
        self.train_loader = datasets.train_loader
        self.val_loader = datasets.val_loader
        self.test_loader = datasets.test_loader

        self.model = model
        self.s3_client = s3_client

        # setup data ----

        if self.config.user_config.save_locally:
            OmegaConf.save(
                config=config,
                f=os.path.join(
                    local_experiment_data_dir,
                    f"{self.config.user_config.experiment_name}_config.yaml",
                ),
            )
            self.log_dir = os.path.join(local_experiment_data_dir, "train_outputs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

        else:
            assert s3_client is not None
            self.s3_client.upload_config_to_s3(self.config)
            self.log_dir = os.path.join(remote_experiment_data_dir, "train_outputs")

        # setup trainer ----

        trainer_config = self.config.trainer.params

        # max_updates, max_epochs = utils.get_max_updates(
        #     trainer_config.max_steps,
        #     trainer_config.max_epochs,
        #     self.train_loader,
        #     trainer_config.accumulate_grad_batches,
        # )
        #
        # self.load_callbacks()

        # tb_writer = TensorBoardLogger(
        #     "{}/tensorboard_logs/".format(self.log_dir),
        #     name=self.config.user_config.experiment_name,
        # )
        wandb_logger = WandbLogger(
            project=self.config.user_config.experiment_name,
            name=f"{self.config.user_config.experiment_name}_{self.config.model_config.name}",
            dir=f"{self.log_dir}/wandb_logs",
        )

        callbacks = []

        # 설정 파일에서 콜백 설정을 읽어 동적으로 생성
        for callback_config in self.config.training.get("callbacks", []):
            callback_type = callback_config.type
            callback_params = callback_config.get("params", {})

            if callback_type == "ModelCheckpoint":
                callback_params["dirpath"] = self.log_dir  # dirpath 설정
                callback = ModelCheckpoint(**callback_params)
                callbacks.append(callback)

        # callbacks.append(SkipEmptyBatchCallback())

        # setup trainer

        lightning_params_dict = OmegaConf.to_container(trainer_config, resolve=True)

        # max epochs specified in trainer_config;
        self.trainer = Trainer(
            default_root_dir=self.log_dir,
            logger=wandb_logger,
            callbacks=callbacks,
            **lightning_params_dict,
        )

    # def load_callbacks(self) -> None:
    #     callbacks = build_callbacks(self.config)
    #     callbacks_list = []
    #     for callbacks_name, _ in callbacks.items():
    #         # metric_val = self.metrics[metric_name](logits, y)
    #         callbacks_list.append(callbacks[callbacks_name])
    #     return callbacks_list

    def train(self) -> None:
        logger.info("===== Model =====")
        logger.info(self.model)
        utils.print_model_parameters(self.model)

        logger.info("Starting training...")

        checkpoint_path = self._load_checkpoint()
        if checkpoint_path is not None:
            print(
                "loading previous checkpoint in the trainer: {}".format(checkpoint_path)
            )
            self.prev_checkpoint_path = checkpoint_path
            self.trainer.fit(
                self.model,
                self.train_loader,
                self.val_loader,
                ckpt_path=checkpoint_path,
            )
        else:
            print("no checkpoint to be loaded")
            self.trainer.fit(self.model, self.train_loader, self.val_loader)
        # perform final validation step at the end of training;
        self.val()

        logger.info("Finished training!")

    def val(self) -> None:
        # Don't run if current iteration is divisble by
        # val check interval as it will just be a repeat

        logger.info("Stepping into final validation check")
        self.trainer.validate(self.model, self.val_loader)

    def test(self) -> None:
        logger.info("===== Model =====")
        logger.info(self.model)
        utils.print_model_parameters(self.model)

        logger.info("Starting Testing...")

        checkpoint_path = self._load_checkpoint()
        if checkpoint_path is not None:
            self.prev_checkpoint_path = checkpoint_path
            print(
                "loading previous checkpoint in the trainer: {}".format(checkpoint_path)
            )
            self.trainer.test(self.model, self.test_loader, ckpt_path=checkpoint_path)
        else:
            print("no checkpoint to be loaded")
            print(
                "since None checkpoint path, current ongoing training / randomly initiated model will be used"
            )
            self.trainer.test(self.model, self.test_loader, ckpt_path=None)

        logger.info("Finished testing!")

    def configure_device(self) -> None:
        pass

    def configure_seed(self) -> None:
        seed = self.config.training.seed
        seed_everything(seed)

    def _load_checkpoint(self) -> Optional[str]:
        if self.config.model_config.load_checkpoint is not None:
            checkpoint_path = self.config.model_config.load_checkpoint
            return checkpoint_path
        else:
            return None
