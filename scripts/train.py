import pyprojroot
import sys
root = pyprojroot.here()
sys.path.append(str(root))
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from lightning.pytorch.loggers import WandbLogger

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
    EarlyStopping
)

from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
import wandb
import yaml


@dataclass
class ESDConfig:
    """
    IMPORTANT: This class is used to define the configuration for the experiment
    Please make sure to use the correct types and default values for the parameters
    and that the path for processed_dir contain the tiles you would like 
    """
    processed_dir: str | os.PathLike = root / 'data/processed/4x4'
    raw_dir: str | os.PathLike = root / 'data/raw/Train'
    # selected_bands: Dict[str, List[str]] = field(default_factory=lambda: {"sentinel2": ["04", "03", "02"], "viirs": ["0"], "gt": ["0"]})
    selected_bands: None = None
    model_type: str = "UNetPlusPlus"
    tile_size_gt: int = 4
    batch_size: int = 16
    max_epochs: int = 2
    seed: int = 12378921
    learning_rate: float = 0.001
    num_workers: int = 0
    accelerator: str = "gpu"
    devices: int = 1
    in_channels: int = 99
    out_channels: int = 4
    depth: int = 2
    n_encoders: int = 2
    embedding_size: int = 64
    pool_sizes: str = "5,5,2" # List[int] = [5,5,2]
    kernel_size: int = 3
    scale_factor: int = 50
    encoder_name: str = "resnet101"
    encoder_weights: str = "imagenet"
    decoder_channels: str = "256, 128, 64, 32, 16"
    activation: str | None = None
    wandb_run_name: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))


def train(options: ESDConfig):
    """
    Prepares datamodule and model, then runs the training loop

    Inputs:
        options: ESDConfig
            options for the experiment
    """
    # Initialize the weights and biases logger
    wandb_logger = WandbLogger(project='esd_segmentation',
                               name=options.wandb_run_name,
                               log_model="all",
                               config=options.__dict__)

    # initiate the ESDDatamodule
    # use the options object to initiate the datamodule correctly
    # make sure to prepare_data in case the data has not been preprocessed
    datamodule = ESDDataModule(
        options.processed_dir,
        options.raw_dir,
        options.selected_bands,
        options.tile_size_gt,
        options.batch_size,
        options.num_workers,
        options.seed
    )
    datamodule.prepare_data()

    options.decoder_channels = tuple(int(i) for i in options.decoder_channels.split(','))

    # create a dictionary with the parameters to pass to the models
    parameters = {
        'model_type': options.model_type,
        'in_channels': options.in_channels,
        'out_channels': options.out_channels,
        'learning_rate': options.learning_rate,
        'model_params': {
            'depth': options.depth,
            'n_encoders': options.n_encoders,
            'embedding_size': options.embedding_size,
            'pool_sizes': [int(i) for i in options.pool_sizes.split(',')],
            'kernel_size': options.kernel_size,
            'scale_factor': options.scale_factor,
            'encoder_name': options.encoder_name,
            'encoder_weights': options.encoder_weights,
            'decoder_channels': options.decoder_channels,
            'activation': options.activation
        }
    }

    # initialize the ESDSegmentation module
    model = ESDSegmentation(**parameters)

    # Use the following callbacks, they're provided for you,
    # but you may change some of the settings
    # ModelCheckpoint: saves intermediate results for the neural network
    # in case it crashes
    # LearningRateMonitor: logs the current learning rate on weights and biases
    # RichProgressBar: nicer looking progress bar (requires the rich package)
    # RichModelSummary: shows a summary of the model before training (requires rich)
    callbacks = [
        ModelCheckpoint(
            dirpath=root / 'models' / options.model_type,
            filename='{epoch}-{loss:.2f}-{train_accuracy:.2f}-{train_loss:.2f}-{val_accuracy:.2f}-{val_loss:.2f}',
            save_top_k=1,
            save_last=True,
            verbose=True,
            monitor='val_loss',
            mode='min',
            every_n_train_steps=1000
        ),
        LearningRateMonitor(),
        RichProgressBar(),
        RichModelSummary(max_depth=3),
        # EarlyStopping(monitor='train_loss', patience=3, mode='min'),
        # EarlyStopping(monitor='train_accuracy', patience=3, mode='max'),
        # EarlyStopping(monitor='val_loss', patience=3, mode='min'),
        # EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
    ]

    # create a pytorch Trainer
    # see pytorch_lightning.Trainer
    # make sure to use the options object to load it with the correct options
    trainer = pl.Trainer(
        max_epochs=options.max_epochs,
        accelerator=options.accelerator,
        devices=options.devices,
        callbacks=callbacks,
        logger=wandb_logger
    )

    # run trainer.fit
    # make sure to use the datamodule option
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == '__main__':
    # load dataclass arguments from yml file
    with open(root / 'scripts' / 'train.yml', 'r') as f:
        yml = yaml.safe_load(f)

    if yml is not None:
        config = ESDConfig(**yml)
    else:
        config = ESDConfig()

    parser = ArgumentParser()

    parser.add_argument("-p", "--processed_dir", type=str, default=config.processed_dir, help="Path to processed directory")
    parser.add_argument("--raw_dir", type=str, default=config.raw_dir, help='Path to raw directory')

    parser.add_argument("--model_type", type=str, help="The model to initialize.", default=config.model_type)
    
    parser.add_argument("--batch_size", type=int, help="The batch size for training model", default=config.batch_size)
    parser.add_argument("--max_epochs", type=int, help="Number of epochs to train for", default=config.max_epochs)

    parser.add_argument("--learning_rate", type=float, help="The learning rate for training model", default=config.learning_rate)
    parser.add_argument("--num_workers", type=int, help="Number of workers for the dataloader", default=config.num_workers)
    parser.add_argument('--in_channels', type=int, default=config.in_channels, help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=config.out_channels, help='Number of output channels')
    parser.add_argument('--depth', type=int, help="Depth of the encoders (CNN only)", default=config.depth)
    parser.add_argument('--n_encoders', type=int, help="Number of encoders (Unet only)", default=config.n_encoders)
    parser.add_argument('--embedding_size', type=int, help="Embedding size of the neural network (CNN/Unet)", default=config.embedding_size)
    parser.add_argument('--pool_sizes', help="A comma separated list of pool_sizes (CNN only)", type=str, default=config.pool_sizes)
    parser.add_argument('--kernel_size', help="Kernel size of the convolutions", type=int, default=config.kernel_size)
    parser.add_argument('--scale_factor', help="Scale factor between the labels and the image (U-Net, Transfer Resnet, and UNet++)", type=int, default=config.scale_factor)
    # --pool_sizes=5,5,2 to call it correctly
    parser.add_argument('--encoder_name', help="Encoder name for the model (UNet++ only)", type=str, default=config.encoder_name)
    parser.add_argument('--encoder_weights', help="Encoder weights for the model (UNet++ only)", type=str, default=config.encoder_weights)
    parser.add_argument('--decoder_channels', help="Decoder channels for the model (UNet++ only)", type=str, default=config.decoder_channels)
    # --decoder_channels=256,128,64,32,16 to call it correctly
    parser.add_argument('--activation', help="Activation function for the model (UNet++ only)", type=str, default=config.activation)

    parse_args = parser.parse_args()

    config = ESDConfig(**parse_args.__dict__)

    train(ESDConfig(**parse_args.__dict__))
