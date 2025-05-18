import pyprojroot
import sys
import os
root = pyprojroot.here()
sys.path.append(str(root))
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
from typing import List
from dataclasses import dataclass, field
from pathlib import Path

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary
)

from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.preprocessing.subtile_esd_hw02 import Subtile
from src.visualization.restitch_plot import (
    restitch_eval,
    restitch_and_plot
)
from typing import Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import tifffile
import wandb
import yaml


@dataclass
class EvalConfig:
    processed_dir: str | os.PathLike = root / 'data/processed/4x4'
    raw_dir: str | os.PathLike = root / 'data/raw/Train'
    results_dir: str | os.PathLike = root / 'data/predictions' / "UNetPlusPlus"
    # selected_bands: Dict[str, List[str]] = field(default_factory=lambda: {"sentinel2": ["04", "03", "02"], "viirs": ["0"], "gt": ["0"]})
    selected_bands: None = None
    tile_size_gt: int = 4
    batch_size: int = 8
    seed: int = 12378921
    num_workers: int = 0 # 11
    accelerator: str = "gpu"
    devices: int = 1
    model_path: str | os.PathLike = root / "models" / "UNetPlusPlus" / "last.ckpt"


def main(options):
    """
    Prepares datamodule and loads model, then runs the evaluation loop

    Inputs:
        options: EvalConfig
            options for the experiment
    """
    options.processed_dir = Path(options.processed_dir)
    options.raw_dir = Path(options.raw_dir)
    options.results_dir = Path(options.results_dir)
    options.model_path = Path(options.model_path)

    # Load datamodule
    datamodule = ESDDataModule(
        options.processed_dir,
        options.raw_dir,
        options.selected_bands,
        options.tile_size_gt,
        options.batch_size,
        options.num_workers,
        options.seed
    )
    wandb.init(project="esd-segmentation")

    # load model from checkpoint at options.model_path
    model = ESDSegmentation.load_from_checkpoint(checkpoint_path=options.model_path)
    # set the model to evaluation mode (model.eval())
    model.eval()

    # instantiate pytorch lightning trainer
    trainer = pl.Trainer(
        accelerator=options.accelerator,
        devices=options.devices)
    
    # run the validation loop with trainer.validate
    trainer.validate(model, datamodule=datamodule)

    # run restitch_and_plot
    # for every subtile in options.processed_dir/Val/subtiles
    # run restitch_eval on that tile followed by picking the best scoring class
    # save the file as a tiff using tifffile
    # save the file as a png using matplotlib
    val_path = Path(options.processed_dir, "Val", "subtiles")
    val_tiles = sorted(set(p.stem.split("_")[0] for p in val_path.glob("*.npz")))
    for parent_tile_id in val_tiles:
        print(f"Restitching: {parent_tile_id}")
        options.results_dir = Path(options.results_dir)
        (options.results_dir  / parent_tile_id).mkdir(parents=True, exist_ok=True)
        restitch_and_plot(options, datamodule, model, parent_tile_id, image_dir=options.results_dir / parent_tile_id)
        # Next code unnecessary, remove continue line if you want to run the restitch_eval code
        # and save the tiff/png files
        continue
        _, _, y_pred = restitch_eval(dir=val_path.parent,
                                     satellite_type="sentinel2",
                                     tile_id=parent_tile_id,
                                     range_x=(0, options.tile_size_gt - 1),
                                     range_y=(0, options.tile_size_gt - 1),
                                     datamodule=datamodule,
                                     model=model)


        # Save the file as TIFF
        tifffile.imwrite(options.results_dir / f"{parent_tile_id}.tiff", y_pred)

        # freebie: plots the predicted image as a png with the correct colors
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(y_pred, vmin=-0.5, vmax=3.5,cmap=cmap)
        plt.savefig(options.results_dir / f"{parent_tile_id}.png")


if __name__ == '__main__':
    # load dataclass arguments from yml file
    with open(root / 'scripts' / 'evaluate.yml', 'r') as f:
        yml = yaml.safe_load(f)

    if yml is not None:
        config = EvalConfig(**yml)
    else:
        config = EvalConfig()
    
    parser = ArgumentParser()

    parser.add_argument("-p", "--processed_dir", type=str, default=config.processed_dir, help="Path to processed directory")
    parser.add_argument("--raw_dir", type=str, default=config.raw_dir, help='Path to raw directory')
    parser.add_argument("--results_dir", type=str, default=config.results_dir, help="Path to results directory")
    parser.add_argument("--batch_size", type=int, help="The batch size for training model", default=config.batch_size)
    parser.add_argument("--num_workers", type=int, help="Number of workers for the dataloader", default=config.num_workers)
    parser.add_argument("--model_path", type=str, help="Model path.", default=config.model_path)
    
    parse_args = parser.parse_args()

    main(EvalConfig(**parse_args.__dict__))
