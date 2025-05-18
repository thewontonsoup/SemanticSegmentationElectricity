import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from src.preprocessing.subtile_esd_hw02 import TileMetadata, Subtile
import torch


def restitch_and_plot(options, datamodule, model, parent_tile_id, satellite_type="sentinel2", rgb_bands=[3, 2, 1],
                      image_dir: None | str | os.PathLike = None):
    """
    Plots the 1) rgb satellite image 2) ground truth 3) model prediction in one row.

    Args:
        options: EvalConfig
        datamodule: ESDDataModule
        model: ESDSegmentation
        parent_tile_id: str
        satellite_type: str
        rgb_bands: List[int]
    """
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)

    rgb_sat_img, gt, pred = restitch_eval(
        dir=options.processed_dir / "Val",
        satellite_type=satellite_type,
        tile_id=parent_tile_id,
        range_x=(0, options.tile_size_gt - 1),
        range_y=(0, options.tile_size_gt - 1),
        datamodule=datamodule,
        model=model
    )

    fig, axs = plt.subplots(nrows=1, ncols=3)

    # make sure to use cmap=cmap, vmin=-0.5 and vmax=3.5 when running
    # axs[i].imshow on the 1d images in order to have the correct
    # colormap for the images.
    # On one of the 1d images' axs[i].imshow, make sure to save its output as
    # `im`, i.e, im = axs[i].imshow
    fig.suptitle(f"Parent Tile ID: {parent_tile_id[4:]}")
    axs[0].set_title("Satellite Image")
    axs[0].imshow(rgb_sat_img[0, rgb_bands, :, :].transpose(1, 2, 0))
    axs[1].set_title("Ground Truth")
    axs[1].imshow(gt, cmap=cmap, vmin=-0.5, vmax=3.5)
    axs[2].set_title("Prediction")
    im = axs[2].imshow(pred, cmap=cmap, vmin=-0.5, vmax=3.5)

    # The following lines sets up the colorbar to the right of the images
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Sttlmnts Wo Elec', 'No Sttlmnts Wo Elec', 'Sttlmnts W Elec', 'No Sttlmnts W Elec'])
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "restitched_visible_gt_predction.png")
        plt.close()


def restitch_eval(dir: str | os.PathLike, satellite_type: str, tile_id: str, range_x: Tuple[int, int],
                  range_y: Tuple[int, int], datamodule, model) -> np.ndarray:
    """
    Given a directory of processed subtiles, a tile_id and a satellite_type,
    this function will retrieve the tiles between (range_x[0],range_y[0])
    and (range_x[1],range_y[1]) in order to stitch them together to their
    original image. It will also get the tiles from the datamodule, evaluate
    it with model, and stitch the ground truth and predictions together.

    Input:
        dir: str | os.PathLike
            Directory where the subtiles are saved
        satellite_type: str
            Satellite type that will be stitched
        tile_id: str
            Tile id that will be stitched
        range_x: Tuple[int, int]
            Range of tiles that will be stitched on width dimension [0,5)]
        range_y: Tuple[int, int]
            Range of tiles that will be stitched on height dimension
        datamodule: pytorch_lightning.LightningDataModule
            Datamodule with the dataset
        model: pytorch_lightning.LightningModule
            LightningModule that will be evaluated

    Output:
        stitched_image: np.ndarray
            Stitched image, of shape (time, bands, width, height)
        stitched_ground_truth: np.ndarray
            Stitched ground truth of shape (width, height)
        stitched_prediction_subtile: np.ndarray
            Stitched predictions of shape (width, height)
    """
    tracker = [[] for _ in range(range_x[0], range_x[1] + 1)]
    satellite_subtile_rows = [[] for _ in range(range_x[0], range_x[1] + 1)]
    ground_truth_subtile_rows = [[] for _ in range(range_x[0], range_x[1] + 1)]
    predictions_subtile_rows = [[] for _ in range(range_x[0], range_x[1] + 1)]

    for (X, y, metadata) in datamodule.val_dataset:
        if metadata.parent_tile_id == tile_id and (range_x[0] <= metadata.x_gt <= range_x[1]) and (range_y[0] <= metadata.y_gt <= range_y[1]):
            i, j = metadata.x_gt, metadata.y_gt
            subtile = Subtile().load(dir / "subtiles" / f"{tile_id}_{i}_{j}.npz")
            X = X.float().unsqueeze(0)
            model.eval()
            y = y.numpy()
            with torch.no_grad():
                prediction = model(X)
            prediction = prediction.cpu().numpy()

            tracker[i].append(j)
            satellite_subtile_rows[i].append(subtile.satellite_stack[satellite_type])
            ground_truth_subtile_rows[i].append(y)
            predictions_subtile_rows[i].append(prediction)

    for i in range(range_x[0], range_x[1] + 1):
        sorted_indices = sorted(range(len(tracker[i])), key=lambda x: tracker[i][x])
        satellite_subtile_rows[i] = [satellite_subtile_rows[i][j] for j in sorted_indices]
        ground_truth_subtile_rows[i] = [ground_truth_subtile_rows[i][j] for j in sorted_indices]
        predictions_subtile_rows[i] = [predictions_subtile_rows[i][j] for j in sorted_indices]

    satellite_subtile = [np.concatenate(satellite_subtile_rows[i], axis=-1) for i in range(range_x[0], range_x[1] + 1)]
    ground_truth_subtile = [np.concatenate(ground_truth_subtile_rows[i], axis=-1) for i in
                            range(range_x[0], range_x[1] + 1)]
    predictions_subtile = [np.concatenate(predictions_subtile_rows[i], axis=-1) for i in
                           range(range_x[0], range_x[1] + 1)]

    return (np.concatenate(satellite_subtile, axis=-2),
            np.concatenate(ground_truth_subtile, axis=-2).squeeze(0),
            np.concatenate(predictions_subtile, axis=-2).squeeze(0)[0]
            )
