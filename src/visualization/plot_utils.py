""" This module contains functions for plotting satellite images. """
import os
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from ..preprocessing.file_utils import Metadata
from ..preprocessing.preprocess_sat import minmax_scale
from ..preprocessing.preprocess_sat import (
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_landsat,
    preprocess_viirs
)


def plot_viirs_histogram(
        viirs_stack: np.ndarray,
        image_dir: None | str | os.PathLike = None,
        n_bins=100
        ) -> None:
    """
    This function plots the histogram over all VIIRS values.
    note: viirs_stack is a 4D array of shape (time, band, height, width)

    Parameters
    ----------
    viirs_stack : np.ndarray
        The VIIRS image stack volume.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    # fill in the code here
    #flatten the data by bands and make seperate histograms for each band.
    plt.hist(viirs_stack.flatten(), bins = n_bins)
    plt.ylabel("Frequency")
    plt.title("VIIRS histogram")
    if image_dir is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(Path(image_dir) / "VIIRS_histogram.png")
        plt.close()
    return

    raise NotImplementedError


def plot_sentinel1_histogram(
        sentinel1_stack: np.ndarray,
        metadata: List[Metadata],
        image_dir: None | str | os.PathLike = None,
        n_bins=20
        ) -> None:
    """
    This function plots the Sentinel-1 histogram over all Sentinel-1 values.
    note: sentinel1_stack is a 4D array of shape (time, band, height, width)

    Parameters
    ----------
    sentinel1_stack : np.ndarray
        The Sentinel-1 image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-1 image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    # fill in the code here

    #print(sentinel2_stack.shape)
    #print(len(metadata[0]))
    #print(metadata[0][0].bands)

    sentinel1_stack = sentinel1_stack.transpose(2, 0, 1, 3, 4)
    #print(sentinel1_stack.shape)
    fig, axis = plt.subplots(2, len(sentinel1_stack))
    plt.suptitle("Sentinel 1 Histogram")
    for i in range(len(sentinel1_stack)):
        axis[0][i].hist(sentinel1_stack[i].flatten(),  log = True, bins=n_bins)
        axis[1][i].hist(sentinel1_stack[i].flatten(), log=False, bins=n_bins)
        axis[0][i].set_title("Band: {band}, Log-Scale".format(band = metadata[0][0].bands[i]))
        axis[1][i].set_title("Band: {band}, Linear Scale".format(band=metadata[0][0].bands[i]))
    plt.tight_layout()
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel1_histogram.png")
        plt.close()
    return
    raise NotImplementedError


def plot_sentinel2_histogram(
        sentinel2_stack: np.ndarray,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None,
        n_bins=20) -> None:
    """
    This function plots the Sentinel-2 histogram over all Sentinel-2 values.

    Parameters
    ----------
    sentinel2_stack : np.ndarray
        The Sentinel-2 image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-2 image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    # fill in the code here
    #print(sentinel2_stack.shape)
    sentinel2_stack = sentinel2_stack.transpose(2, 0, 1, 3, 4)
    #print(sentinel2_stack.shape)
    #print(len(metadata[0]))
    #print(metadata[0][0].bands)
    fig, axis = plt.subplots(len(sentinel2_stack)//4, len(sentinel2_stack)//3, figsize = (8,8))
    plt.suptitle("Sentinel2 Histogram")
    for i in range(len(sentinel2_stack)):
        axis[i//4][i%4].hist(sentinel2_stack[i].flatten(),  log = True, bins=n_bins)
        axis[i//4][i%4].set_title("Band: {band}".format(band = metadata[0][0].bands[i]))
    plt.tight_layout()
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel2_histogram.png")
        plt.close()
    return
    raise NotImplementedError


def plot_landsat_histogram(
        landsat_stack: np.ndarray,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None,
        n_bins=20
        ) -> None:
    """
    This function plots the landsat histogram over all landsat values over all
    tiles present in the landsat_stack.

    Parameters
    ----------
    landsat_stack : np.ndarray
        The landsat image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the landsat image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    #print(landsat_stack.shape)
    # fill in the code here
    landsat_stack= landsat_stack.transpose(2, 0, 1, 3, 4)
    #print(landsat_stack.shape)
    #print(len(metadata[0]))
    #print(metadata[0][0].bands)
    fig, axis = plt.subplots((len(landsat_stack)+1)//4, (len(landsat_stack)+1)//3, figsize = (10,10))
    plt.suptitle("Landsat Histogram by bands")
    for i in range(len(landsat_stack)):
        axis[i//4][i%4].hist(landsat_stack[i].flatten(), bins=n_bins)
        axis[i//4][i%4].set_title("Band: {band}".format(band = metadata[0][0].bands[i]))
        plt.setp(axis[i//4][i%4].get_xticklabels(), rotation=30, ha='right')
        #print(i//4, i%4)
    fig.delaxes(axis[(len(landsat_stack)+1)//4-1][(len(landsat_stack)+1)//3-1])
    plt.tight_layout()
    #plt.title("Landsat Histogram")

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "landsat_histogram.png")
        plt.close()
    return
    raise NotImplementedError


def plot_gt_counts(ground_truth: np.ndarray,
                   image_dir: None | str | os.PathLike = None
                   ) -> None:
    """
    This function plots the ground truth histogram over all ground truth
    values over all tiles present in the groundTruth_stack.

    Parameters
    ----------
    groundTruth : np.ndarray
        The ground truth image stack volume.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    # fill in the code here
    plt.hist(ground_truth.flatten())
    plt.ylabel("Frequency")
    plt.title("Ground Truth Histogram")

    if image_dir is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(Path(image_dir) / "ground_truth_histogram.png")
        plt.close()
    return
    raise NotImplementedError


def plot_viirs(
        viirs: np.ndarray, plot_title: str = '',
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """ This function plots the VIIRS image.

    Parameters
    ----------
    viirs : np.ndarray
        The VIIRS image.
    plot_title : str
        The title of the plot.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    # fill in the code here
    plt.imshow(viirs)
    plt.title(plot_title)
    if image_dir is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(Path(image_dir) / "viirs_max_projection.png")
        plt.close()
    return
    raise NotImplementedError


def plot_viirs_by_date(
        viirs_stack: np.array,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None) -> None:
    """
    This function plots the VIIRS image by band in subplots.

    Parameters
    ----------
    viirs_stack : np.ndarray
        The VIIRS image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the VIIRS image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    # fill in the code here
    figure, axis = plt.subplots(2, (len(metadata)+1)//2)
    plt.suptitle("Viirs by Date")
    j = (len(metadata)+1)//2
    for i in range(len(metadata)):
        axis[i//j][i-j].set_title(metadata[i].time)
        axis[i//j][i-j].imshow(viirs_stack[i][0])
        #axis[i].title.set_size(10)
        axis[i//j][i-j].axes.get_yaxis().set_visible(False)
        axis[i//j][i-j].axes.get_xaxis().set_visible(False)
    if len(metadata) % 2 == 1:
        figure.delaxes(axis[1][(len(metadata)+1)//2-1])
    if image_dir is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(Path(image_dir) / "viirs_plot_by_date.png")
        plt.close()
    return
    raise NotImplementedError


def preprocess_data(
        satellite_stack: np.ndarray,
        satellite_type: str
        ) -> np.ndarray:
    """
    This function preprocesses the satellite data based on the satellite type.

    Parameters
    ----------
    satellite_stack : np.ndarray
        The satellite image stack volume.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    np.ndarray
    """
    process_funcs = {"sentinel2": preprocess_sentinel2, "sentinel1": preprocess_sentinel1,
        "landsat": preprocess_landsat, "viirs":preprocess_viirs}
    if satellite_type in process_funcs:
        return process_funcs[satellite_type](satellite_stack)
    raise ValueError("That's not a valid satellite type")


def create_rgb_composite_s1(
        processed_stack: np.ndarray,
        bands_to_plot: List[List[str]],
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function creates an RGB composite for Sentinel-1.
    This function needs to extract the band identifiers from the metadata
    and then create the RGB composite. For the VV-VH composite, after
    the subtraction, you need to minmax scale the image.

    Parameters
    ----------
    processed_stack : np.ndarray
        The Sentinel-1 image stack volume.
    bands_to_plot : List[List[str]]
        The bands to plot. Cannot accept more than 3 bands.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-1 image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """

    '''
    print(processed_stack.shape)
    print(bands_to_plot)
    print(len(metadata))
    print(metadata)
    '''

    if len(bands_to_plot) > 3:
        raise ValueError("Cannot plot more than 3 bands.")
    def normalized_pixel_value(pixel, min, max):
        return (pixel-min) / (max-min)
    vectorized_function = np.vectorize(normalized_pixel_value)
    clip = 0.05

    fig, axis = plt.subplots(len(processed_stack), len(bands_to_plot), figsize = (10,10))
    plt.suptitle("Sentinel 1 Plots")

    for i in range(len(processed_stack)):
        for j in range(len(bands_to_plot)):
            coords = []
            '''
            for k in range(len(bands_to_plot[j])-1):
                coords.append(processed_stack[i][k])
            '''
            coords.append(processed_stack[i][metadata[i].bands.index("VV")])
            coords.append(processed_stack[i][metadata[i].bands.index("VH")])

            #print(coords[0][0][0:9])
            #print(coords[1][0][0:9])

            coords[0] = vectorized_function(coords[0], np.min(coords[0]), np.max(coords[0]))
            coords[0] = np.clip(coords[0], np.quantile(coords[0], clip), np.quantile(coords[0], 1 - clip))
            coords[0] = vectorized_function(coords[0], np.min(coords[0]), np.max(coords[0]))


            coords[1] = vectorized_function(coords[1], np.min(coords[1]), np.max(coords[1]))
            coords[1] = np.clip(coords[1], np.quantile(coords[1], clip), np.quantile(coords[1], 1 - clip))
            coords[1] = vectorized_function(coords[1], np.min(coords[1]), np.max(coords[1]))


            #print(coords[0][0][0:9])
            #print(coords[1][0][0:9])

            blue = coords[0]-coords[1]
            blue = vectorized_function(blue, np.min(blue), np.max(blue))
            blue = np.clip(blue, np.quantile(blue, clip), np.quantile(blue, 1-clip))
            blue = vectorized_function(blue, np.min(blue), np.max(blue))
            coords.append(blue)

            #print(blue[0][0:9], '\n\n')

            coords = np.stack(coords)

            if(len(bands_to_plot) > 1):
                axis[i][j].imshow(coords.T)
                axis[i][j].set_title(metadata[i].time)
                axis[i][j].axes.get_xaxis().set_visible(False)
                axis[i][j].axes.get_yaxis().set_visible(False)
            else:
                axis[i].imshow(coords.T)
                axis[i].set_title(metadata[i].time)
                axis[i].axes.get_xaxis().set_visible(False)
                axis[i].axes.get_yaxis().set_visible(False)
    # fill in the code here
    '''
    This function needs to extract the band identifiers from the metadata
    and then create the RGB composite. For the VV-VH composite, after
    the subtraction, you need to minmax scale the image.
    '''
    plt.tight_layout()
    if image_dir is None:
        plt.show()
    else:
        #plt.show()
        plt.savefig(Path(image_dir) / "plot_sentinel1.png")
        plt.close()
    return
    raise NotImplementedError



def validate_band_identifiers(
          bands_to_plot: List[List[str]],
          band_mapping: dict) -> None:
    """
    This function validates the band identifiers.

    Parameters
    ----------
    bands_to_plot : List[List[str]]
        The bands to plot.
    band_mapping : dict
        The band mapping.

    Returns
    -------
    None
    """
    '''
    band_mapping = {band_id: idx for
                    idx, band_id in enumerate(unique_band_ids)}
    validate_band_identifiers(bands_to_plot, band_mapping)
    bands_to_plot = [['04', '03', '02'], ['08', '04', '03'], ['12', '8A', '04']]'''
    for i in range(len(bands_to_plot)):
        for j in range(len(bands_to_plot[i])):
            if bands_to_plot[i][j] not in band_mapping:
                raise ValueError
    #print(bands_to_plot)
    #print(band_mapping)
    return
    raise NotImplementedError


def plot_images(
        processed_stack: np.ndarray,
        bands_to_plot: List[List[str]],
        band_mapping: dict,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None
        ):
    """
    This function plots the satellite images.

    Parameters
    ----------
    processed_stack : np.ndarray
        The satellite image stack volume.
    bands_to_plot : List[List[str]]
        The bands to plot.
    band_mapping : dict
        The band mapping.
    metadata : List[List[Metadata]]
        The metadata for the satellite image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """

    #print(processed_stack.shape)
    #print(bands_to_plot, '\n\n')
    #print(band_mapping, '\n\n')
    #print(metadata, '\n\n')
    #print(len(metadata))
    # fill in the code here
    #So for each time just print the bands that are  requested
    #use the bandmapping to choose the array in the processed stack
    fig, axis = plt.subplots(len(processed_stack), len(bands_to_plot), figsize = (10,10))

    for i in range(len(processed_stack)):
        for j in range(len(bands_to_plot)):
            coords = []
            for k in range(len(bands_to_plot[j])):
                band = band_mapping[bands_to_plot[j][k]]
                coords.append(processed_stack[i][band])
            coords = np.stack(coords)
            axis[i][j].imshow(coords.T)
            axis[i][j].axes.get_yaxis().set_visible(False)
            axis[i][j].axes.get_xaxis().set_visible(False)
            time = metadata[i].time + "\n"
            if(i == 0):
                axis[i][j].set_title(time+",".join(bands_to_plot[j]), )
            else:
                axis[i][j].set_title(time)
        #axis[i].set_title(metadata[i].time)
    plt.tight_layout()
    if image_dir is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(
            Path(image_dir) / f"plot_{metadata[0].satellite_type}.png"
            )
        plt.close()
    return
    raise NotImplementedError


def plot_satellite_by_bands(
        satellite_stack: np.ndarray,
        metadata: List[Metadata],
        bands_to_plot: List[List[str]],
        satellite_type: str,
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function plots the satellite image by band in subplots.

    Parameters
    ----------
    satellite_stack : np.ndarray
        The satellite image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the satellite image stack.
    bands_to_plot : List[List[str]]
        The bands to plot.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    None
    """
    #print("Starting out", satellite_stack[0][0])
    processed_stack = preprocess_data(satellite_stack, satellite_type)
    #print("Completely done", processed_stack[0][0])

    if satellite_type == "sentinel1":
        create_rgb_composite_s1(processed_stack, bands_to_plot, metadata, image_dir=image_dir)
    else:
        band_ids_per_timestamp = extract_band_ids(metadata)
        all_band_ids = [band_id for timestamp in band_ids_per_timestamp for
                        band_id in timestamp]
        unique_band_ids = sorted(list(set(all_band_ids)))
        band_mapping = {band_id: idx for
                        idx, band_id in enumerate(unique_band_ids)}
        validate_band_identifiers(bands_to_plot, band_mapping)
        plot_images(
            processed_stack,
            bands_to_plot,
            band_mapping,
            metadata,
            image_dir
            )


def extract_band_ids(metadata: List[Metadata]) -> List[List[str]]:
    """
    Extract the band identifiers from file names for each timestamp based on
    satellite type.

    Parameters
    ----------
    file_names : List[Metadata]
        A list of file names.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    List[List[str]]
        A list of band identifiers.
    """
    band_identifiers = []
    for meta in metadata:
        band_identifiers.append(meta.bands)
    return band_identifiers
    raise NotImplementedError


def plot_ground_truth(
        ground_truth: np.array,
        plot_title: str = '',
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function plots the groundTruth image.

    Parameters
    ----------
    tile_dir : str
        The directory containing the VIIRS tiles.
    """
    # fill in the code here

    figures, axis = plt.subplots(1, len(ground_truth))
    if(len(ground_truth) == 4):
        for i in range(len(ground_truth)):
            axis[i].imshow(ground_truth[i][0])
    else:
        plt.imshow(ground_truth[0][0])
        plt.title(plot_title)
    if image_dir is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(Path(image_dir) / "ground_truth.png")
        plt.close()
    return
    raise NotImplementedError
