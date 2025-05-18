"""
This module contains functions for loading satellite data from a directory of
tiles.
"""
from pathlib import Path
from typing import Tuple, List, Set
import os
from itertools import groupby
import re
from dataclasses import dataclass
import tifffile
import numpy as np


@dataclass
class Metadata:
    """
    A class to store metadata about a satellite file.
    """
    satellite_type: str #may be optional
    file_name: List[str]
    tile_id: str
    bands: List[str]
    time: str


def process_viirs_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a VIIRS file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: DNB_VNP46A1_A2020221.tif
    Example output: ("2020221", "0")

    Parameters
    ----------
    filename : str
        The filename of the VIIRS file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    filename = filename.split("\\")[-1]
    split_file = filename.split("_")
    if(len(split_file) != 3):
        raise SyntaxError
    if(not split_file[2][0].isalpha()):
        raise FileExistsError
    date = split_file[2][1:].split(".")[0]
    return (date, "0")

    raise NotImplementedError


def process_s1_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Sentinel-1 file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: S1A_IW_GRDH_20200804_VV.tif
    Example output: ("20200804", "VV")

    Parameters
    ----------
    filename : str
        The filename of the Sentinel-1 file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    filename = filename.split("\\")[-1]
    split_file = filename.split("_")
    if(len(split_file) != 5):
        raise FileExistsError
    date = split_file[3]
    band = split_file[4].split(".")[0]
    return (date, band)

    raise NotImplementedError


def process_s2_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Sentinel-2 file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: L2A_20200816_B01.tif
    Example output: ("20200804", "B01")

    Parameters
    ----------
    filename : str
        The filename of the Sentinel-2 file.

    Returns
    -------
    Tuple[str, str]
    """
    filename = filename.split("\\")[-1]
    split_file = filename.split("_")
    if(len(split_file) != 3):
        raise FileExistsError
    date = split_file[1]
    band = split_file[2].split(".")[0][1:]
    return (date, band)

    raise NotImplementedError


def process_landsat_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Landsat file and outputs
    a tuple containing two strings, in the format (date, band)

    Example input: LC08_L1TP_2020-08-30_B9.tif
    Example output: ("2020-08-30", "B9")

    Parameters
    ----------
    filename : str
        The filename of the Landsat file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    filename = filename.split("\\")[-1]
    split_file = filename.split("_")
    if(len(split_file) != 4):
        raise FileExistsError
    date = split_file[2]
    band = split_file[3].split(".")[0][1:]
    return (date, band)
    raise NotImplementedError


def process_ground_truth_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of the ground truth file and returns
    ("0", "0"), as there is only one ground truth file.

    Example input: groundTruth.tif
    Example output: ("0", "0")

    Parameters
    ----------
    filename: str
        The filename of the ground truth file though we will ignore it.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    return ("0", "0")


def get_satellite_files(tile_dir: Path, satellite_type: str) -> List[Path]:
    """
    Retrieve all satellite files matching the satellite type pattern.

    Parameters
    ----------
    tile_dir : Path
        The directory containing the satellite tiles.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    List[Path]
        A list of Path objects for each satellite file.
    """
    #str(p)
    #p.iterdir()
    sat_types = {"viirs": "DNB", "sentinel1": "S1A", "sentinel2": "L2A", "landsat": "LC08", "gt": "groundTruth"}
    type = sat_types[satellite_type]
    files = [x for x in tile_dir.iterdir() if x.is_file()]
    #files = [str(x.name) for x in tile_dir.iterdir() if x.is_file()]
    validfiles = []
    for file in files:
        str_file = str(file.name)
        if(type == str_file[0:len(type)] and str_file[-3:] == 'tif'):
            validfiles.append(file)
    #print(files)
    return validfiles
    #raise NotImplementedError


def get_filename_pattern(satellite_type: str) -> str:
    """
    Return the filename pattern for the given satellite type.

    Parameters
    ----------
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    str
        The filename pattern for the given satellite type.
    """
    patterns = {
        "viirs": 'DNB_VNP46A1_*',
        "sentinel1": 'S1A_IW_GRDH_*',
        "sentinel2": 'L2A_*',
        "landsat": 'LC08_L1TP_*',
        "gt": "groundTruth.tif"
    }
    return patterns[satellite_type]
    raise NotImplementedError


def read_satellite_files(sat_files: List[Path]) -> List[np.ndarray]:
    """
    Read satellite files into a list of numpy arrays.

    Parameters
    ----------
    sat_files : List[Path]
        A list of Path objects for each satellite file.

    Returns
    -------
    List[np.ndarray]

    """
    #files = [x for x in tile_dir.iterdir() if x.is_file()]
    nplist = []
    for path in sat_files:
        if(not path.is_file()):
            raise FileNotFoundError
        else:
            img = np.array(tifffile.imread(str(path)), dtype = np.float32)
            nplist.append(img)
    #print(type(nplist[0]))
    return nplist
    raise NotImplementedError


def stack_satellite_data(
        sat_data: List[np.ndarray],
        file_names: List[Path],
        satellite_type: str
        ) -> Tuple[np.ndarray, List[Metadata]]:
    """
    Stack satellite data into a single array and collect filenames.

    Parameters
    ----------
    sat_data : List[np.ndarray]
        A list contianing data for all bands with respect ot single satellite
    file_names : List[str]
        List of filenames corresponding to the satelite and timestamp

    Returns
    -------
    Tuple[np.ndarray, List[Metadata]
        A tuple containing the satellite data as a volume with dimensions
        (time, band, height, width) and a list of the filenames.
    """
    sat_types_dic = {"viirs": process_viirs_filename, "sentinel1": process_s1_filename, "sentinel2": process_s2_filename,
                     "landsat": process_landsat_filename, "gt": process_ground_truth_filename}
    file_pattern = get_filename_pattern(satellite_type)[:-1]
    parse_function = sat_types_dic[satellite_type]
    dates = []

    # Get the function to group the data based on the satellite type
    for i, file in enumerate(file_names):
        if(isinstance(file, str)):
            file = Path(file)
            file_names[i] = file
        str_file = str(file.name)
        if(file_pattern != str_file[0:len(file_pattern)]): #First confirm that the list of sat_data  actually is from the same satelite
            continue
        dates.append(parse_function(str_file)) #then take out all the dates and stuff and add it to a list
    #then sort the files based on the sorted dtaes and bands and stuff
    #(list(zip(sat_data, dates)))
    sat_data = [_ for _, x in sorted(zip(sat_data, dates), key = lambda x: (x[1][0], x[1][1]))]
    file_names = [_ for _, x in sorted(zip(file_names, dates), key = lambda x: (x[1][0], x[1][1]))]
    dates = sorted(dates, key=lambda x: (x[0], x[1]))

    stacked_data = []
    metadatas = []
    '''
    Metadata(satellite_type, )
    satellite_type: str #may be optional
    file_name: List[str]
    tile_id: str
    bands: List[str]
    time: str'''

    temp_filenames = []
    temp_bands = []
    temp_time = ""
    temp_tile = ""
    temp_stacked = []

    for i, date in enumerate(dates):
        if (temp_time != date[0] and temp_time != ""):
            metadatas.append(Metadata(satellite_type, temp_filenames, temp_tile, temp_bands, temp_time))
            temp_filenames = []
            temp_bands = []
            stacked_data.append(np.stack(temp_stacked))
            temp_stacked =[]
        temp_time = date[0]
        temp_bands.append(date[1])
        temp_tile = str(file_names[i].parent.name)
        temp_filenames.append(str(file_names[i].name))
        temp_stacked.append(sat_data[i])
    if(len(temp_bands) != 0):
        metadatas.append(Metadata(satellite_type, temp_filenames, temp_tile, temp_bands, temp_time))
        stacked_data.append(np.stack(temp_stacked))

    # Apply the grouping function to each file name to get the date and band

    # Sort the satellite data and file names based on the date and band

    # Initialize lists to store the stacked data and metadata

    # Group the data by date
        # Sort the group by band

        # Extract the date and band, satellite data, and file names from the
        # sorted group

        # Stack the satellite data along a new axis and append it to the list

        # Create a Metadata object and append it to the list

    # Stack the list of satellite data arrays along a new axis to create a
    # 4D array with dimensions (time, band, height, width)

    # Return the stacked satelliet data and the list Metadata objects.
    return (np.stack(stacked_data), metadatas)
    raise NotImplementedError


def get_grouping_function(satellite_type: str):
    """
    Return the function to group satellite files by date and band.

    Parameters
    ----------
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    function
        The function to group satellite files by date and band.
    """
    sat_types_dic = {"viirs": process_viirs_filename, "sentinel1": process_s1_filename, "sentinel2": process_s2_filename,
                     "landsat": process_landsat_filename, "gt": process_ground_truth_filename}
    return sat_types_dic[satellite_type]
    raise NotImplementedError


def get_unique_dates_and_bands(
        metadata_keys: Set[Tuple[str, str]]
        ) -> Tuple[Set[str], Set[str]]:
    """
    Extract unique dates and bands from satellite metadata keys.

    Parameters
    ----------
    metadata_keys : Set[Tuple[str, str]]
        A set of tuples containing the date and band for each satellite file.

    Returns
    -------
    Tuple[Set[str], Set[str]]
        A tuple containing the unique dates and bands.
    """
    dates = set()
    bands = set()
    for key in metadata_keys:
        dates.add(key[0])
        bands.add(key[1])
    #print(len(dates))
    #print(type(bands))
    return (dates, bands)
    raise NotImplementedError


def load_satellite(
        tile_dir: str | os.PathLike,
        satellite_type: str
        ) -> Tuple[np.ndarray, List[Metadata]]:
    """
    Load all bands for a given satellite type from a directory of tile files.

    Parameters
    ----------
    tile_dir : str or os.PathLike
        The Tile directory containing the satellite tiff files.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        A tuple containing the satellite data as a volume with
        dimensions (time, band, height, width) and a list of the filenames.
    """
    # first get a list of all files
    if(isinstance(tile_dir, str)):
        tile_dir = Path(tile_dir)
    all_files = [x for x in tile_dir.iterdir() if x.is_file()]
    wanted_files = []
    file_pattern = get_filename_pattern(satellite_type)[:-1]

    # then get the list of all files that match said satelite type
    for file in all_files:
        str_file = str(file.name)
        if(file_pattern != str_file[0:len(file_pattern)]):
            continue
        wanted_files.append(file)

    #retrive the data of all of these files
    data_arr = read_satellite_files(wanted_files)
    #print(type(data_arr[0]))
    #print(len(data_arr))
    #print(len(data_arr[0]))

    return stack_satellite_data(data_arr, wanted_files, satellite_type)

    #finally
    raise NotImplementedError


def load_satellite_dir(
        data_dir: str | os.PathLike,
        satellite_type: str
        ) -> Tuple[np.ndarray, List[Metadata]]:
    """
    Load all bands for a given satellite type from a directory of multiple
    tile files.

    Parameters
    ----------
    data_dir : str or os.PathLike
        The directory containing the satellite tiles.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    Tuple[np.ndarray, List[Metadata]]
        A tuple containing the satellite data as a volume with
        dimensions (tile_dir, time, band, height, width) and a list of the
        Metadata objects.
    """
    '''
    load_satellite(
        tile_dir: str | os.PathLike,
    satellite_type: str) returns a tuple of np array and a list of metadatas'''
    if(isinstance(data_dir, str)):
        data_dir = Path(data_dir)
    all_dirs = [x for x in data_dir.iterdir() if x.is_dir()]
    datas = []
    meta_lists = []
    for dir in all_dirs:
        data, meta = load_satellite(dir, satellite_type)
        datas.append(data)
        meta_lists.append(meta)
    return (np.array(datas), meta_lists)

    #raise NotImplementedError
