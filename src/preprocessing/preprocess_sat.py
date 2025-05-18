""" This code applies preprocessing functions on the IEEE GRSS ESD satellite data."""
import numpy as np
from scipy.ndimage import gaussian_filter


def per_band_gaussian_filter(img: np.ndarray, sigma: float = 1):
    """
    For each band in the image, apply a gaussian filter with the given sigma.

    Parameters
    ----------
    img : np.ndarray
        The image to be filtered.
    sigma : float
        The sigma of the gaussian filter.

    Returns
    -------
    np.ndarray
        The filtered image.
    """
    for i in range(len(img)): #time
        for j in range(len(img[i])): #band
            img[i][j] = gaussian_filter(img[i][j], sigma = sigma)
    return img
    raise NotImplementedError


def quantile_clip(img_stack: np.ndarray,
                  clip_quantile: float,
                  group_by_time=True
                  ) -> np.ndarray:
    """
    This function clips the outliers of the image stack by the given quantile.

    Parameters
    ----------
    img_stack : np.ndarray
        The image stack to be clipped.
    clip_quantile : float
        The quantile to clip the outliers by.

    Returns
    -------
    np.ndarray
        The clipped image stack.
    """
    if(group_by_time): #calculate the quantile across all the times
        #np.clip(array, min, max, none)
        for i in range(len(img_stack)): #time
            min_quantile = np.quantile(img_stack[i], clip_quantile)
            max_quantile = np.quantile(img_stack[i], 1-clip_quantile)
            for j in range(len(img_stack[i])):
                img_stack[i][j]= np.clip(img_stack[i][j], min_quantile, max_quantile)
    else: #calculate quantile limits for each image
        for i in range(len(img_stack)): #time
            for j in range(len(img_stack[i])):
                min_quantile = np.quantile(img_stack[i][j], clip_quantile)
                max_quantile = np.quantile(img_stack[i][j], 1 - clip_quantile)
                img_stack[i][j]= np.clip(img_stack[i][j], min_quantile, max_quantile)
    return img_stack

    raise NotImplementedError


def minmax_scale(img: np.ndarray, group_by_time=True):
    """
    This function minmax scales the image stack.

    Parameters
    ----------
    img : np.ndarray
        The image stack to be minmax scaled.
    group_by_time : bool
        Whether to group by time or not.

    Returns
    -------
    np.ndarray
        The minmax scaled image stack.
    """
    def normalized_pixel_value(pixel, min, max):
        return (pixel-min) / (max-min)
    vectorized_function = np.vectorize(normalized_pixel_value)
    if(group_by_time): #will be shared amongs the time dimension
        for i in range(len(img)):  # time
            min = np.min(img[i])
            max = np.max(img[i])
           # print(max, min)
            for j in range(len(img[i])):
                img[i][j] = vectorized_function(img[i][j], min, max)
                #print(img[i][j], "\n\n")
    else:  # calculate quantile limits for each image
        for i in range(len(img)):  # time
            for j in range(len(img[i])):
                min = np.min(img[i][j])
                max = np.max(img[i][j])
                img[i][j] = vectorized_function(img[i][j], min, max)
        #np.vectorize(normalixed_pixel_value, min, max)
    #print(img[0][0])
    return img
    raise NotImplementedError



def brighten(img, alpha=0.13, beta=0):
    """
    Function to brighten the image.

    Parameters
    ----------
    img : np.ndarray
        The image to be brightened.
    alpha : float
        The alpha parameter of the brightening.
    beta : float
        The beta parameter of the brightening.

    Returns
    -------
    np.ndarray
        The brightened image.
    """
    def brightenpixel(pixel, alpha, beta):
        newpixel = alpha*pixel + beta
        if newpixel > 1:
            return 1
        elif newpixel <0:
            return 0
        return newpixel

    vectorized_function = np.vectorize(brightenpixel)
    for i in range(len(img)):  # time
        for j in range(len(img[i])):
            img[i][j] = vectorized_function(img[i][j], alpha, beta)
    return img
    raise NotImplementedError


def gammacorr(band, gamma=2):
    """
    This function applies a gamma correction to the image.

    Parameters
    ----------
    band : np.ndarray
        The image to be gamma corrected.
    gamma : float
        The gamma parameter of the gamma correction.

    Returns
    -------
    np.ndarray
        The gamma corrected image.
    """
    def gamma_cor(pixel, gamma):
        newpixel = pow(pixel, 1/gamma)
        return newpixel
    vectorized_function = np.vectorize(gamma_cor)
    for i in range(len(band)):  # time
        for j in range(len(band[i])):
            band[i][j] = vectorized_function(band[i][j], gamma)
    return band
    raise NotImplementedError


def maxprojection_viirs(
        viirs_stack: np.ndarray,
        clip_quantile: float = 0.01
        ) -> np.ndarray:
    """
    This function takes a directory of VIIRS tiles and returns a single
    image that is the max projection of the tiles.

    Parameters
    ----------
    tile_dir : str
        The directory containing the VIIRS tiles.

    Returns
    -------
    np.ndarray
    """
    # HINT: use the time dimension to perform the max projection over.
    #np.max(viirs_stack, axis = 3)
    #axes = 3?
    viirs_stack = quantile_clip(viirs_stack, clip_quantile)
    viirs_stack = np.max(viirs_stack, axis = 0)
    def normalized_pixel_value(pixel, min, max):
        return (pixel-min) / (max-min)

    min = np.min(viirs_stack)
    max = np.max(viirs_stack)
    vectorized_function = np.vectorize(normalized_pixel_value)
    for i in range(len(viirs_stack)):
        viirs_stack[i] = vectorized_function(viirs_stack[i], min, max)
    #viirs_stack = minmax_scale(viirs_stack)
    return viirs_stack
    raise NotImplementedError


def preprocess_sentinel1(
        sentinel1_stack: np.ndarray,
        clip_quantile: float = 0.01,
        sigma=1
        ) -> np.ndarray:
    """
    In this function we will preprocess sentinel1. The steps for preprocessing
    are the following:
        - Convert data to dB (log scale)
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gaussian filter
        - Minmax scale
        
    """
    #print(sentinel1_stack[0][0], "\n\n")
    def lin_to_db(pixel):
        return 10*np.log10(pixel)
    vectorized_function = np.vectorize(lin_to_db)
    for i in range(len(sentinel1_stack)):  # time
        for j in range(len(sentinel1_stack[i])):
            sentinel1_stack[i][j] = vectorized_function(sentinel1_stack[i][j])
    #print(sentinel1_stack[0][0], "\n\n")
    sentinel1_stack = quantile_clip(sentinel1_stack, clip_quantile)
    sentinel1_stack = per_band_gaussian_filter(sentinel1_stack, sigma)
    #print(sentinel1_stack[0][0], "\n\n")
    sentinel1_stack = minmax_scale(sentinel1_stack)
    #print(sentinel1_stack[0][0])
    return sentinel1_stack

    raise NotImplementedError


def preprocess_sentinel2(sentinel2_stack: np.ndarray,
                         clip_quantile: float = 0.05,
                         gamma: float = 2.2
                         ) -> np.ndarray:
    """
    In this function we will preprocess sentinel-2. The steps for
    preprocessing are the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gamma correction
        - Minmax scale
    """
    #print("Before Clipping", sentinel2_stack[0][0])
    sentinel2_stack = quantile_clip(sentinel2_stack, clip_quantile)
    #print("Before gamma", sentinel2_stack[0][0])
    sentinel2_stack = gammacorr(sentinel2_stack, gamma)
    #print("Before minmax", sentinel2_stack[0][0])
    sentinel2_stack = minmax_scale(sentinel2_stack)
    #print("Finish process", sentinel2_stack[0][0])
    return sentinel2_stack

def preprocess_landsat(
        landsat_stack: np.ndarray,
        clip_quantile: float = 0.05,
        gamma: float = 2.2
        ) -> np.ndarray:
    """
    In this function we will preprocess landsat. The steps for preprocessing
    are the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gamma correction
        - Minmax scale
    """
    landsat_stack = quantile_clip(landsat_stack, clip_quantile)
    landsat_stack = gammacorr(landsat_stack, gamma)
    landsat_stack = minmax_scale(landsat_stack)
    return landsat_stack

def preprocess_viirs(viirs_stack, clip_quantile=0.05) -> np.ndarray:
    """
    In this function we will preprocess viirs. The steps for preprocessing are
    the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Minmax scale
    """
    viirs_stack = quantile_clip(viirs_stack, clip_quantile)
    viirs_stack = minmax_scale(viirs_stack)
    return viirs_stack
