from PIL import Image
import numpy as np
import requests
from typing import List


def _from_url(url: str) -> Image.Image:
    """Takes an image url, scrapes the image, and returns it as a numpy array."""
    try:
        return Image.open(requests.get(url, stream=True).raw)
    except:
        raise Exception("Error: Please provide a valid image URL.")


def _handle_images(images: list) -> List[np.ndarray]:
    """Given a list of elements(images or image URLs) or both mixed, 
    converts them to numpy arrays, returns images of the same size.
    
    Parameters
    ----------
    images : list
        A list of mixed types. Supported types:
            -PIL.Image.Image
            -numpy.ndarray
            -str (an image url)

    Returns
    -------
    list[numpy.ndarray]
        a list of images in the form of numpy arrays.
    """
    fixed = []
    mapping = {Image.Image: lambda x: x, np.ndarray: Image.fromarray, str: _from_url}
    for im in images:
        if type(im) in mapping.keys():
            fixed.append(mapping[type(im)](im))
        else:
            raise TypeError("Error: Unknown image type.")
    # finding the maximum possible size possible to represent all images.
    sizes = [im.size for im in fixed]
    final_size = tuple(map(min, list(zip(*sizes))))
    print(f"All images have been downscaled to {final_size}.")
    return [np.array(im.resize(final_size)) for im in fixed]


def _combine_images(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images and combines them. Every image 
    is combined into an image that looks like the first image.
    
    The combination is done by combining as much significant 
        bits as possible per image.
        
    If two images -> combines their 4 most significant digits.
    If three images -> combines 3 most from the first two images
        and 2 most from the last image.
    If four images -> combines their 2 most significant digits.
    
    Parameters
    ----------
    images : list
        A list of images to be combined.

    Returns
    -------
    list[numpy.ndarray]
        a list of images in the form of numpy arrays.
    """
    if len(images) == 4:
        return sum([image >> 6 << i for (image, i) in zip(images, range(6, -1, -2))])
    if len(images) == 3:
        return (images[0] >> 5 << 5) + (images[1] >> 5 << 2) + (images[2] >> 6)
    if len(images) == 2:
        return (images[0] >> 4 << 4) + (images[1] >> 4)
    raise Exception("Error: Number of images more than 4 is not supported.")


def _get_hidden_images(combined_image: np.ndarray, 
                       num_hidden: int) -> List[Image.Image]:
    """Extracts and returns the `num_hidden` images from the given image.
    
    Parameters
    ----------
    combined_image : numpy.ndarray
        An image represented by a numpy array.
    num_hidden : int
        Total number of images to be extracted.

    Returns
    -------
    PIL.Image.Image
        a list of extracted images.
    """
    if num_hidden == 4:
        return [Image.fromarray(combined_image>>i<<6) for i in range(6, -1, -2)]
    if num_hidden == 3:
        return [Image.fromarray(im) for im in [(combined_image >> 5 << 5), 
                                               (combined_image >> 2 << 5), 
                                               (combined_image << 6)]]
    if num_hidden == 2:
        return [Image.fromarray(combined_image>>4<<4), Image.fromarray(combined_image<<4)]
    raise Exception("Range of the number of images is [2, 4].")