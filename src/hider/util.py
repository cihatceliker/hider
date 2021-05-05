import requests
import numpy as np
from PIL import Image
from typing import List
from .hp import IMAGE_SIZE


def _from_url(url: str) -> Image.Image:
    """Takes an image url, scrapes the image, and returns it as a numpy array."""
    try:
        return Image.open(requests.get(url, stream=True).raw)
    except:
        raise Exception("Error: Please provide a valid image URL.")


def handle_images(images: list) -> List[np.ndarray]:
    """Given a list of elements(images or image URLs) or both mixed, 
    converts them to numpy arrays and returns. All images are resized 
    except the original(the first) one.

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
    mapping = {Image.Image: lambda x: x,
               np.ndarray: Image.fromarray, str: _from_url}
    for im in images:
        if type(im) in mapping.keys():
            fixed.append(mapping[type(im)](im))
        elif Image.isImageType(im):
            fixed.append(Image.fromarray(np.array(im)))
        else:
            print(type(im))
            raise TypeError("Error: Unknown image type.")
    # resizing the hidden images to the predetermined size
    fixed[1:] = [im.resize(IMAGE_SIZE) for im in fixed[1:]]
    fixed = [np.array(im) for im in fixed]
    # now it will check for the volumes if images will be combined
    if len(fixed) > 1:
        volumes = [np.prod(im.shape) for im in fixed]
        if volumes[0] < sum(volumes[1:]):
            raise Exception("""Error: Not enough space to put images. 
                Either decrease the number of images or decrease the 
                hyperparameter `IMAGE_SIZE`""")
    return fixed
