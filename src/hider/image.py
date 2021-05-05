import numpy as np
from .hp import IMAGE_SIZE
from PIL import Image
from typing import List


def image_router(*images, key: int=None):
    """Just a simple encapsulation, logic is described in the core file."""
    if key:
        if len(images) == 1:
            return _decode_image_with_key(images[0], key)
        return _encode_images_with_key(images, key)
    if len(images) == 1:
        return _decode_image_no_key(images[0])
    return _encode_images_no_key(images)
    

def _encode_images_with_key(org_images: List[np.ndarray], key: int) -> Image.Image:
    """Takes a list of images and combines them. Every image 
    except the first one is hid into the first image.

    The hiding is done by storing the 4 most significant bits of 
    the hidden images in the randomly chosen places of first 
    image's least significant bits.
    
    Parameters
    ----------
    images : list
        A list of images to be combined. First is the "cover", the
        rest is hidden.
    key : int
        The key is needed for decoding the image later.

    Returns
    -------
    PIL.Image.Image
        A cover image with hidden images in it.
    """
    # the function works with the flattened data
    images = [im.reshape(np.prod(im.shape)).copy() for im in org_images]
    # the cover image
    data = images[0]
    # generating the indexes using key
    np.random.seed(key)
    final_idxs = []
    full_idxs = []
    for _ in range(len(images)-1):
        availables = np.setdiff1d(np.arange(data.shape[0]), full_idxs)
        idxs = np.random.choice(availables, np.prod((*IMAGE_SIZE, 3)), replace=False)
        full_idxs = [*full_idxs, *idxs]
        final_idxs.append(idxs)
    # making the magic happen using bit shifting operators
    for i in range(len(images)-1):
        data[final_idxs[i]] = (data[final_idxs[i]] >> 4 << 4) + (images[i+1] >> 4)
    return Image.fromarray(data.reshape(org_images[0].shape))


def _decode_image_with_key(image: np.ndarray, key: int) -> List[Image.Image]:
    """Takes an image with hidden images in it and returns 
    a list of hidden images.
    
    Parameters
    ----------
    images : list
        A list of images to be combined. First is the "cover", the
        rest is hidden.
    key : int
        The key used when encoding the data.

    Returns
    -------
    list[PIL.Image.Image]
        List of extracted images.
    """
    # determining the max number of hiddens
    num_hidden = np.prod(image.shape) // np.prod((*IMAGE_SIZE, 3))
    # the function works with the flattened data
    data = image.reshape(np.prod(image.shape)).copy()
    # generating the indexes using key
    np.random.seed(key)
    final_idxs = []
    full_idxs = []
    for _ in range(num_hidden):
        availables = np.setdiff1d(np.arange(data.shape[0]), full_idxs)
        idxs = np.random.choice(availables, np.prod((*IMAGE_SIZE, 3)), replace=False)
        full_idxs = [*full_idxs, *idxs]
        final_idxs.append(idxs)
    # getting hidden images from the right indexes using bitshifting
    images = []
    for i in range(num_hidden):
        images.append(Image.fromarray((data[final_idxs[i]] << 4).reshape((*IMAGE_SIZE, 3))))
    return images


def _encode_images_no_key(images: List[np.ndarray]) -> np.ndarray:
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


def _decode_image_no_key(combined_image: np.ndarray,
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
    list[PIL.Image.Image]
        a list of extracted images.
    """
    if num_hidden == 4:
        return [Image.fromarray(combined_image >> i << 6) for i in range(6, -1, -2)]
    if num_hidden == 3:
        return [Image.fromarray(im) for im in [(combined_image >> 5 << 5),
                                               (combined_image >> 2 << 5),
                                               (combined_image << 6)]]
    if num_hidden == 2:
        return [Image.fromarray(combined_image >> 4 << 4), Image.fromarray(combined_image << 4)]
    raise Exception("Range of the number of images is [2, 4].")