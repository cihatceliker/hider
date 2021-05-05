import numpy as np
import requests
import warnings
import PIL
from PIL import Image
from typing import List


MAX_LENGTH = 100
BYTE = 8


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
        elif Image.isImageType(im):
            fixed.append(Image.fromarray(np.array(im)))
        else:
            print(type(im))
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
    list[PIL.Image.Image]
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


def _text_to_binary(text: str) -> np.ndarray:
    """Takes a string and converts it to a list of 1's and 0's.
    Any character with ASCII code bigger than 2^8 will be neglected.
    """
    # chars to ascii
    ord_text = [ord(ch) for ch in text]
    if max(ord_text) > 255:
        warnings.warn("""
            Warning: Chars with ascii codes higher than 
                     255 is not supported and will be lost""")
        ord_text = [min(ord_, 255) for ord_ in ord_text]
    # padding the text with whitespace until it reaches the max length 
    ord_text = ord_text + [32 for _ in range(MAX_LENGTH-len(ord_text))]
    # converting ascii codes to their byte represantation
    ord_text = [("0"*BYTE+"{0:b}".format(num))[-BYTE:] for num in ord_text]
    # converting it to 1's and 0's
    return np.array(list(map(int, "".join(ord_text))))


def _encode_image_with_text(org_data: np.ndarray, 
                            text: str, 
                            key: int, 
                            nnoise: int) -> Image.Image:
    """Encodes a text in an image.
    
    Parameters
    ----------
    org_data : numpy.ndarray
        The image that will keep the text in it.
    text : str
        The piece of text that will be encoded.
    key : int
        A key that will be used for RNG. It's will be needed 
        when extracting the text.
    nnoise : int
        For further protection from the people with the original image,
        "salt" is added. This is a coefficent to determine the size of the salt.
        salt size = nnoise * size of the encoded data.
        
    Returns
    -------
    PIL.Image.Image
        An image that has text encoded in it.
    """
    # the function works with the flattened data
    data = org_data.reshape(np.prod(org_data.shape)).copy()
    # the key is used for generating the random indexes in the flattened data
    np.random.seed(key)
    if MAX_LENGTH*BYTE > data.shape[0]:
        raise Exception("Error: Image size is to small for the text.")
    idxs = np.random.choice(data.shape[0], MAX_LENGTH*BYTE, replace=False)
    # converting the text to 1's and 0's for storing
    hidden = _text_to_binary(text)
    # out of all the indexes, it only changes the ones that dont 
    # match already, by subtracting 1 (changes the leftmost bit)
    data[idxs[hidden != data[idxs]%2]] -= 1
    
    # if someone else has the original image, part of the text could be
    # uncovered, so now will add random noise(salt?) to some other indexes
    # nnoise is a variable that controls the number of noise is added
    # more nnoise is more secure but also means more changes to the original
    if nnoise*MAX_LENGTH*BYTE > data.shape[0]:
        nnoise = data.shape[0]//(MAX_LENGTH*BYTE)
        warnings.warn("Warning: Lowering the nnoise.")
        
    noise_idxs = np.random.choice(data.shape[0], nnoise*MAX_LENGTH*BYTE, replace=False)
    # filtering the ones thats in the hidden text's indexes
    noise_idxs = np.setdiff1d(noise_idxs, idxs)
    # adding the noise if any space available
    if len(noise_idxs) > 0:
        data[noise_idxs] -= 1
    # return the reshaped flattened data with the original shape
    return Image.fromarray(data.reshape(org_data.shape))


def _decode_image(encoded_data: np.ndarray, key: int) -> str:
    """Extracts the text hidden in an image.
    
    Parameters
    ----------
    encoded_data : numpy.ndarray
        The image that has the text in it.
    key : int
        A key that is used for RNG. It's should be the
        same with the one used when encoding the text.
        
    Returns
    -------
    str
        The extracted text.
    """
    # flattenes the encoded image
    data = encoded_data.reshape(np.prod(encoded_data.shape)).copy()
    # sets the key and gets the indexes of the hidden text
    np.random.seed(key)
    idxs = np.random.choice(data.shape[0], MAX_LENGTH*BYTE, replace=False)
    # takes the leftmost bits from the indexes
    s = "".join(map(str, data[idxs]%2))
    # splits the bits into groups of bytes
    s = [s[i:i+BYTE] for i in range(0, len(s), BYTE)]
    # first, converts each of the bytes to ascii code then to chars
    # and gets the encoded text within the image
    s = "".join([chr(int(bin_, base=2)) for bin_ in s])
    # removes paddings and returns the text
    return s.strip()


def hider(*images, 
          key: int=None, 
          num_hidden: int=None, 
          text: str=None, 
          nnoise: int=1):
    """The main function of the package. Depending on the combination of inputs,
    can do the followings:
        - Encode text in an image
        - Decode a text from an image
        - Hide image(s) in an image
        - Extract hidden image(s) from an image
        
    If multiple images is passed:
        Hides every image except the first one, in the first image and returns it.
    If an image and a key is passed:
        If text is passed:
            Encodes the text inside the image using the key.
            This key is used in order to extract the text later.
        If no text:
            Extracts a text from the image using the key.
    If an image and `num_hidden` is passed:
        Extract `num_hidden` images from the given image and returns them.
        
    Parameters
    ----------
    images : list
        A list of mixed types. Supported types:
            -PIL.Image.Image
            -numpy.ndarray
            -str (an image url)
    key : int = None
        A key that will be used for RNG.
    num_hidden : int = None
        Total number of images to be extracted.
    text : str = None
        The piece of text that will be encoded.
    nnoise : int
        Used when encoding an image with text. 
        This is a coefficent to determine the size of the salt.
        salt size = nnoise * size of the encoded data.
    """
    if len(images) == 0:
        raise Exception("Error: No image passed.")
    images = _handle_images(images)
    if len(images) > 1:
        return _combine_images(images)
    # just to indicate
    assert len(images) == 1
    image = images[0]
    if not key and not num_hidden:
        raise Exception("Error: key or num_hidden is needed.")
    if key:
        if num_hidden:
            raise Exception("""Error: Passing both key and num_hidden is unnecessary. 
                                            What is your intention?""")
        if text:
            if len(text) > MAX_LENGTH:
                raise Exception("Error: Text is too long.")
            return _encode_image_with_text(image, text=text, key=key, nnoise=nnoise)
        else: # no text
            return _decode_image(image, key=key)
    # just to indicate
    assert num_hidden != None
    return _get_hidden_images(image, num_hidden)

