import numpy as np
import warnings
from PIL import Image
from .hp import MAX_TEXT_LENGTH, BYTE

def text_router(image: Image.Image, 
                key: int=None, 
                text: str=None, 
                nnoise: int=1):
    """Just a simple encapsulation, logic is described in the core file."""
    if text:
        if len(text) > MAX_TEXT_LENGTH:
            raise Exception("Error: Text is too long.")
        return _encode_image_with_text(image, text=text, key=key, nnoise=nnoise)
    return _decode_image(image, key=key)


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
    ord_text = ord_text + [32 for _ in range(MAX_TEXT_LENGTH-len(ord_text))]
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
    if MAX_TEXT_LENGTH*BYTE > data.shape[0]:
        raise Exception("Error: Image size is to small for the text.")
    idxs = np.random.choice(data.shape[0], MAX_TEXT_LENGTH*BYTE, replace=False)
    # converting the text to 1's and 0's for storing
    hidden = _text_to_binary(text)
    # out of all the indexes, it only changes the ones that dont
    # match already, by subtracting 1 (changes the leftmost bit)
    data[idxs[hidden != data[idxs] % 2]] -= 1

    # if someone else has the original image, part of the text could be
    # uncovered, so now will add random noise(salt?) to some other indexes
    # nnoise is a variable that controls the number of noise is added
    # more nnoise is more secure but also means more changes to the original
    if nnoise*MAX_TEXT_LENGTH*BYTE > data.shape[0]:
        nnoise = data.shape[0]//(MAX_TEXT_LENGTH*BYTE)
        warnings.warn("Warning: Lowering the nnoise.")

    noise_idxs = np.random.choice(
        data.shape[0], nnoise*MAX_TEXT_LENGTH*BYTE, replace=False)
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
    idxs = np.random.choice(data.shape[0], MAX_TEXT_LENGTH*BYTE, replace=False)
    # takes the leftmost bits from the indexes
    s = "".join(map(str, data[idxs] % 2))
    # splits the bits into groups of bytes
    s = [s[i:i+BYTE] for i in range(0, len(s), BYTE)]
    # first, converts each of the bytes to ascii code then to chars
    # and gets the encoded text within the image
    s = "".join([chr(int(bin_, base=2)) for bin_ in s])
    # removes paddings and returns the text
    return s.strip()
