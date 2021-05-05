from PIL import Image
from .util import handle_images
from .image import image_router
from .text import text_router


def image_in_image(*images, key: int=None):
    """Hides image(s) in an image or extracts hidden image(s) from an image.
    
    If multiple images passed:
        Hides every image except the first one, in the first image and returns it.
    If only an image is passed:
        Extracts images from the given image and returns them.
        
    Parameters
    ----------
    images : list[np.ndarray]
        A list of images represented with numpy arrays.
    key : int = None
        A key that will be used for RNG.
    """
    if len(images) == 0:
        raise Exception("Error: No image passed.")
    images = handle_images(images)
    return image_router(*images, key=key)
    
    
def text_in_image(image: Image.Image, 
                  key: int=None, 
                  text: str=None, 
                  nnoise: int=1):
    """Hides text in an image or extracts hidden text from an image.
    
    If text is passed:
        Encodes the text inside the image using the key.
        The same key will be used to decode the data later.
    If no text:
        Extracts a text from the image using the key.
        
    Parameters
    ----------
    image : numpy.ndarray
        An image represented in a numpy array.
    key : int = None
        A key that will be used for RNG.
    text : str = None
        The piece of text that will be encoded.
    nnoise : int
        Used when encoding an image with text. 
        This is a coefficent to determine the size of the salt.
        salt size = nnoise * size of the encoded data.
    """
    image = handle_images([image])[0]
    if not key:
        raise Exception("Error: No key is passed.")
    return text_router(image=image, key=key, text=text, nnoise=nnoise)
