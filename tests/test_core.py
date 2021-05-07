import numpy as np
import unittest
from src.hider.core import image_in_image, text_in_image
from src.hider.hp import BYTE

# End to end tests for the main functions

class CoreTest(unittest.TestCase):
    
    def test_image_in_image(self):
        images = [np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)]
        for _ in range(5):
            row = np.random.randint(1, 800)
            col = np.random.randint(1, 800)
            images.append(np.random.randint(0, 255, (row, col, 3), dtype=np.uint8))
        
        # tests if only the last [ratio] bits are changed
        for i in [2,3,4,5,6,7]:
            encoded_image = image_in_image(*images, key=1, ratio=i)
            rt = BYTE-i
            original = images[0]>>rt<<rt
            modified = np.array(encoded_image)>>rt<<rt
            self.assertEqual((modified!=original).sum(), 0)
            
            
    def test_text_in_image(self):
        image = np.random.randint(0, 255, (400,400,3), dtype=np.uint8)
        text = "asdasdasdasdasdasd"
        encoded_image = text_in_image(image, text=text, key=1)
        # tests if only the last bit is changed
        self.assertEqual(((np.array(encoded_image)>>1<<1) != (image>>1<<1)).sum(), 0)
        extracted_text = text_in_image(encoded_image, key=1)
        
        # tests if text is extracted without loss
        self.assertEqual(text, extracted_text)
