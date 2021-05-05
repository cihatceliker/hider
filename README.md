**Information Systems Security Project**

**Cihat Emre Çeliker, 20160808028**

***
## Hider
`"Steganography is the practice of concealing a message within another message or a physical object."`

Hider is a package that is capable of hiding text or images in an image using bit manipulation techniques and random number generators to make unbreakable encryptions.

***
## Method
There are 3 different encoding options:
- Image(s) in image with key:
    - Using key to generate random indexes, the program hides the 4 most significant bits of the to be hidden images, in the cover image's 4 least significant digits.
- Image(s) in image without key:
    - Depending on the number of images to be hidden, the program hides images in different parts the cover image's bits:
        - To hide 1 image, divides cover image's bits into 2 parts, 4, 4.
        - To hide 2 images, divides cover image's bits into 3 parts, 3, 3, 2.
        - To hide 3 images, divides cover image's bits into 4 parts, 2, 2, 2, 2.
- Text in image with key:
    - Using key to generate random indexes, the program hides the text in the least significant bits of the cover image.

When key is used, it's impossible to retrieve hidden data even with the original file.
***


your methodology and findings
A youtube link to your demo video