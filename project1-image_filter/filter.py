import multiprocessing as mp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cProfile
import functions as my

# sets the number of cores
NUMCORES = 4

# opens image
image = np.array(Image.open('lena.png'))

# sets the kernel
image_filter = np.ones((3, 3))

# initializes image
my.init_globalimage(image,image_filter)

print(f'Image shape: {image.shape}')

# display original image
plt.figure(figsize = (4, 3),dpi = 100)
plt.imshow(image)
plt.title('original')
plt.show()

def filter_image(my_image):
    shape = my_image.shape
    rows = shape[0]
    v = range(rows)
    with mp.Pool(NUMCORES, initializer = my.init_globalimage, initargs=[image, image_filter]) as p:
        result = p.map(my.filter_image, v)
    return result

filtered_image = filter_image(image)    

def reconstruct_image(shape,image):
    new_image = np.ndarray(shape,dtype=uint8)
    for r in range(len(image)):
        new_image[r] = image[r]
    return new_image


fimage = reconstruct_image(image.shape,filtered_image)

# display original image
plt.figure(figsize=(4,3),dpi=200)
plt.imshow(fimage)    
plt.title('filtered')
plt.show()