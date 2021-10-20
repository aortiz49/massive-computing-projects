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

print(image.shape)
