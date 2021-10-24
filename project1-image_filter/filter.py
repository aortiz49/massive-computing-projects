import multiprocessing as mp
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cProfile
import functions as my
import numpy as np
from multiprocessing.sharedctypes import Array
import ctypes

# import myfunctions as my


# sets the number of cores
NUMCORES = mp.cpu_count()

# opens images
image = np.array(Image.open('lena.png'))

# sets the kernels
filter1 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
filter2 = np.array([0.5, 0, -0.5])
filter3 = np.array([[0.5], [0], [0.5]])

filter4 = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])
filter5 = np.array([
    [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
    [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
    [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
    [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
    [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]
])

# initializes image

print(f'Image shape: {image.shape}')


def tonumpyarray(mp_arr):
    # mp_array is a shared memory array with lock

    return np.frombuffer(mp_arr.get_obj(), dtype=np.uint8)


# HERE YOU HAVE TO DEFINE THE MULTIPROCESSING VECTOR FOR IMAGE1
data_buffer1_size = image.shape[0] * image.shape[1] * image.shape[2]
shared_space1 = Array(ctypes.c_byte, data_buffer1_size)

# HERE YOU HAVE TO DEFINE THE MULTIPROCESSING VECTOR FOR IMAGE2
data_buffer2_size = image.shape[0] * image.shape[1] * image.shape[2]
shared_space2 = Array(ctypes.c_byte, data_buffer2_size)

NUMCORES = mp.cpu_count()

my.image_filter(image, filter3, NUMCORES, shared_space1)
#my.image_filter(image, filter2, NUMCORES, shared_space2)

print('hey')


def filters_execution(p_image, p_filter1, p_filter2, p_numprocessors, p_shared_space1,
                      p_shared_space2):
    # creates a lock to handle memory access
    lock = mp.Lock()

    # define and start the processes
    p1 = mp.Process(target=my.image_filter, args=(p_image, p_filter1, p_numprocessors,
                                                  p_shared_space1))

    p2 = mp.Process(target=my.image_filter, args=(p_image, p_filter2, p_numprocessors,
                                                  p_shared_space2))

    p1.start()
    p2.start()

    # wait until the processes have ended
    p1.join()
    p2.join()


filters_execution(image1, filter1, filter2, NUMCORES, shared_space1, shared_space2)
