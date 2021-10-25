import multiprocessing as mp
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cProfile

import functions as my
import numpy as np
from multiprocessing.sharedctypes import Array
import ctypes

# sets the number of cores that will be destined to the thread pool
NUMCORES = int(mp.cpu_count() / 2)

# opens the image to be filtered
image = np.array(Image.open('lena.png'))

# sets the kernels that will be used to filter the images

# this filter echoes the original image with no changes 
filter1 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]])

# this is a first order edge filter by rows
filter2 = np.array([0.5, 0, -0.5])

# this is first order filter by columns
filter3 = np.array([[0.5], [0], [0.5]])

# this is a second-order bi-directional edge filter 
filter4 = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]])

# this is a gaussian blur filter
filter5 = np.array([
    [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
    [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
    [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
    [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
    [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]
])

# this is a test 1x5 filter
filter6 = np.array([0.5, 0, 0, 0, -0.5])

# this is a test 5x1 filter
filter7 = np.array([[0.5], [0], [0], [0], [0.5]])

# initializes image
print(f'Processing image..')


def tonumpyarray(mp_arr):
    # mp_array is a shared memory array with lock

    return np.frombuffer(mp_arr.get_obj(), dtype=np.uint8)


# defines the mp vector1 for the image
data_buffer1_size = image.shape[0] * image.shape[1] * image.shape[2]
shared_space1 = Array(ctypes.c_byte, data_buffer1_size)

# defines the mp vector2 for the image
data_buffer2_size = image.shape[0] * image.shape[1] * image.shape[2]
shared_space2 = Array(ctypes.c_byte, data_buffer2_size)


def filters_execution(p_image, p_filter1, p_filter2, p_numprocessors, p_shared_space1,
                      p_shared_space2):
    """
    This functions executes the two processes that will use the thread pools to run the fulters

    :param p_image: the image to be filtered
    :param p_filter1: the first filter
    :param p_filter2: the seconds filter
    :param p_numprocessors: the number of processors to be used
    :param p_shared_space1: the first shared memory space
    :param p_shared_space2: the second shared memory space
    """
    # creates a lock to handle memory access
    lock = mp.Lock()

    # define and start the processes
    p1 = mp.Process(target=my.image_filter, args=(p_image, p_filter1, p_numprocessors,
                                                  p_shared_space1))
    p2 = mp.Process(target=my.image_filter, args=(p_image, p_filter2, p_numprocessors,
                                                  p_shared_space2))

    # start the processes
    p1.start()
    p2.start()

    # wait until the processes have ended
    p1.join()
    p2.join()

# define this in order to run the filters
if __name__ == '__main__':
    filters_execution(image, filter4, filter5, NUMCORES, shared_space1, shared_space2)

    filtered_image1 = tonumpyarray(shared_space1).reshape(image.shape)
    filtered_image2 = tonumpyarray(shared_space2).reshape(image.shape)

    fig = plt.figure(figsize=(10, 7))

    # Adds a subplot at the 1st position
    fig.add_subplot(1, 2, 1)

    # showing image
    plt.imshow(filtered_image1)
    plt.axis('off')
    plt.title("Filter1")

    # Adds a subplot at the 2nd position
    fig.add_subplot(1, 2, 2)

    # showing image
    plt.imshow(filtered_image2)
    plt.axis('off')
    plt.title("Filter2")

    plt.show()
