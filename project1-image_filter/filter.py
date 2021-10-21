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
image_filter1 = np.ones((3, 3))
image_filter2 = np.ones((1, 3))

# initializes image

print(f'Image shape: {image.shape}')

def tonumpyarray(mp_arr):
    # mp_array is a shared memory array with lock

    return np.frombuffer(mp_arr.get_obj(), dtype=np.uint8)


#HERE YOU HAVE TO DEFINE THE MULTIPROCESSING VECTOR FOR IMAGE1
data_buffer1_size=image.shape[0]*image.shape[1]*image.shape[2]
shared_space1= Array(ctypes.c_byte,data_buffer1_size)
filtered_image1_VECTOR=tonumpyarray(shared_space1)
filtered_image1= filtered_image1_VECTOR.reshape(image.shape)

#HERE YOU HAVE TO DEFINE THE MULTIPROCESSING VECTOR FOR IMAGE2
data_buffer2_size=image.shape[0]*image.shape[1]*image.shape[2]
shared_space2= Array(ctypes.c_byte,data_buffer2_size)
filtered_image2_VECTOR=tonumpyarray(shared_space2)
filtered_image2= filtered_image2_VECTOR.reshape(image.shape)

NUMCORES = mp.cpu_count()

my.image_filter(image,image_filter1,NUMCORES,filtered_image1)
my.image_filter(image,image_filter2,NUMCORES,filtered_image2)

print('hey')
# # executes the parallel filtering processes
# def filters_execution(image,filter1,filter2,numprocessors,filtered_image1,filtered_image2):
#     # creates a lock to handle memory access
#     lock = mp.Lock()
#
#     # define and start the processes
#     p1 = mp.Process(target=my.image_filter,args=(image,filter1,numprocessors,filtered_image1))
#     p2 = mp.Process(target=my.image_filter,args=(image,filter2,numprocessors,filtered_image2))
#
#     p1.start()
#     p2.start()
#
#     # wait until the processes have ended
#     p1.join()
#     p2.join()
#
#
#
# filters_execution(image1,filter1,filtered_image2,NUMCORES,filtered_image1,filtered_image2)





