from multiprocessing.sharedctypes import Value, Array, RawArray
from multiprocessing import Process, Lock
import ctypes
import numpy as np


# this function declares a global variable names matrix_2 and sets this matrix equal to the matrix that enters by the parameter and prints its dimmensions
def init_second(shape,my_matrix2):
    global matrix_2
    matrix_2=my_matrix2
    print(matrix_2.shape)


def init_globalimage(img,filt):
    global image
    global my_filter
    image=img
    my_filter=filt

def parallel_matmul(v):
    # v: is the input row
    # matrix_2: is the second matrix, shared by memory
    
    #here we calculate the shape of the second matrix, to generate the resultant row
    matrix_2 # we will uses the global matrix
    
    (rows,columns)=matrix_2.shape
    
    #we allocate the final vector of size the number of columns of matrix_2
    d=np.zeros(columns)
    
    #we calculate the dot product between vector v and each column of matrix_2
    for i in range(columns):
        d[i] = np.dot(v,matrix_2[:,i])
    
    #returns the final vector d
    return d


def filter_image(row_index):
    '''
    This is a box filter algorithm applied to an image
    row_index: the index of the image row to filter
    '''

    # image is the global memory array. This is a 3d numpy array image[a,b,c] in which a is the row, b is the layer, and c is the value.

    global image

    # my_filter is the kernel that will be applied to the image
    global my_filter
    
    # the shape of the gloabl image variable
    (rows,cols,depth) = image.shape 

    # obtains the current row of the image
    c_row = image[r,:,:]

    # edge cases

    # sets the previous row to the current row if we are in the first row or the row index is negative
    p_row = image[r - 1,:,:] if row_index > 0 else image[r,:,:]

    # sets the next row to the current row if we are in the last row 
    n_row = image[r + 1,:,:] if row_index == (rows - 1) else image[r,:,:]

    # defines the result vector and sets each value to 0
    res_row = np.zeros((cols,depth),dtype=np.uint8)

    # for each layer in the image
    for d in range(depth):
        # new pixel to store the result of the convolution  
        new_pixel = 0.0

        # since our kernel is a 3x3 matrix of just ones, our convolution calculations will just multiple "1" to each 
        # pixel and add the result. 

        # the left border will take calculate the convolution between the first pixel and the kernel. 
        # [x| | | | | ] since it is the first column, we have to replicate the first pexel and extend it to the left

        # normalize the 
        frow[0,d] = int((p_row[0,d]*1.0+p_row[0,d]*1.0+p_row[1,d]*1.0)+
                     (p_row[0,d]*1.0+c_row[0,d]*1.0+c_row[1,d]*1.0)+
                     (n_row[0,d]*1.0+n_row[0,d]*1.0+n_row[1,d]*1.0)/9.0)



#This functions just create a numpy array structure of type unsigned int8, with the memory used by our global r/w shared memory
def tonumpyarray(mp_arr):
    #mp_array is a shared memory array with lock
    
    return np.frombuffer(mp_arr.get_obj(),dtype=np.uint8)

#This function initialize the global shared memory data

def pool_init(shared_array_,srcimg, imgfilter):
    #shared_array_: is the shared read/write data, with lock. It is a vector (because the shared memory should be allocated as a vector
    #srcimg: is the original image
    #imgfilter is the filter which will be applied to the image and stor the results in the shared memory array
    
    #We defines the local process memory reference for shared memory space
    global shared_space
    #Here we define the numpy matrix handler
    global shared_matrix
    
    #Here, we will define the readonly memory data as global (the scope of this global variables is the local module)
    global image
    global my_filter
    
    #here, we initialize the global read only memory data
    image=srcimg
    my_filter=imgfilter
    size = image.shape
    
    #Assign the shared memory  to the local reference
    shared_space = shared_array_
    #Defines the numpy matrix reference to handle data, which will uses the shared memory buffer
    shared_matrix = tonumpyarray(shared_space).reshape(size)

#this function just copy the original image to the global r/w shared  memory 
def parallel_shared_imagecopy(row):
    global image
    global my_filter
    global shared_space    
    # with this instruction we lock the shared memory space, avoidin other parallel processes tries to write on it
    with shared_space.get_lock():
        #while we are in this code block no ones, except this execution thread, can write in the shared memory
        shared_matrix[row,:,:]=image[row,:,:]
    return




