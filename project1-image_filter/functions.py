import pylab
import multiprocessing as mp
from multiprocessing.sharedctypes import Value, Array, RawArray
from multiprocessing import Process, Lock
import ctypes
import numpy as np


def image_filter(p_image, p_filter, p_numprocessors, p_shared_space):
    # size contains the dimmensions of the filter
    size = p_filter.shape
    # Until here we had it OK yesterday

    # Now we will make define an object function that contains the filter that will be used in
    # the pool function according to the dim of the filter: func.chosenfilter

    rows = range(p_image.shape[0])
    if size[0] < size[1]:  # row filter
        with mp.Pool(processes=p_numprocessors, initializer=pool_init,
                     initargs=[p_shared_space, p_image, p_filter]) as p:
            p.map(rowed_filter, rows)
    elif size[0] == size[1]:  # square filter
        with mp.Pool(processes=p_numprocessors, initializer=pool_init,
                     initargs=[p_shared_space, p_image, p_filter]) as p:
            p.map(squared_filter, rows)
    else:  # column filter
        with mp.Pool(processes=p_numprocessors, initializer=pool_init,
                     initargs=[p_shared_space, p_image, p_filter]) as p:
            p.map(columned_filter, rows)


'''
Here I'm fusing the two variants of the colums filters into one named columned_filter to pass it to the object function when corresponding 
'''


def columned_filter(r):
    # Global memory recalling 
    global image
    global my_filter
    global shared_space

    print('here I am in the columned filter')
    # the shape of the gloabl image variable
    (rows, cols, depth) = image.shape

    # obtains the current row of the image
    c_row = image[r, :, :]

    # edge cases
    if (my_filter.shape[0] == 5):
        # sets the previous row to the current row if we are in the first row or the row index is
        # negative
        p_row = image[r - 1, :, :] if r > 0 else image[r, :, :]

        # sets the previous, previous row to the current row if we are in the first row or the
        # row index is negative
        pp_row = image[r - 2, :, :] if r > 0 else image[r, :, :]

        # sets the next row to the current row if we are in the last row 
        n_row = image[r, :, :] if r == (rows - 1) else image[r + 1, :, :]

        # sets the next row to the current row if we are in the last row 
        nn_row = image[r, :, :] if r == (rows - 1) else image[r + 1, :, :]

        # defines the result vector and sets each value to 0
        res_row = np.zeros((cols, depth), dtype=np.uint8)

        # for each layer in the image
        for d in range(depth):

            # for each pixel in the row
            for i in range(cols - 1):
                res_row[i, d] = int((pp_row[i, d] * my_filter[4] + p_row[i, d] * my_filter[3] + c_row[
                    i, d] * my_filter[2] + n_row[i, d] * my_filter[1] + nn_row[i,] * my_filter[0]))

            # res_row contains the filtered row that will be saved into the shared space
    elif (my_filter.shape[0] == 3):

        # edge cases

        # sets the previous row to the current row if we are in the first row or the row index is negative
        p_row = image[r - 1, :, :] if r > 0 else image[r, :, :]

        print('I AM HERE')
        # sets the next row to the current row if we are in the last row
        n_row = image[r, :, :] if r == (rows - 1) else image[r + 1, :, :]

        # defines the result vector and sets each value to 0
        res_row = np.zeros((cols, depth), dtype=np.uint8)

        # for each layer in the image
        for d in range(depth):

            # for each pixel in the row
            for i in range(cols - 1):
                res_row[i, d] = int(
                    (p_row[i, d] * my_filter[2] + c_row[i, d] * my_filter[1] +
                     n_row[i, d] *
                     my_filter[0]))

        # res_row contains the resulting array as before 

    # Once out of the if loop, when we have computed in one of the ways res_row, we will copy it to the shared space 
    # And now we will unlock the shared memory in order to rewrite the row in its place: 
    with shared_space.get_lock():
        # while we are in this code block no ones, except this execution thread, can write in the shared memory
        shared_matrix[r, :, :] = res_row
        print('hello')

    # ----------------------------------------------------------------------------------------------------


# This functions just create a numpy array structure of type unsigned int8, with the memory used
# by our global r/w shared memory
def tonumpyarray(mp_arr):
    # mp_array is a shared memory array with lock

    return np.frombuffer(mp_arr.get_obj(), dtype=np.uint8)


# This function initialize the global shared memory data

def pool_init(shared_array_, srcimg, imgfilter):
    # shared_array_: is the shared read/write data, with lock. It is a vector (because the shared
    # memory should be allocated as a vector srcimg: is the original image imgfilter is the
    # filter which will be applied to the image and stor the results in the shared memory array

    # We defines the local process memory reference for shared memory space
    global shared_space
    # Here we define the numpy matrix handler
    global shared_matrix

    # Here, we will define the readonly memory data as global (the scope of this global variables
    # is the local module)
    global image
    global my_filter

    # here, we initialize the global read only memory data
    image = srcimg
    my_filter = imgfilter
    size = image.shape

    # Assign the shared memory  to the local reference
    shared_space = shared_array_
    # Defines the numpy matrix reference to handle data, which will uses the shared memory buffer
    shared_matrix = tonumpyarray(shared_space).reshape(size)


'''
Here I'm going to include my functions and changes in order to have them organised
'''


def dp(x, y):
    '''
    We will create this function (dot product)to ease the process of applying the filter:
    '''
    # The inputs are just two vectors of the same length and dot multiplies them
    r = 0

    for i in range(len(x)):
        r += float(x[i] * y[i])

    return r




'''
Here we will define the first approach for the combined functions squared filter and row filters 
'''


def squared_filter(row):
    # Global memory recalling 
    global image
    global my_filter
    global shared_space

    (rows, cols, depth) = image.shape
    size = my_filter.shape[0]

    # Creation of the rows that must be taken into account for a 3x3:
    srow = image[row, :, :]

    if (row > 0):
        prow = image[row - 1, :, :]
    else:
        prow = image[row, :, :]

    if (row == (rows - 1)):
        nrow = srow
    else:
        nrow = image[row + 1, :, :]

    # Now we will define the extra rows in the case the filter is a 5x5 

    if size == 5:

        if (row > 1):
            p2row = image[row - 2, :, :]
        else:
            p2row = prow

        if (row >= (rows - 2)):
            n2row = nrow
        else:
            n2row = image[row + 2, :, :]

    # Initialization of the result vector: frow 
    frow = np.zeros((cols, depth))

    # Implementation of the filter itself : 

    if size == 3:
        # First the main body of the filter is carried out (the one of a 3x3):

        for j in range(depth):
            for i in range(1, cols - 1):
                frow[i, j] = int(dp(prow[i - 1:i + 2, j], my_filter[0, :]) +
                                 dp(srow[i - 1:i + 2, j], my_filter[1, :]) +
                                 dp(nrow[i - 1:i + 2, j], my_filter[2, :])
                                 )

        # Now we will copy the first one of the columns computed and the last ones 
        # (in this case the third one and the n-cols-3) and copy it in the boundary positions 

        frow[0, :] = frow[2, :]
        frow[1, :] = frow[2, :]

    else:  # Here we are defining the 5x5 case
        for j in range(depth):
            for i in range(2, cols - 2):
                frow[i, j] = int(dp(p2row[i - 2:i + 3, j], my_filter[0, :]) +
                                 dp(prow[i - 2:i + 3, j], my_filter[1, :]) +
                                 dp(srow[i - 2:i + 3, j], my_filter[2, :]) +
                                 dp(nrow[i - 2:i + 3, j], my_filter[3, :]) +
                                 dp(n2row[i - 2:i + 3, j], my_filter[4, :])
                                 )

        # And now the boundary positions: 

        frow[0, :] = frow[2, :]
        frow[1, :] = frow[2, :]
        frow[cols - 2, :] = frow[cols - 3, :]
        frow[cols - 1, :] = frow[cols - 3, :]

    ''' Need to be checked if the names are correct for this last part, the one with the global memories '''

    # And now we will unlock the shared memory in order to rewrite the row in its place: 
    with shared_space.get_lock():
        # while we are in this code block no ones, except this execution thread, can write in the shared memory
        shared_matrix[row, :, :] = frow

    return


# ------------------------------------------------------------------------------------------------

def rowed_filter(row):
    # Global memory recalling
    global image
    global my_filter
    global shared_space

    (rows, cols, depth) = image.shape
    size = len(my_filter)  # !!! It shall be a single 1x5 vector

    # Initialization of the result vector and the row we are filtering:  

    srow = image[row, :, :]
    frow = np.zeros((cols, depth))

    # Application of the filter: 

    # First we will fulfill the general case:
    if size == 5:
        for j in range(depth):
            for i in range(2, cols - 2):
                frow[i, j] = int(dp(srow[i - 2:i + 3, j], my_filter))

                # And now we will fulfill the two borders with the most-adjacent value

        frow[0, :] = frow[2, :]
        frow[1, :] = frow[2, :]
        frow[cols - 2, :] = frow[cols - 3, :]
        frow[cols - 1, :] = frow[cols - 3, :]

    else:
        for j in range(depth):
            for i in range(1, cols - 1):
                frow[i,j]= int(dp(srow[i-1:i+2,j], my_filter))
            # And now we will fulfill the two borders with the most-adjacent value

        frow[0, :] = frow[1, :]
        frow[cols - 1, :] = frow[cols - 2, :]

    '''Need to be checked if the names are correct for this last part, the one with the global 
    memories '''

    # And now we will unlock the shared memory in order to rewrite the row in its place: 
    with shared_space.get_lock():
        # while we are in this code block no ones, except this execution thread, can write in the
        # shared memory
        shared_matrix[row, :, :] = frow

    return
