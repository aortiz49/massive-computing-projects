import pylab
import multiprocessing as mp
from multiprocessing.sharedctypes import Value, Array, RawArray
from multiprocessing import Process, Lock
import ctypes
import numpy as np


# Auxiliary functions
def tonumpyarray(mp_arr):
    """
    This function creates a numpy array of uint8
    :param mp_arr: the array to be transformed
    """
    # mp_array is a shared memory array with a lock
    return np.frombuffer(mp_arr.get_obj(), dtype=np.uint8)


def pool_init(shared_array_, srcimg, imgfilter):
    """
    This initializes the global shared memory data

    :param shared_array_: the shared array containaing the data that has a lock and must be in
    vector format
    :param srcimg: the source image to be used
    :param imgfilter: the image filter that will be applied
    """
    # defines the local memory reference for the shared memory space
    global shared_space
    # defines the numpy matrix handler
    global shared_matrix

    # defines the read only data
    global image
    global my_filter

    # here, we initialize the global read only memory data
    image = srcimg
    my_filter = imgfilter
    size = image.shape

    # assigns the shared memory  to the local reference
    shared_space = shared_array_

    # defines the numpy matrix reference to handle data, which will use the shared memory buffer
    shared_matrix = tonumpyarray(shared_space).reshape(size)


def dp(x, y):
    """
    This function simplifies the process of calculating the dot product
    :param x: the first element of the dot product
    :param y: the second element of the dot product
    """

    r = 0

    for i in range(len(x)):
        r += float(x[i] * y[i])

    return r


# Filter functions
def image_filter(p_image, p_filter, p_numprocessors, p_shared_space):
    """
    This image filter receives the image to be filtered and the filter that will be applied
    and classifies the filter based on the dimmension

    :param p_image:  the image to be filtered
    :param p_filter: the image kernel
    :param p_numprocessors: the number of processors to use in the algorithm
    :param p_shared_space: the shared space of the destination image

    """

    # size contains the dimmensions of the filter
    size = p_filter.shape

    rows = range(p_image.shape[0])

    if len(size) == 1:  # rowed filter
        with mp.Pool(processes=p_numprocessors, initializer=pool_init,
                     initargs=[p_shared_space, p_image, p_filter]) as p:
            p.map(rowed_filter, rows)
    elif size[0] > size[1]:  # columned filter
        with mp.Pool(processes=p_numprocessors, initializer=pool_init,
                     initargs=[p_shared_space, p_image, p_filter]) as p:
            p.map(columned_filter, rows)
    else:  # square filter
        with mp.Pool(processes=p_numprocessors, initializer=pool_init,
                     initargs=[p_shared_space, p_image, p_filter]) as p:
            p.map(squared_filter, rows)


def columned_filter(r):
    """
    This function applies a filter defined by columns and determines if the filter is
    3x1 or 5x1.
    :param r: the row that will be operated on
    """
    # Global memory recalling 
    global image, res_row
    global my_filter
    global shared_space

    # the shape of the gloabl image variable
    (rows, cols, depth) = image.shape

    # obtains the current row of the image
    c_row = image[r, :, :]

    # handles the case for when the filter is 5x1
    if my_filter.shape[0] == 5:

        # the below section is the logic that is required to handle the edge cases for when the
        # previous and previous-previous rows are undefined.

        # sets the previous row to the current row if we are in the first row or the row index is
        # negative
        p_row = image[r - 1, :, :] if r > 0 else image[r, :, :]

        # sets the previous, previous row to the current row if we are in the first row or the
        # row index is negative
        pp_row = image[r - 2, :, :] if r > 1 else image[r, :, :]
        if r > 1:
            pp_row = image[r - 2, :, :]
        elif r == 1:
            pp_row = p_row
        else:
            pp_row = image[r, :, :]

        # sets the next row to the current row if we are in the last row 
        n_row = image[r, :, :] if r == (rows - 1) else image[r + 1, :, :]

        # the below section is the logic that is required to handle the edge cases for when the
        # next and next-next rows are undefined.

        # sets the next row to the current row if we are in the last row
        if r <= (rows - 3):
            nn_row = image[r + 2, :, :]
        elif r == (rows - 2):
            nn_row = image[r + 1, :, :]
        else:
            nn_row = image[r, :, :]

        # defines the result vector and sets each value to 0
        res_row = np.zeros((cols, depth), dtype=np.uint8)

        # for each layer in the image
        for d in range(depth):

            # for each pixel in the row, apply the convolution
            for i in range(cols - 1):
                res_row[i, d] = int((pp_row[i, d] * my_filter[4] + p_row[i, d] * my_filter[3] +
                                     c_row[i, d] * my_filter[2] + n_row[i, d] * my_filter[1] +
                                     nn_row[i, d] * my_filter[0]))
            # res_row contains the filtered row that will be saved into the shared space

    # handles the case for when the filter is 3x1
    elif my_filter.shape[0] == 3:

        # the below section is the logic that is required to handle the edge cases for when the
        # previous row is undefined or if the next row is invalid for this memory spaces

        # edge cases sets the previous row to the current row if we are in the first row or the
        # row index is negative
        p_row = image[r - 1, :, :] if r > 0 else image[r, :, :]

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

    # Once out of the if loop, when we have computed in one of the ways res_row, we will copy it
    # to the shared space And now we will unlock the shared memory in order to rewrite the row in
    # its place:
    with shared_space.get_lock():
        # while we are in this code block only the current execution thread can write to the
        # shared memory
        shared_matrix[r, :, :] = res_row


def squared_filter(row):
    """
    This function applies a square filter and determiens if its 3x3 or 5x5
    :param row: the row that will be operated on
    """
    # Global memory recalling
    global image
    global my_filter
    global shared_space

    (rows, cols, depth) = image.shape
    size = my_filter.shape[0]

    # Creation of the rows that must be taken into account for a 3x3:
    srow = image[row, :, :]

    if row > 0:
        prow = image[row - 1, :, :]
    else:
        prow = image[row, :, :]

    if row == (rows - 1):
        nrow = srow
    else:
        nrow = image[row + 1, :, :]

    # Now we will define the extra rows in the case the filter is a 5x5
    if size == 5:

        if row > 1:
            p2row = image[row - 2, :, :]
        else:
            p2row = prow

        if row >= (rows - 2):
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

        # handle edge cases on the boundary
        frow[0, :] = frow[2, :]
        frow[1, :] = frow[2, :]
        frow[cols - 2, :] = frow[cols - 3, :]
        frow[cols - 1, :] = frow[cols - 3, :]

    # unlock the shared memory in order to rewrite the row in its place
    with shared_space.get_lock():
        shared_matrix[row, :, :] = frow

    return


def rowed_filter(row):
    """
    This function applies a filter defined by rows and determines if the filter is
    1x3 or 1x5.
    :param row: the row that will be operated on
     """

    # Global memory recalling
    global image
    global my_filter
    global shared_space

    (rows, cols, depth) = image.shape
    size = len(my_filter)

    # initialize the rows used
    srow = image[row, :, :]
    frow = np.zeros((cols, depth))

    # handles the case when the filter is 1x5
    if size == 5:
        for j in range(depth):
            for i in range(2, cols - 2):
                frow[i, j] = int(dp(srow[i - 2:i + 3, j], my_filter))

        frow[0, :] = frow[2, :]
        frow[1, :] = frow[2, :]
        frow[cols - 2, :] = frow[cols - 3, :]
        frow[cols - 1, :] = frow[cols - 3, :]

    # handles the case when the filter ix 1x3
    else:
        for j in range(depth):
            for i in range(1, cols - 1):
                frow[i, j] = int(dp(srow[i - 1:i + 2, j], my_filter))

        frow[0, :] = frow[1, :]
        frow[cols - 1, :] = frow[cols - 2, :]

    # unlock the shared memory in order to rewrite the row in its place
    with shared_space.get_lock():
        shared_matrix[row, :, :] = frow

    return
