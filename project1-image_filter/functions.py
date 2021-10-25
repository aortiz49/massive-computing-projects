import multiprocessing as mp
import numpy as np

# ----------------------------------------------------------------------------------------------------

def image_filter(p_image, p_filter, p_numprocessors, p_shared_space):
    '''
    This function will initilialize a pool which will run the chosen filter into the specified image.

    :param p_image: the image that will be filtered as a numpy array (either bi or tridimensional)
    :param p_filter: the filter that will be applied to the image. Consists in a 2D numpy array.
    :param p_numprocessors: the number of proccessor available to maximize the performance. Integer.
    :param p_shared_space:  the pre-initialization of the shared vector containing the result of the filter.
                            Shared memory vector.

    :return: the result of this vector will be stored into the shared memory, so no output is needed to be specified.

    '''
    # size contains the dimmensions of the filter been used -> determines the filter function being used
    size = p_filter.shape

    # Then the identification of the filter and its implementation is done. According to the nature of the filter, the
    # pool will be initialized with a different function which contains the different filters.

    rows = range(p_image.shape[0])

    if len(size) == 1:          # row-shaped filter
        with mp.Pool(processes=p_numprocessors, initializer=pool_init,
                     initargs=[p_shared_space, p_image, p_filter]) as p:
            p.map(rowed_filter, rows)
    elif size[0] > size[1]:     # column-shaped filter
        with mp.Pool(processes=p_numprocessors, initializer=pool_init,
                     initargs=[p_shared_space, p_image, p_filter]) as p:
            p.map(columned_filter, rows)
    else:                       # square-shaped filter
        with mp.Pool(processes=p_numprocessors, initializer=pool_init,
                     initargs=[p_shared_space, p_image, p_filter]) as p:
            p.map(squared_filter, rows)

# ----------------------------------------------------------------------------------------------------

def tonumpyarray(mp_arr):
    '''
    This functions just creates a numpy array structure of type unsigned int8, with the memory used
    by our global r/w shared memory

    :param mp_arr: mp_array is a shared memory array with lock

    :return: it will return the same vector introduced as input but as a numpy array
    '''
    return np.frombuffer(mp_arr.get_obj(), dtype=np.uint8)

# ----------------------------------------------------------------------------------------------------

def pool_init(shared_array_, srcimg, imgfilter):
    '''
    This function initializes the global shared memory data

    :param shared_array_: is the shared read/write data, with lock. It is a vector (because the shared
                          memory should be allocated as a vector
    :param srcimg: is the original image
    :param imgfilter: is the filter which will be applied to the image and store the results in the shared memory array

    :return: It will initialize all the required shared variables, including shared_matrix, were the results of the
             different rows being filtered will be stored
    '''

    # We define the local process memory reference for shared memory space
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

# ----------------------------------------------------------------------------------------------------

def dp(x, y):
    '''
    This functions computes the dot product between two vectors (they MUST be the same length)

    :param x: the first one of the vectors
    :param y: second vector

    :return: dot product of them both (float)
    '''
    r = 0
    for i in range(len(x)):
        r += float(x[i] * y[i])
    return r

# ----------------------------------------------------------------------------------------------------

def columned_filter(r):
    '''
    This function will be called by the pools when a column-shaped filter is applied.

    :param r: indicates the ID of the row were the filter is been applied. Integer belonging to range(rows).

    :return: The filtered row is stored into its preallocated shared memory object, so no extra output is needed.
    '''

    # Global memory recalling
    global image
    global my_filter
    global shared_space

    # the shape of the gloabl image variable
    (rows, cols, depth) = image.shape

    # obtains the current row of the image
    c_row = image[r, :, :]

    # edge cases. They do depend if the filter is 5x1 or 3x1.

    if (my_filter.shape[0] == 5):  # 5x1 filter

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

        # sets the next row to the current row if we are in the last row
        if r <= (rows - 3):
            nn_row = image[r + 2, :, :]
        elif r == (rows - 2):
            nn_row = image[r + 1, :, :]
        else:
            nn_row = image[r, :, :]

        # defines the result vector and sets each value to 0
        res_row = np.zeros((cols, depth), dtype=np.uint8)

        # now we will have to apply the filter itself:

        # for each layer in the image
        for d in range(depth):
            # for each pixel in the row
            for i in range(cols - 1):
                res_row[i, d] = int((pp_row[i, d] * my_filter[4] + p_row[i, d] * my_filter[3] +
                                     c_row[i, d] * my_filter[2] + n_row[i, d] * my_filter[1] +
                                     nn_row[i, d] * my_filter[0]))

            # res_row contains the filtered row that will be saved into the shared space

    elif (my_filter.shape[0] == 3):  # 3x1 filter

        # sets the previous row to the current row if we are in the first row or the row index is negative
        p_row = image[r - 1, :, :] if r > 0 else image[r, :, :]

        # sets the next row to the current row if we are in the last row
        n_row = image[r, :, :] if r == (rows - 1) else image[r + 1, :, :]

        # defines the result vector and sets each value to 0
        res_row = np.zeros((cols, depth), dtype=np.uint8)

        # now we will have to apply the filter itself:

        # for each layer in the image
        for d in range(depth):
            # for each pixel in the row
            for i in range(cols - 1):
                res_row[i, d] = int(
                    (p_row[i, d] * my_filter[2] + c_row[i, d] * my_filter[1] +
                     n_row[i, d] *
                     my_filter[0]))

        # res_row contains the resulting array as before

    # Once out of the if loop, when we have computed in either one of the ways res_row, we will copy it to the
    # shared space.
    # And now we unlock the shared memory in order to rewrite the row in its shared memory place:

    with shared_space.get_lock():
        # while we are in this code block no ones, except this execution thread, can write in the shared memory
        shared_matrix[r, :, :] = res_row

# ----------------------------------------------------------------------------------------------------

def squared_filter(row):

    '''
    This function will be called by the pools when a square-shaped filter is applied.

    :param r: indicates the ID of the row were the filter is been applied. Integer belonging to range(rows).

    :return: The filtered row is stored into its pre allocated shared memory object, so no extra output is needed.
    '''

    # Global memory recalling 
    global image
    global my_filter
    global shared_space

    # the shape of the global image variable and the filter characteristic one
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

    # Initialization of the result vector
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

    # And now we will unlock the shared memory in order to rewrite the row in its place:

    with shared_space.get_lock():
        # while we are in this code block no ones, except this execution thread, can write in the shared memory
        shared_matrix[row, :, :] = frow
    return

# ------------------------------------------------------------------------------------------------

def rowed_filter(row):
    '''
        This function will be called by the pools when a row-shaped filter is applied.

        :param r: indicates the ID of the row were the filter is been applied. Integer belonging to range(rows).

        :return: The filtered row is stored into its pre allocated shared memory object, so no extra output is needed.
        '''

    # Global memory recalling
    global image
    global my_filter
    global shared_space

    # the shape of the global image variable and the filter characteristic one
    (rows, cols, depth) = image.shape
    size = len(my_filter)

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
                frow[i, j] = int(dp(srow[i - 1:i + 2, j], my_filter))
            # And now we will fulfill the two borders with the most-adjacent value

        frow[0, :] = frow[1, :]
        frow[cols - 1, :] = frow[cols - 2, :]

    # And now we will unlock the shared memory in order to rewrite the row in its place: 
    with shared_space.get_lock():
        # while we are in this code block no ones, except this execution thread, can write in the
        # shared memory
        shared_matrix[row, :, :] = frow
    return


