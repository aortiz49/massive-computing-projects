import pylab
from multiprocessing.sharedctypes import Value, Array, RawArray
from multiprocessing import Process, Lock
import ctypes
import numpy as np


def init_globalimage(img, filt):
    global image
    global my_filter
    image = img
    my_filter = filt


def image_filter(image,filter,numprocessors,filtered_image):
    init_globalimage(image, filter) #initialize global image based in filter specified in param

    global my_filter
    global shared_space

    # size contains the dimmensions of the filter
    size = my_filter.shape
    print(size)                                        

    


def filter_image3x3(r):
    '''
    This is a box filter algorithm applied to an image
    row_index: the index of the image row to filter
    '''

    # the shape of the gloabl image variable
    (rows, cols, depth) = image.shape 

    # obtains the current row of the image
    c_row = image[r, :, :]

    # edge cases

    # sets the previous row to the current row if we are in the first row or the row index is negative
    p_row = image[r - 1, :, :] if r > 0 else image[r, :, :]

    # sets the next row to the current row if we are in the last row 
    n_row = image[r, :, :] if r  ==  (rows - 1) else image[r + 1, :, :]

    # defines the result vector and sets each value to 0
    res_row = np.zeros((cols, depth), dtype = np.uint8)

    # for each layer in the image
    for d in range(depth):
        # new pixel to store the result of the dot product  
        new_pixel = 0.0

        # since our kernel is a 3x3 matrix of just ones, our dot product calculations will just multiple "1" to each 
        # pixel and add the result. 

        # the left border will take calculate the dot product between the first pixel and the kernel. 
        # [x| | | | | ] since it is the first column, we have to replicate the first pixel and extend it to the left

        # normalize the pixel and store in result var
        res_row[0, d] = int((p_row[0, d]*1.0+p_row[0, d]*1.0+p_row[1, d]*1.0)+
                     (c_row[0, d]*1.0+c_row[0, d]*1.0+c_row[1, d]*1.0)+
                     (n_row[0, d]*1.0+n_row[0, d]*1.0+n_row[1, d]*1.0)/9.0)

        #calculate the middle pixels
        for i in range (1, cols-1):
            new_pixel = 0.0
            for j in range(3):
                pixel_c = i+j-1
                new_pixel += p_row[pixel_c, d]*1.0+c_row[pixel_c, d]*1.0+n_row[pixel_c, d]*1.0
            res_row[i, d] = int(new_pixel/9.0)  

        # calculate the filtered pixel for the last column
        res_row[cols-1, d] = int((p_row[cols-2, d]*1.0+p_row[cols-1, d]*1.0+p_row[cols-1, d]*1.0)+
                     (c_row[cols-2, d]*1.0+c_row[cols-1, d]*1.0+c_row[cols-1, d]*1.0)+
                     (n_row[cols-2, d]*1.0+n_row[cols-1, d]*1.0+n_row[cols-1, d]*1.0)/9.0)

    #return the filtered row
    return res_row

def filter_image3x1(r):
    '''
    This is a box filter algorithm applied to an image
    row_index: the index of the image row to filter
    '''

    # image is the global memory array. This is a 3d numpy array image[a, b, c] in which a is the row, b is the layer, and c is the value.

    global image

    # my_filter is the kernel that will be applied to the image
    global my_filter
    
    # the shape of the gloabl image variable
    (rows, cols, depth) = image.shape 

    # obtains the current row of the image
    c_row = image[r, :, :]

    # edge cases

    # sets the previous row to the current row if we are in the first row or the row index is negative
    p_row = image[r - 1, :, :] if r > 0 else image[r, :, :]

    # sets the next row to the current row if we are in the last row 
    n_row = image[r, :, :] if r  ==  (rows - 1) else image[r + 1, :, :]

    # defines the result vector and sets each value to 0
    res_row = np.zeros((cols, depth), dtype = np.uint8)

    # for each layer in the image
    for d in range(depth):
        # new pixel to store the result of the dot product  
        new_pixel = 0.0

        # for each pixel in the row
        for i in range(cols - 1):
            res_row[i, d] = int((p_row[i, d]*1.0 + c_row[i, d]*1.0 + n_row[i, d]*1.0)/3.0)

    #return the filtered row
    return res_row    

def filter_image5x1(r):
    '''
    This is a box filter algorithm applied to an image
    row_index: the index of the image row to filter
    '''

    # image is the global memory array. This is a 3d numpy array image[a, b, c] in which a is the row, b is the layer, and c is the value.

    global image

    # my_filter is the kernel that will be applied to the image
    global my_filter
    
    # the shape of the gloabl image variable
    (rows, cols, depth) = image.shape 

    # obtains the current row of the image
    c_row = image[r, :, :]

    # edge cases

    # sets the previous row to the current row if we are in the first row or the row index is negative
    p_row = image[r - 1, :, :] if r > 0 else image[r, :, :]

    # sets the previous, previous row to the current row if we are in the first row or the row index is negative
    pp_row = image[r - 2, :, :] if r > 0 else image[r, :, :]

    # sets the next row to the current row if we are in the last row 
    n_row = image[r, :, :] if r  ==  (rows - 1) else image[r + 1, :, :]

    # sets the next row to the current row if we are in the last row 
    nn_row = image[r, :, :] if r  ==  (rows - 1) else image[r + 1, :, :]

    # defines the result vector and sets each value to 0
    res_row = np.zeros((cols, depth), dtype = np.uint8)

    # for each layer in the image
    for d in range(depth):
        # new pixel to store the result of the dot product  
        new_pixel = 0.0

        # for each pixel in the row
        for i in range(cols - 1):
            res_row[i, d] = int((pp_row[i, d]*1.0 + p_row[i, d]*1.0 + c_row[i, d]*1.0 + n_row[i, d]*1.0+nn_row[i, d]*1.0)/5.0)

    #return the filtered row
    return res_row        

#This functions just create a numpy array structure of type unsigned int8, with the memory used by our global r/w shared memory
def tonumpyarray(mp_arr):
    #mp_array is a shared memory array with lock
    
    return np.frombuffer(mp_arr.get_obj(), dtype = np.uint8)

#This function initialize the global shared memory data

def pool_init(shared_array_, srcimg, imgfilter):
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
    image = srcimg
    my_filter = imgfilter
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
        shared_matrix[row, :, :] = image[row, :, :]
    return


'''
Here I'm going to include my functions and changes in order to have them organised
''' 

def dp(v1,v2):
    
    '''
    We will create this function (dot product)to ease the process of applying the filter: 
    '''
    # The inputs are just two vectors of the same length and dot multiplies them 
    r=0
    
    for i in range(len(x)):
        r += float(x[i]*y[i])
    
    return r 

# ---------------------------------------------------------------------------------------------------------------------

def square_filter_5x5(row): 
    
    '''
    # The only input the function accepts is the row where the filter mask is been applied (as we did in the assignments).       # Then, all the additional information must be initialized previously and belongs to the global memory. 
    # As this function will only be done when the filter has been checked to be 5x5, it will be assumed from the beggining.
    '''
    
    # Global memory recalling 
    global image
    global my_filter
    global shared_space
    
    (rows,cols,depth) = image.shape
    
    # Creation of the rows that must be taken into account, they are 5 as the filter is 5x5 (p2row, prow, srow, nrow, n2row)
    srow=image[row,:,:]
    
    if ( row>0 ):
            prow=image[row-1,:,:]
    else:
            prow=image[row,:,:]
                        
    if ( row>1):
            p2row=image[row-2,:,:]
    else:
            p2row=prow 
            
    if ( row == (rows-2)):
            nrow=image[row+1,:,:]
            n2row= nrow
    elif: (row == (rows-1)):
            nrow=image[row,:,:]
            n2row= nrow
    else: 
            nrow=image[row+1,:,:]
            n2row= image[row+2,:,:]
    
    # Initialization of the result vector: frow 
    frow= np.zeros((cols,depth))
    
    # Implementation of the filter itself : 
        
        # First the main body of the filter is carried out:
        
    for j in range(depth): 
        for i in range(2,cols-2):
            frow[i,j]= (dp(p2row[i-2:i+3,j]),my_filter[0,:]+
                        dp(prow[i-2:i+3,j]),my_filter[1,:]+
                        dp(srow[i-2:i+3,j]),my_filter[2,:]+
                        dp(nrow[i-2:i+3,j]),my_filter[3,:]+
                        dp(n2row[i-2:i+3,j]),my_filter[4,:]+
                        )
                
        # Now we will copy the first one of the columns computed and the last ones 
        # (in this case the third one and the n-cols-3) and copy it in the boundary positions 
        
    frow[0,:]= frow[2,:]
    frow[1,:]= frow[2,:]
    frow[cols-2,:]= frow[cols-3,:]
    frow[cols-1,:]= frow[cols-3,:]
    
''' Need to be checked if the names are correct for this last part, the one with the global memories '''

    # And now we will unlock the shared memory in order to rewrite the row in its place: 
    with shared_space.get_lock():                   
        #while we are in this code block no ones, except this execution thread, can write in the shared memory 
            shared_matrix[row,:,:]=frow
      
          
        
        
'''      This could be an alternative way without the dot product, but the other one is much more elegant.

        for j in range(depth): 
            for i in range(2,cols-2): 
                temp1= (p2row[i-2,j]*my_filter[0,0] + prow[i-2,j]*my_filter[0,1]+ srow[i-2,j]*my_filter[0,2] + 
                        nrow[i-2,j]*my_filter[0,3] + n2row[i-2,j]*my_filter[0,4])
                temp2= (p2row[i-1,j]*my_filter[1,0] + prow[i-1,j]*my_filter[1,1]+ srow[i-1,j]*my_filter[1,2] + 
                        nrow[i-1,j]*my_filter[1,3] + n2row[i-1,j]*my_filter[1,4])
                temp3= (p2row[i,j]*my_filter[2,0] + prow[i,j]*my_filter[2,1]+ srow[i,j]*my_filter[2,2] + 
                        nrow[i,j]*my_filter[2,3] + n2row[i,j]*my_filter[2,4])
                temp4= (p2row[i+1,j]*my_filter[3,0] + prow[i+1,j]*my_filter[3,1]+ srow[i+1,j]*my_filter[3,2] + 
                        nrow[i+1,j]*my_filter[3,3] + n2row[i+1,j]*my_filter[3,4])
'''                  
    return  


# ---------------------------------------------------------------------------------------------------------------------

def row_filter_1x3(row) 

    '''
    # The only input the function accepts is the row where the filter mask is been applied (as we did in the assignments).       # Then, all the additional information must be initialized previously and belongs to the global memory. 
    # As this function will only be done when the filter has been checked to be 1x3, it will be assumed from the beggining.
    '''
    
    # Global memory recalling 
    global image
    global my_filter
    global shared_space
    
    (rows,cols,depth) = image.shape 
    size= len(my_filter)                   # !!! It shall be a single 1x3 vector
    
    # Initialization of the result vector and the row we are filtering:  
    
    srow=image[row,:,:] 
    frow= np.zeros((cols,depth))
    
    # Application of the filter: 
    
        #First we will fulfill the general case: 
        
    for j in range(depth):
        for i in range(1,cols-1): 
            frow[i,j]= dp(srow(i-1:i+2,j), my_filter) 
            
        #And now we will fulfill the two borders with the most-adjacent value 
        
    frow[0,:]= frow[1,:]
    frow[cols-1,:]= frow[cols-2,:] 
    
''' Need to be checked if the names are correct for this last part, the one with the global memories '''  

    # And now we will unlock the shared memory in order to rewrite the row in its place: 
    with shared_space.get_lock():                   
        #while we are in this code block no ones, except this execution thread, can write in the shared memory 
            shared_matrix[row,:,:]=frow
            
    return 
    

# ---------------------------------------------------------------------------------------------------------------------

def row_filter_1x5(row):          # Adaptation of the 1x3 

    '''
    # The only input the function accepts is the row where the filter mask is been applied (as we did in the assignments).       # Then, all the additional information must be initialized previously and belongs to the global memory. 
    # As this function will only be done when the filter has been checked to be 1x5, it will be assumed from the beggining.
    '''
    
    # Global memory recalling 
    global image
    global my_filter
    global shared_space
    
    (rows,cols,depth) = image.shape 
    size= len(my_filter)                   # !!! It shall be a single 1x5 vector
    
    # Initialization of the result vector and the row we are filtering:  
    
    srow=image[row,:,:] 
    frow= np.zeros((cols,depth))
    
    # Application of the filter: 
    
        #First we will fulfill the general case: 
        
    for j in range(depth):
        for i in range(2,cols-2): 
            frow[i,j]= dp(srow(i-2:i+3,j), my_filter) 
            
        #And now we will fulfill the two borders with the most-adjacent value 
        
    frow[0,:]= frow[2,:]
    frow[1,:]= frow[2,:]
    frow[cols-2,:]= frow[cols-3,:]
    frow[cols-1,:]= frow[cols-3,:]
''' Need to be checked if the names are correct for this last part, the one with the global memories '''  

    # And now we will unlock the shared memory in order to rewrite the row in its place: 
    with shared_space.get_lock():                   
        #while we are in this code block no ones, except this execution thread, can write in the shared memory 
            shared_matrix[row,:,:]=frow
            
    return 


# ---------------------------------------------------------------------------------------------------------------------


def square_filter_3x3(row): 
    
    '''
    # The only input the function accepts is the row where the filter mask is been applied (as we did in the assignments).       # Then, all the additional information must be initialized previously and belongs to the global memory. 
    # As this function will only be done when the filter has been checked to be 3x3, it will be assumed from the beggining.
    '''
    
    # Global memory recalling 
    global image
    global my_filter
    global shared_space
    
    (rows,cols,depth) = image.shape
    
    # Creation of the rows that must be taken into account:
    srow=image[row,:,:]
    
    if ( row>0 ):
            prow=image[row-1,:,:]
    else:
            prow=image[row,:,:]
                        
    if: (row == (rows-1)):
            nrow = srow
    else: 
            nrow=image[row+1,:,:]
                
    # Initialization of the result vector: frow 
    frow= np.zeros((cols,depth))
    
    # Implementation of the filter itself : 
        
        # First the main body of the filter is carried out:
        
    for j in range(depth): 
        for i in range(1,cols-1):
            frow[i,j]= (dp(prow[i-1:i+2,j]),my_filter[0,:]+
                        dp(srow[i-1:i+2,j]),my_filter[1,:]+
                        dp(nrow[i-1:i+2,j]),my_filter[2,:]+
                        )
                
        # Now we will copy the first one of the columns computed and the last ones 
        # (in this case the third one and the n-cols-3) and copy it in the boundary positions 
        
    frow[0,:]= frow[2,:]
    frow[1,:]= frow[2,:]
        
''' Need to be checked if the names are correct for this last part, the one with the global memories '''

    # And now we will unlock the shared memory in order to rewrite the row in its place: 
    with shared_space.get_lock():                   
        #while we are in this code block no ones, except this execution thread, can write in the shared memory 
            shared_matrix[row,:,:]=frow
         
    return  



# ---------------------------------------------------------------------------------------------------------------------

'''
Here we will define the first approach for the combined functions squared filter and row filters 
'''

def squared_filter(row): 
    
    # Global memory recalling 
    global image
    global my_filter
    global shared_space
    
    (rows,cols,depth) = image.shape
    size= my_filter.shape[0]
    
    # Creation of the rows that must be taken into account for a 3x3:
    srow=image[row,:,:]
    
    if ( row>0 ):
            prow=image[row-1,:,:]
    else:
            prow=image[row,:,:]
                        
    if: (row == (rows-1)):
            nrow = srow
    else: 
            nrow=image[row+1,:,:]
            
    # Now we will define the extra rows in the case the filter is a 5x5 
    
    if size==5: 
        
        if (row>1):
            p2row=image[row-2,:,:]
        else:
            p2row=prow 
            
        if (row >= (rows-2)):
            n2row= nrow
        else: 
            n2row= image[row+2,:,:]
    
                
    # Initialization of the result vector: frow 
    frow= np.zeros((cols,depth))
    
    # Implementation of the filter itself : 
    
    if size==3: 
        # First the main body of the filter is carried out (the one of a 3x3):
        
        for j in range(depth): 
            for i in range(1,cols-1):
                frow[i,j]= (dp(prow[i-1:i+2,j]),my_filter[0,:]+
                            dp(srow[i-1:i+2,j]),my_filter[1,:]+
                            dp(nrow[i-1:i+2,j]),my_filter[2,:]+
                            )
                
        # Now we will copy the first one of the columns computed and the last ones 
        # (in this case the third one and the n-cols-3) and copy it in the boundary positions 
        
        frow[0,:]= frow[2,:]
        frow[1,:]= frow[2,:]
        
    else:                                                # Here we are defining the 5x5 case 
        for j in range(depth): 
            for i in range(2,cols-2):
                frow[i,j]= (dp(p2row[i-2:i+3,j]),my_filter[0,:]+
                            dp(prow[i-2:i+3,j]),my_filter[1,:]+
                            dp(srow[i-2:i+3,j]),my_filter[2,:]+
                            dp(nrow[i-2:i+3,j]),my_filter[3,:]+
                            dp(n2row[i-2:i+3,j]),my_filter[4,:]+
                            )
                
        # And now the boundary positions: 
        
        frow[0,:]= frow[2,:]
        frow[1,:]= frow[2,:]
        frow[cols-2,:]= frow[cols-3,:]
        frow[cols-1,:]= frow[cols-3,:]
        
''' Need to be checked if the names are correct for this last part, the one with the global memories '''

    # And now we will unlock the shared memory in order to rewrite the row in its place: 
    with shared_space.get_lock():                   
        #while we are in this code block no ones, except this execution thread, can write in the shared memory 
            shared_matrix[row,:,:]=frow
         
    return  


# ------------------------------------------------------------------------------------------------

def rowed_filter(row):
     # Global memory recalling 
    global image
    global my_filter
    global shared_space
    
    (rows,cols,depth) = image.shape 
    size= len(my_filter)                   # !!! It shall be a single 1x5 vector
    
    # Initialization of the result vector and the row we are filtering:  
    
    srow=image[row,:,:] 
    frow= np.zeros((cols,depth))
    
    # Application of the filter: 
    
        #First we will fulfill the general case: 
    if size==5:    
        for j in range(depth):
            for i in range(2,cols-2): 
                frow[i,j]= dp(srow(i-2:i+3,j), my_filter) 
            
            #And now we will fulfill the two borders with the most-adjacent value 
        
        frow[0,:]= frow[2,:]
        frow[1,:]= frow[2,:]
        frow[cols-2,:]= frow[cols-3,:]
        frow[cols-1,:]= frow[cols-3,:]
        
    else: 
         for j in range(depth):
            for i in range(1,cols-1): 
                frow[i,j]= dp(srow(i-1:i+2,j), my_filter) 
            
            #And now we will fulfill the two borders with the most-adjacent value 
        
        frow[0,:]= frow[1,:]
        frow[cols-1,:]= frow[cols-2,:]
        
''' Need to be checked if the names are correct for this last part, the one with the global memories '''  

    # And now we will unlock the shared memory in order to rewrite the row in its place: 
    with shared_space.get_lock():                   
        #while we are in this code block no ones, except this execution thread, can write in the shared memory 
            shared_matrix[row,:,:]=frow
           
    return 

