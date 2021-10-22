'''
THIS FUNCTIONS WILL HAVE TO BE PLUGGED INTO THE 'FUNCTIONS. PY' FILE, BUT IN ORDER TO HAVE EVERYTHING AS CLEAR AS POSSIBLE AT THIS STEP AND IN ORDER FOR YO TO SEE THEM ANDY, I WILL UPLOAD THEM IN THIS FILE UNTIL THE MAIN FUNCTIONS ARE CREATED AND WE NEED THIS SECONDARY ONES
'''


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




# ----------------------------- Here are the individual functions -----------------------------


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
    # The only input the function accepts is the row where the filter mask is being applied (as we did in the assignments).       # Then, all the additional information must be initialized previously and belongs to the global memory. 
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



