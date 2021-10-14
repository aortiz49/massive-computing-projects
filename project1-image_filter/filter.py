import multiprocessing as mp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cProfile

import myfunctions as my

# sets the number of cores
NUMCORES = 4

# constructs a 500x1000 matrix
matrix_1 = np.random.rand(500,1000)

# constructs a 1000x500 matrix
matrix_2 = np.random.rand(1000,500)

# initializes global matrix "matrix_2" which is shared memory. Multiple threads can read from this matrix
my.init_second(matrix_2.shape,matrix_2)

# assign variables to the shape of matrix_1
(rows,columns) = matrix_1.shape

# sequential multiplication of matrix_1 and matrix_2
for f in range(rows):
    v = matrix_1[f,:]
    c = my.parallel_matmul(v)

# splits matrix_1 into a list of rows
v = list()
for f in range(row):
    v.append(matrix_1[f,:])


# executes the parallel multiplication
# maps the list v to parallel_matmul which gets executed in a thread pool
def matrix_multiplication(v):
    with mp.Pool(processes = NUMCORES,initializer = my.init_second,initargs = [matrix_2.shape,matrix_2]) as p:
        result = p.map(my.parallel_matmul,v)
    return result

# obtains the multiplication result from the parallel execution
parallel_result = matrix_multiplication(v)    

