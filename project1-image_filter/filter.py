import multiprocessing as mp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cProfile

import functions as my

# sets the number of cores
NUMCORES = 4

print(NUMCORES)