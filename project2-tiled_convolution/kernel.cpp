kernel  =  SourceModule ("""
__global__ void image_filter( float * image,          //Source GPU array floating point 32 bits,
                                  float * filter_mask,    //Filter Mask GPU array 2D floating point 32 bits
                              float * filtered_image,  //Target GPU array 2D floating point 32 bits,
                              int NumRowsImg,          //Image Numrows,
                              int NumColsImg,          //Int32 Image Numcolumns,
                              int NumRowsFilter,       //Int32 Image NumRows filter mask,
                              int NumColsFilter        //Int32 Image NumCols filter mask
                              ) 
{ 
int WA = NumColsImg; 
int HA = NumRowsImg; 

// Define the kernel radius (the amount of cells that are used either right or less)

int HC = NumRowsFilter/2; 
int WC = NumColsFilter/2; 

const uint ts = 12;    // defines the tile size

// define image subdividision where the filters will be applied
__shared__ float imgMat[ 16 * 16 ];

// define thread and block indexes 
const uint tx = threadIdx.x;
const uint ty = threadIdx.y;

const uint bx = blockIdx.x;
const uint by = blockIdx.y;

// Define several indexes 

int col = bx * ts + tx ;
int row = by * ts + ty ; 

int row_i = row - HC; 
int col_i = col - WC;

// Fill the tile memory 

if (row_i< HA && row_i>=0 && col_i<WA && col_i>=0)
{
  imgMat[ty * blockDim.x + tx]= image[col_i + row_i * WA]; 
}
else 
{
  imgMat[ty * blockDim.x + tx] = 0;                        
}

__syncthreads();

// Computing of the filter itself 

float temp = 0.0f; 

if (ty < ts && tx < ts)
{ 
  temp = 0; 
  for (int i = 0; i < NumRowsFilter ; i++)
  {
    for (int j = 0; j < NumColsFilter ; j++)
    { 
      temp += filter_mask[j + i * NumColsFilter] * imgMat[(ty + i)* blockDim.x + (tx + j)];
    }
  }
  if (row < HA && col < WA)
  {
    filtered_image[col + row * WA] = temp; 
  }    
}  


}

""")