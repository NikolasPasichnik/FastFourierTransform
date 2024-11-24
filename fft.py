import argparse
import numpy as np
import cv2
import math 
from scipy.fft import fft, ifft, fft2, ifft2


def parse_input():
    '''
    Parsing input from the cmd 
    '''
    parser = argparse.ArgumentParser(usage='fft.py [-m mode] [-i image]')
    parser.add_argument('-m', "--mode", type=int, default=1, metavar='mode', help="""
                        [1] Fast mode (default): Convert image to FFT form and display
                        [2] Denoise: The image is denoised by applying an FFT, truncating high frequencies and then displayed
                        [3] Compress: Compress image and plot
                        [4] Plotting: Plot runtime graphs for the report
                        """)
    parser.add_argument('-i', "--image", type=str, default="moonlanding.png", metavar='image', help='Filename of the image for the DFT')

    return parser.parse_args()

# Adjusting the size of the inputted image to make sure the algorithms work. 
def get_adjusted_dimensions(image_path):
    image_array = cv2.imread(image_path)

    height_power = math.ceil(math.log2(len(image_array)))
    width_power = math.ceil(math.log2(len(image_array[0])))

    adjusted_height = pow(2,height_power)
    adjusted_width = pow(2,width_power)

    adjusted_image_array = cv2.resize(image_array, (adjusted_width,adjusted_height))

    return adjusted_image_array


# ========================================= Discrete Fourier Transform (1D and 2D) =========================================

# Discrete Fourier Transform (DFT) -> X_k = sum{n: 0 to N-1}(x_n * e^(-i*[2{pi}*k*n]/N)) for k = 0 to N-1
def naive_DFT_1D(vector):
    # print("naive_DFT_1D")

    N = len(vector)
    X = np.zeros(N, dtype=complex)

    for k in range(N): 
        for n in range(N): 
            X[k] = X[k] + (vector[n] * np.exp((-2j * math.pi * k * n)/N))

    return X 


# Discrete fourier transform (inverse) -> x_n = 1/N * sum{k: 0 to N-1}(X_n * e^(-i*[2{pi}*k*n]/N)) for k = 0 to N-1
def naive_inverse_DFT_1D(vector):
    # print("naive_inverse_DFT_1D")

    N = len(vector)
    x = np.zeros(N, dtype=complex)

    for n in range(N): 
        for k in range(N): 
            x[n] = x[n] + (vector[k] * np.exp((2j * math.pi * k * n)/N))
        
        x[n] = 1/N * x[n]

    return x 

# Discrete fourier transform 2D
def naive_DFT_2D(vector_2D):
    print("naive_DFT_2D")

    M = len(vector_2D[0]) #row/width 
    N = len(vector_2D) #column/height
    F = np.zeros((N,M), dtype=complex)
    
    # Iterating over every row of the 2D vector (?)
    for m in range(M): 
        F[:, m] = naive_DFT_1D(vector_2D[:, m])
    
    # Iterating over every column of the 2D vector (?)
    for n in range(N): 
        F[n] = naive_DFT_1D(F[n])
    
    return F

# Discrete fourier transform 2D (Inverse)
def naive_inverse_DFT_2D(vector_2D):
    print("naive_inverse_DFT_2D")

    M = len(vector_2D[0]) #row/width 
    N = len(vector_2D) #column/height
    F = np.zeros((N,M), dtype=complex)
    
    # Iterating over every row of the 2D vector (?)
    for m in range(M): 
        F[:, m] = naive_inverse_DFT_1D(vector_2D[:, m])
    
    # Iterating over every column of the 2D vector (?)
    for n in range(N): 
        F[n] = naive_inverse_DFT_1D(F[n])

    return F

# ========================================= Fast Fourier Transform (1D and 2D) =========================================

def FFT_1D(vector):
    print("TODO")

def inverse_FFT_1D(vector):
    print("TODO")

def FFT_1D(vector):
    print("TODO")

def inverse_FFT_2D(vector):
    print("TODO")

# # ===================================================================================================================

if __name__ == "__main__":
    args = parse_input()
    # Getting input 
    print(args.mode)
    print(args.image)

    # Original image dimensions
    image_array = cv2.imread(args.image)
    print("original dimensions:")
    print(len(image_array)) #height
    print(len(image_array[0])) #width 
    
    # Adjusted image dimensions 
    adjusted_image_array = get_adjusted_dimensions(args.image)
    print("adjusted dimensions:")
    print(len(adjusted_image_array))
    print(len(adjusted_image_array[0]))

    # naive_dft_1D(adjusted_image_array)
    
    # Testing the 1D DFT
    # x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
    # # print(x)
    # y = fft(x)
    # # print(y)
    # w = ifft(y)
    # print(w)
    # z = naive_inverse_DFT_1D(y)
    # print(z)


    # Testing the 2D DFT
    # x = np.mgrid[:3, :3][0]
    # print(x)
    # print(ifft2(x))
    # print(naive_inverse_DFT_2D(x))
    
