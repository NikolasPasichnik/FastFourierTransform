import argparse
import numpy as np
import cv2
import math 
from scipy.fft import fft, ifft, fft2, ifft2
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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
    image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

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
            X[k] = X[k] + (vector[n] * np.exp((-2j * np.pi * k * n)/N))

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

def FFT_1D(vector_1D):
    N = len(vector_1D)

    # Base case: Use naive when size is small 
    if N <= 16:
        return naive_DFT_1D(vector_1D)
    
    # Split input array into even-indexed elements and odd-indexed elements and compute the FFT for each
    even = FFT_1D(vector_1D[::2]) # take every second element starting from index 0
    odd = FFT_1D(vector_1D[1::2]) # same as above but starting from index 1

    # Combine the results of even and odd
    # X = even + factor * odd
    X = np.zeros(N, dtype=complex)
    k = np.arange(N // 2)
    factor = np.exp((-2j * np.pi * k) / N)


    X[:N // 2] = even + factor * odd # first half: lower frequencies
    X[N // 2:] = even - factor * odd # second half: higher frequencies

    return X

def inverse_FFT_1D(vector_1D):
    N = len(vector_1D)

    # Base case: Use naive when size is small 
    if N <= 16:
        return naive_inverse_DFT_1D(vector_1D)
    
    # Split input array into even-indexed elements and odd-indexed elements and compute the FFT for each
    even = inverse_FFT_1D(vector_1D[0::2]) # take every second element starting from index 0
    odd = inverse_FFT_1D(vector_1D[1::2]) # same as above but starting from index 1

    # Combine the results of even and odd
    # X = even + factor * odd
    X = np.zeros(N, dtype=complex)
    k = np.arange(N // 2)
    factor = np.exp((2j * np.pi * k) / N)

    X[:N // 2] = (N//2)*(even + factor * odd) # first half: lower frequencies
    X[N // 2:] = (N//2)*(even - factor * odd) # second half: higher frequencies

    return X/N

def FFT_2D(vector_2D):
    M = len(vector_2D[0]) #row/width 
    N = len(vector_2D) #column/height
    F = np.zeros((N,M), dtype=complex)
    
    # Iterating over every row of the 2D vector (?)
    for m in range(M): 
        F[:, m] = FFT_1D(vector_2D[:, m])
    
    # Iterating over every column of the 2D vector (?)
    for n in range(N): 
        F[n] = FFT_1D(F[n])
    
    return F

def inverse_FFT_2D(vector_2D):
    M = len(vector_2D[0]) #row/width 
    N = len(vector_2D) #column/height
    F = np.zeros((N,M), dtype=complex)
    
    # Iterating over every row of the 2D vector (?)
    for m in range(M): 
        F[:, m] = inverse_FFT_1D(vector_2D[:, m])
    
    # Iterating over every column of the 2D vector (?)
    for n in range(N): 
        F[n] = inverse_FFT_1D(F[n])
    
    return F

# ========================================= Different Modes =========================================

# Original vs. Fast Fourier Transform
def mode_1(image_array):
    print("Mode 1")

    # Obtaining the Fast Fourier Transform of the inputted image (its array) 
    fft_image = FFT_2D(image_array)

    # Plotting the resulting Fourier Transform 
    fig, (graph1, graph2) = plt.subplots(1, 2)
    graph1.set_title('Original Image')
    graph1.imshow(image_array, cmap="gray")
    graph2.set_title('Fourier Transform')
    graph2.imshow(np.abs(fft_image), norm=colors.LogNorm())
    plt.show()


'''
My vision for mode 2: 

First get the fft_2d, that's fine it follows the instructions 

Then, since they ask us to "set all the high frequencies" to 0, i want to to first find what a "high frequency" means using percentile 
Now the reason for the np.abs is bcs np.percentile doesnt take imaginary numbers. 

After getting the frequencies that represent the highest (~0 and ~2pi) 

I want to filter the 2d array and set all the values that are <= lowest and >= highest to 0 

then i want to get the ifft of this array. in theory, we should get the denoised version. 
'''

# Original vs. Denoised 
def mode_2(image_array):
    print("mode 2")
    
    # Obtaining the Fast Fourier Transform of the inputted image (its array) 
    fft_image = FFT_2D(image_array)

    print("step 2")
    # ~~Denoise Process~~
    # High frequencies -> near 0 or 2pi, so we can get the bottom percentile (near 0) or the top percentile (near 2pi) 

    # Getting the cutoffs 
    low_frequency = np.percentile(np.abs(fft_image), 1)
    high_frequency = np.percentile(np.abs(fft_image), 99)

    print("step 3")
    # Setting the high frequencies to 0 
    choice = [0]
    fft_image_filtered = np.select(np.logical_or(fft_image <= low_frequency,fft_image >= high_frequency), choice, fft_image)

    print("step 4")
    # Inverting the filtered image
    denoised_image_filtered = inverse_FFT_2D(fft_image_filtered)
    print("step 5")
    # plotting 
    # Plotting the resulting Fourier Transform 
    fig, (graph1, graph2) = plt.subplots(1, 2)
    graph1.set_title('Original Image')
    graph1.imshow(image_array, cmap="gray")
    graph2.set_title('Denoised Image')
    graph2.imshow(np.abs(denoised_image_filtered), cmap="gray")
    plt.show()

    # Denoise: 
    #   - take FFT of the image 
    #   - set all the high frequencies to 0 
    #   - take the IFFT of the FFT with updated high frequencies 



def mode_3():
    print("mode 3")

def mode_4():
    print("mode 4")

# ===================================================================================================

if __name__ == "__main__":
    args = parse_input()

    # Getting input 
    print(args.mode)
    print(args.image)

    # Getting adjusted dimensions 
    adjusted_image_array = get_adjusted_dimensions(args.image)

    # # Testing the functions
    y = adjusted_image_array
    # y = fft2(adjusted_image_array)

    # za = inverse_FFT_2D(y)
    # print("ours:")
    # print(za)

    # zb = ifft2(y)
    # print("actual:")
    # print(zb)

    # Running the correct mode
    mode = args.mode
    if mode == 1:
        mode_1(adjusted_image_array)
    elif mode == 2: 
        print("Work in progress here")
        mode_2(adjusted_image_array)
    elif mode == 3: 
        mode_3()
    elif mode == 4: 
        mode_4()
    else: 
        print("Invalid mode") #add actual error


    
