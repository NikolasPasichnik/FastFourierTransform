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

def fft_dft_1d(img_1D_array):
        # we can assume that the size of img_1D_array is a power of 2 as when 
        # converting the original image into a NumPy array, we resize it so that it is.
    N = len(img_1D_array)
    if N <= 16: # stop splitting the problem and use naive method instead
        # We chose to stop splitting the problems at 16
        # We just want the runtime of your FFT to be in the same order of magnitude as what is theoretically expected 
        return naive_DFT_1D(img_1D_array)
    else:
        # Split the sum in the even and odd indices which we sum separately and then put together
        X = np.zeros(N, dtype=complex)
        # list[start:end:step].
        even = fft_dft_1d(img_1D_array[0::2])
        odd = fft_dft_1d(img_1D_array[1::2])

        for k in range (N //2):
            X[k] = even[k] + np.exp((-2j * np.pi * k) / N) * odd[k]
            X[k + (N // 2)] = even[k] + np.exp((-2j * np.pi * (k + (N // 2))) / N) * odd[k]
        return X
        
def fft_dft_1d_inverse(img_1D_array):
    # we can assume that the size of img_1D_array is a power of 2 as when 
    # converting the original image into a NumPy array, we resize it so that it is.
    N = len(img_1D_array)
    if N <= 16:
        return naive_DFT_1D(img_1D_array)
    else:
        # Split the sum in the even and odd indices which we sum separately and then put together
        x = np.zeros(N, dtype=complex)
        # list[start:end:step].
        even = fft_dft_1d_inverse(img_1D_array[0::2])
        odd = fft_dft_1d_inverse(img_1D_array[1::2])

        for k in range (N //2):
            x[k] = (1/2)* (even[k] + np.exp((2j * np.pi * k) / N) * odd[k])
            x[k + (N // 2)] = (1/2)* (even[k] + np.exp((2j * np.pi * (k + (N // 2))) / N) * odd[k])

        return x
        
def fft_dft_2d_inverse(img_2D_array):
    complex_img_array = np.asarray(img_2D_array, dtype=complex)
    h, w = complex_img_array.shape[:2]

    F = np.zeros((h, w), dtype=complex)

    for row in range(h):
        F[row, :] = fft_dft_1d_inverse(complex_img_array[row, :])

    for column in range(w):
        F[:, column] = fft_dft_1d_inverse(F[:,column])
    return F   
    
def fft_dft_2d(img_2D_array):
    complex_img_array = np.asarray(img_2D_array, dtype=complex)
    h, w = complex_img_array.shape[:2]

    F = np.zeros((h, w), dtype=complex)

    for row in range(h):
        F[row, :] = fft_dft_1d(complex_img_array[row, :])

    for column in range(w):
        F[:, column] = fft_dft_1d(F[:,column])
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

# Original vs. Denoised 
def mode_2(image_array):
    print("mode 2")
    
    # Obtaining the Fast Fourier Transform of the inputted image (its array) 
    fft_image = FFT_2D(image_array)

    # ~~Denoise Process~~
    # High frequencies -> near 0 or 2pi, so we can get the bottom percentile (near 0) or the top percentile (near 2pi) 
    # Getting the cutoffs 
    # We are using .real to only account for real numbers and not complex ones
    low_cutoff = np.percentile(fft_image.real, 0.0001)
    high_cutoff = np.percentile(fft_image.real, 99.999)
   
    # Setting the high frequencies to 0 
    fft_image_filtered = np.where(np.logical_or(fft_image <= low_cutoff,fft_image >= high_cutoff), 0, fft_image)

    # Count and print the number of non-zeros and fraction represented of the original Fourier coefficients
    count_nonzeros = np.count_nonzero(fft_image_filtered)
    print("Number of non-zeros: " + str(count_nonzeros))
    # Fraction =  # of non-zeros / # pixels in the image (nbr of rows * nbr of columns)
    print("Fraction of non-zeros: " + str(count_nonzeros / (len(image_array) * len(image_array[0]))))

    # Inverting the filtered image to get denoised image
    denoised_image_filtered = inverse_FFT_2D(fft_image_filtered).real

    # Plotting the original and denoised images
    fig, (graph1, graph2) = plt.subplots(1, 2)
    graph1.set_title('Original Image')
    graph1.imshow(image_array, cmap="gray")
    graph2.set_title('Denoised Image')
    graph2.imshow(denoised_image_filtered, cmap="gray")
    plt.show()

def mode_3(image_array):
    print("mode 3")
    # Obtaining the Fast Fourier Transform of the inputted image (its array) 
    fft_image = FFT_2D(image_array)

    compressed_img = [] # Array storing the compressed images at different compression levels, used for plotting
    compression_levels = [0, 20, 40, 60, 80, 99.9] 

    # Loop iterating over each level defined in compression_levels
    for level in compression_levels:
        # Create a copy of the image to not modify the original image while performing the compression
        fft_copy = np.copy(fft_image) 

        # Compute the amount of frequencies to keep aka threshold
        threshold = 100 - level

        # Calculate the lower and upper bounds for frequency values to keep
        low = np.percentile(fft_image.real, threshold // 2)
        high = np.percentile(fft_image.real, 100 - threshold // 2)

        # Condition checking if frequencies are within or outside of bounds
        condition = np.logical_or(fft_copy >= high, fft_copy <= low)

        # All frequencies outside bounds will be set to 0 (compression)
        fft_copy = fft_copy * condition

        # Count the number of non-zeros in the compressed image
        count_nonzero = np.count_nonzero(fft_copy)
        print(f"Level {level}% compression has {count_nonzero} non-zeros out of {fft_copy.size}")

        # Obtain final transformed image
        fft_copy = inverse_FFT_2D(fft_copy).real

        compressed_img.append(fft_copy)

    # Plot the compressed images for each level
    fig, graph = plt.subplots(2, 3) # 2 x 3 grid
    for i in range (len(compressed_img)):
        # Calculate row and column for the current subplot
        r, c = divmod(i,3) 
        # Plot
        graph[r,c].imshow(compressed_img[i], cmap="gray")
        graph[r,c].set_title(f"Compression: {compression_levels[i]}%")
    
    plt.show()
    


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
        mode_3(adjusted_image_array)
    elif mode == 4: 
        mode_4()
    else: 
        print("Invalid mode") #add actual error


    
