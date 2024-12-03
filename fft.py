import argparse
import time
import numpy as np
import cv2
import math 
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

# ========================================= Naive Discrete Fourier Transform (1D and 2D) =========================================

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
    # print("naive_DFT_2D")

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
    print("Executing: Mode 1")

    # Obtaining the Fast Fourier Transform of the inputted image (its array) 
    fft_output = FFT_2D(image_array)

    # Obtaining the Fast Fourier Transform using np for reference (experiment)
    # fft_image_lib = np.fft.fft2(image_array)

    # Plotting the resulting Fourier Transform 
    # fig, (graph1, graph2, graph3) = plt.subplots(1, 3) (experiment)
    fig, (graph1, graph2) = plt.subplots(1, 3)
    fig.subplots_adjust(wspace=0.5)
    graph1.set_title('Original Image')
    graph1.imshow(image_array, cmap="gray")
    graph2.set_title('Our Implementation')
    graph2.imshow(np.abs(fft_output), norm=colors.LogNorm())
    # Experiment plotting below
    # graph3.set_title("np.fft.fft2")
    # graph3.imshow(np.abs(fft_image_lib), norm=colors.LogNorm())

    plt.show()

# Original vs. Denoised 
def mode_2(image_array):
    print("Executing: Mode 2")
    
    # Obtaining the Fast Fourier Transform of the inputted image (its array) 
    fft_output = FFT_2D(image_array)
    filtered_fft_output = np.copy(fft_output)

    # ~~Denoise Process~~
    # High frequencies -> near 0 or 2pi, so we can get the bottom percentile (near 0) or the top percentile (near 2pi) 
    # Getting the cutoffs 
    # We are using .real to only account for real numbers and not complex ones
    low_cutoff = np.percentile(fft_output.real, 0.0001)
    high_cutoff = np.percentile(fft_output.real, 99.9999)
   
    # Setting the high frequencies to 0 
    filtered_fft_output = np.where(np.logical_or(fft_output <= low_cutoff,fft_output >= high_cutoff), 0, fft_output)
    # filtered_fft_output = np.where(np.logical_and(fft_output >= low_cutoff,fft_output <= high_cutoff), 0, fft_output) (experiment)


    # Count and print the number of non-zeros and fraction represented of the original Fourier coefficients
    count_nonzeros = np.count_nonzero(filtered_fft_output)
    print("Number of non-zeros: " + str(count_nonzeros))

    # Fraction =  # of non-zeros / # pixels in the image (nbr of rows * nbr of columns)
    print("Fraction of non-zeros: " + str(count_nonzeros / (len(image_array) * len(image_array[0]))))

    # Inverting the filtered image to get denoised image
    denoised_filtered_fft_output = inverse_FFT_2D(filtered_fft_output).real

    # Plotting the original and denoised images
    fig, (graph1, graph2) = plt.subplots(1, 2)
    graph1.set_title('Original Image')
    graph1.imshow(image_array, cmap="gray")
    graph2.set_title('Denoised Image')
    graph2.imshow(denoised_filtered_fft_output, cmap="gray")
    plt.show()

# Compression
def mode_3(image_array):
    print("Executing: Mode 3")
    # Obtaining the Fast Fourier Transform of the inputted image (its array) 
    fft_output = FFT_2D(image_array)

    compressed_images = [] # Array storing the compressed images at different compression levels, used for plotting
    compression_levels = [0, 20, 40, 60, 80, 99.9] 

    # Loop iterating over each level defined in compression_levels
    for level in compression_levels:
        # Create a copy of the image to not modify the original image while performing the compression
        fft_output_copy = np.copy(fft_output) 

        # Compute the amount of frequencies to keep aka threshold
        threshold = (100 - level) // 2 # the // 2 ensures a symmetric cutoff

        # Calculate the lower and upper bounds for frequency values to keep
        low_cutoff = np.percentile(fft_output.real, threshold)
        high_cutoff = np.percentile(fft_output.real, 100 - threshold)

        # Condition checking if frequencies are within or outside of bounds
        condition = np.logical_or(fft_output_copy >= high_cutoff, fft_output_copy <= low_cutoff)

        # All frequencies outside bounds will be set to 0 (compression)
        compressed_fft_output = fft_output_copy * condition

        # Count the number of non-zeros in the compressed image
        count_nonzero = np.count_nonzero(compressed_fft_output)
        print(f"Level {level}% compression has {count_nonzero} non-zeros out of {compressed_fft_output.size}")

        # Obtain final transformed image
        compressed_img = inverse_FFT_2D(compressed_fft_output).real

        compressed_images.append(compressed_img)

    # Plot the compressed images for each level
    fig, graph = plt.subplots(2, 3) # 2 x 3 grid
    for i in range (len(compressed_images)):
        # Calculate row and column for the current subplot
        r, c = divmod(i,3) 
        # Plot
        graph[r,c].imshow(compressed_images[i], cmap="gray")
        graph[r,c].set_title(f"Compression: {compression_levels[i]}%")
    
    plt.show()
    
def mode_4():
    print("Executing: Mode 4")

    # sizes = [2^5, 2^6, 2^7, 2^8, 2^9, 2^10]
    sizes = [32, 64, 128, 256]

    # arrays to hold data that will be plotted 
    naive_dft_averages = [] 
    naive_dft_std_dev = []
    fft_averages = []
    fft_std_dev = []

    print("========== Computing Average and Variances for varying problem sizes ==========\n")
    # Executing a set of naive dft and fft for each problem size
    for index in range(len(sizes)):
        # print(f"-----Size #{sizes[index]}----") 
        current_size = sizes[index]
        runtime_naive_dft = []
        runtime_fft = []

        # Creating a random 2D array of size current_size
        array_2d = np.random.random((current_size, current_size))

        # Executing the naive dft and fft for each provlem size 10 times to get a representative average
        for iteration in range(10):
            # print(f"-----Iteration #{iteration}----")
            # Running Naive DFT
            start = time.time()
            naive_DFT_2D(array_2d)
            end = time.time() 
            # Updating runtime array
            runtime_naive_dft.append(end-start) 

            # Running FFT DFT
            start = time.time()
            FFT_2D(array_2d)
            end = time.time() 
            # Updating runtime array
            runtime_fft.append(end-start) 

        # Computing the averages for naive and fft
        naive_dft_averages.append(np.average(runtime_naive_dft))
        fft_averages.append(np.average(runtime_fft))

        # Computing the standard deviation for naive and fft
        naive_dft_std_dev.append(np.std(runtime_naive_dft))
        fft_std_dev.append(np.std(runtime_fft))

        # print("===============================================================================")
        print(f"Problem Size: {current_size}")
        print(f"Average Naive DFT Runtime: {naive_dft_averages[index]}")
        print(f"Standard Deviation of Naive DFT Runtime: {naive_dft_std_dev[index]}")
        print(f"Average FFT Runtime: {fft_averages[index]}")
        print(f"Standard Deviation of FFT Runtime: {fft_std_dev[index]}")
        print("\n===============================================================================\n")
    
    # Plotting 
    plt.title("Fourier Transform Runtime vs Array Size")
    plt.xlabel("Problem Size")
    plt.ylabel("Runtime (s)")

    # Plotting the Naive Discrete Fourier Transform runtime (with error bars)
    naive_dft_error_bar = [std * 2 for std in naive_dft_std_dev]
    plt.errorbar(sizes, naive_dft_averages, label= "Naive DFT", color="red", yerr=naive_dft_error_bar, capsize=5, ecolor="black")

    # Plotting the Fast Fourier Transform runtime (with error bars)
    fft_error_bar = [std * 2 for std in fft_std_dev]
    plt.errorbar(sizes, fft_averages, label= "FFT", color="blue", yerr=fft_error_bar, capsize=5, ecolor="black")

    # Displaying the graph
    plt.legend() 
    plt.show() 

if __name__ == "__main__":
    args = parse_input()

    # Getting adjusted dimensions 
    adjusted_image_array = get_adjusted_dimensions(args.image)

    # Running the correct mode
    mode = args.mode
    if mode == 1:
        mode_1(adjusted_image_array)
    elif mode == 2: 
        mode_2(adjusted_image_array)
    elif mode == 3: 
        mode_3(adjusted_image_array)
    elif mode == 4: 
        mode_4()
    else: 
        print("Invalid Mode") 