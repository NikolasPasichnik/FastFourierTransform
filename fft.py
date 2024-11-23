import argparse
import numpy as np
import matplotlib.image as pltimage
import cv2
import math 

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

# Discrete fourier transform -> X_k = sum{n: 0 to N-1}(x_n * e^(-i2{pi}kn/N)) for k = 0 to N-1
def naive_dft_1D(image_array):
    print("naive_dft_1D")

# Discrete fourier transform (inverse) -> x_n = 1/N * sum{k: 0 to N-1}(X_n * e^(i2{pi}kn/N)) for k = 0 to N-1
def naive_inverse_dft_1D(image_array):
    print("naive_inverse_dft_1D")

# check instructions 
def naive_dft_2D(image_array):
    print("naive_dft_2D")


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