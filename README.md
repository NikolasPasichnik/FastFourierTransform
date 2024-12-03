# Fast Fourier Transform and Applications
This program implements two types of Discrete Fourier Transform (DFT) in Python: naive using a brute-force approach and Fast Fourier Transform (FFT) using the Cooley-Tukey approach. FFT will then be used for image compression and denoising.

To invoke the application, the following syntax must be used in the command line: 

`python fft.py [-m mode] [-i image]`

where:

- mode (optional): 
    - [1] (Default) Fast mode: Convert image to FFT form and display.
    – [2] Denoise: The image is denoised by applying an FFT, and displayed.
    – [3] Compress: Compress image and plot.
    – [4] Plot runtime graphs for the report.
- image (optional): Filename of the image for the DFT. The default image is the given moonlanding.png
  
Note that this code was written and executed in the Visual Studio Code terminal on a Windows 10 (vers. 22H2) machine using Python version 3.11.5


