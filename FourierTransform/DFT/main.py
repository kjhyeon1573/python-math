# Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
# This file contains the implementation of the Discrete Fourier Transform (DFT) algorithm.

# Import the necessary libraries
import numpy as np

# Import type hints
import typing

# Define the function that computes the Discrete Fourier Transform (DFT) of a given signal x
# The function takes as input the signal x and the number of samples N
def DFT(x : np.array, N : int) -> np.array:
        
    # Compute the twiddle factors
    twiddlefactors = np.exp(-2*np.pi*1j*np.arange(N)/N)

    # Initialize the output array
    y = np.zeros(N,dtype=np.complex64)

    # Initialize the temporary array
    _x = x+0j

    # Compute the DFT
    for j in range(N):

        y[j] = np.sum(_x)

        _x = twiddlefactors * _x

    return y

if __name__ == "__main__":
    
    # Test the function with N = 320
    x = np.random.randn(320)
    y = DFT(x, 320)

    # Compare with numpy's FFT
    y_np = np.fft.fft(x)

    # Check if the results are the same
    print(y - y_np)
    print(np.allclose(y, y_np))