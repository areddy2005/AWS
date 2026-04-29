import os
import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"]= " --disable-dge "

"""
Performs a 2D convolution operation using NKI.
Args:
    X: Input tensor of shape (batch_size, in_channels, input_height, input_width).
    W: Weight tensor of shape (out_channels, in_channels, filter_height, filter_width).
    bias: Bias tensor of shape (out_channels).
Returns:
    out_tensor: The result of the 2D convolution operation, with shape 
                (batch_size, out_channels, output_height, output_width).
Note:
    For ease of implementation, you can expect the inputs to abide by the following restrictions
    - filter_height == filter_width
    - input_channels % 128 == 0
    - output_channels % 128 == 0
    - output_width * output_height % 512 == 0
"""
@nki.jit
def conv2d_nki(X, W, bias):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    assert filter_height == filter_width, "Filter height must be equal to filter width"
    assert in_channels % 128 == 0, "Input channels must be divisible by 128"
    assert out_channels % 128 == 0, "Output channels must be divisible by 128"
    assert out_width * out_height % 512 == 0, "Output width * output height must be divisible by 512"

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Tiling deminsions
    c_in_tile = nl.tile_size.pmax   # The partition dimension (for SBUF and Tensor Engine) = 128
    c_out_tile = nl.tile_size.gemm_stationary_fmax   # The width of the Tensor Engine = 128
    n_tiles_c_in = in_channels // c_in_tile
    n_tiles_c_out = out_channels // c_out_tile

    # Load in the weights into the SBUF, aranged for matmuls
    w = nl.ndarray(
        shape=(c_in_tile, c_out_tile, n_tiles_c_out, n_tiles_c_in, filter_height, filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf
    )
    for c_out_tile_idx in nl.affine_range(n_tiles_c_out):
        for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    # 1. Load the weight tile for the current input and output channel tiles idx and filter position
                    # 2. Store it in the w array at the correct location and orientation
                    # YOUR CODE HERE
                    w[:, :, c_out_tile_idx, c_in_tile_idx, i, j] = nl.load_transpose2d(W[c_out_tile_idx * c_out_tile : (c_out_tile_idx + 1) * c_out_tile,c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile, i, j])
    # Process the images one-by-one
    for img in nl.affine_range(batch_size):
        # Process each output channel tile
        for c_out_tile_idx in nl.affine_range(n_tiles_c_out):
            # Convolve: for each output row, convolve over the input channel tiles and filter positions
            for out_row in nl.affine_range(out_height):
                # Assign PSUM buffer to accumulate output row
                # YOUR CODE HERE
                row_out = nl.zeros(shape = (c_out_tile,out_width), dtype = nl.float32, buffer = nl.psum)                
                
                # Loop over the input channel tiles and filter positions, accumulating the output row
                for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                    for i in nl.affine_range(filter_height):
                        for j in nl.affine_range(filter_width):
                            # 1. Select the weight tile for the current input and output channel tiles idx and filter position
                            # 2. Load the input tile for the current input channel tile idx, output row, filter position
                            # 3. Matmul the weight tile and input tile, and accumulate the result in row_out
                            # YOUR CODE HERE
                           curr_w = w[:,:,c_out_tile_idx,c_in_tile_idx,i,j]
                           Xl = nl.load(X[img,c_in_tile_idx*c_in_tile: (c_in_tile_idx+1)*c_in_tile,out_row+i,j:j+out_width])
                           row_out += nisa.nc_matmul(curr_w, Xl)

                # Load and add the bias to the row_out based on the current output channel tile idx
                # YOUR CODE HERE
                bias = nl.load(bias[c_out_tile_idx*c_out_tile:(c_out_tile_idx+1)*c_out_tile])
                # Store the output  
                # YOUR CODE HERE
                row_out = nl.add(row_out,bias)
                nl.store(X_out[img,c_out_tile_idx*c_out_tile: (c_out_tile_idx+1)*c_out_tile,out_row,:],row_out)

    return X_out
