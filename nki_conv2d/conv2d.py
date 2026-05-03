import os
import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["NEURON_CC_FLAGS"] = " --disable-dge "

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

    K = filter_height

    # Tiling dimensions
    c_in_tile = nl.tile_size.pmax                       # partition dim = 128
    c_out_tile = nl.tile_size.gemm_stationary_fmax      # tensor engine free dim = 128
    n_tiles_c_in = in_channels // c_in_tile
    n_tiles_c_out = out_channels // c_out_tile

    # Block of output rows processed per spatial iteration.
    # The asserted divisibility by 512 means (out_h * out_w) is always a multiple
    # of 512, but out_w may not divide 512 cleanly for arbitrary shapes.
    # Pick the largest block_rows that evenly tiles out_height.
    if out_width <= 512 and (512 % out_width == 0):
        candidate = 512 // out_width
        # Shrink if it does not divide out_height
        while candidate > 1 and out_height % candidate != 0:
            candidate //= 2
        block_rows = candidate
    else:
        block_rows = 1
    n_row_blocks = out_height // block_rows

    # Output tensor in HBM
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Stage all weights into SBUF, transposed so that nc_matmul sees
    # stationary tile (P=c_in=128, F=c_out=128).
    w = nl.ndarray(
        shape=(c_in_tile, c_out_tile, n_tiles_c_out, n_tiles_c_in, filter_height, filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )
    for c_out_tile_idx in nl.affine_range(n_tiles_c_out):
        for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    w[:, :, c_out_tile_idx, c_in_tile_idx, i, j] = nl.load_transpose2d(
                        W[c_out_tile_idx * c_out_tile : (c_out_tile_idx + 1) * c_out_tile,
                          c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                          i, j]
                    )

    # Hoist bias loads: one SBUF tile per output-channel tile, reused for every
    # image and every spatial block. Trace-time Python loop so each tile is a
    # distinct SBUF allocation.
    bias_sbuf = []
    for c_out_idx in range(n_tiles_c_out):
        bias_sbuf.append(
            nl.load(bias[c_out_idx * c_out_tile : (c_out_idx + 1) * c_out_tile])
        )

    # Main compute loop
    for img in nl.affine_range(batch_size):
        for row_block in nl.affine_range(n_row_blocks):
            row_start = row_block * block_rows

            # Allocate PSUM accumulators for this spatial block.
            # One (128, out_width) tile per (c_out_tile, in-block row).
            psum_tiles = []
            for c_out_idx in range(n_tiles_c_out):
                row_psums = []
                for r in range(block_rows):
                    row_psums.append(
                        nl.zeros(
                            shape=(c_out_tile, out_width),
                            dtype=nl.float32,
                            buffer=nl.psum,
                        )
                    )
                psum_tiles.append(row_psums)

            # Reduce over input-channel tiles
            for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                # Single wide activation load that covers the full output row block
                # plus the kernel halo. Reused for every (i, j, r) tap below.
                X_band = nl.load(
                    X[img,
                      c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                      row_start : row_start + block_rows + K - 1,
                      0 : out_width + K - 1]
                )

                # Fully unrolled kernel taps and output-channel tiles.
                # Inner loop over r so consecutive matmuls write to different
                # PSUM accumulators.
                for i in range(K):
                    for j in range(K):
                        for c_out_idx in range(n_tiles_c_out):
                            W_tile = w[:, :, c_out_idx, c_in_tile_idx, i, j]
                            for r in range(block_rows):
                                X_slice = X_band[:, r + i, j : j + out_width]
                                psum_tiles[c_out_idx][r] += nisa.nc_matmul(W_tile, X_slice)

            # Add hoisted bias and store one HBM row per in-block output row.
            for c_out_idx in range(n_tiles_c_out):
                for r in range(block_rows):
                    result = nl.add(psum_tiles[c_out_idx][r], bias_sbuf[c_out_idx])
                    nl.store(
                        X_out[img,
                              c_out_idx * c_out_tile : (c_out_idx + 1) * c_out_tile,
                              row_start + r,
                              :],
                        result,
                    )

    return X_out
