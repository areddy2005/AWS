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

    # Block of output rows per spatial iteration. Sized so the packed matmul
    # free dim block_rows * out_width hits the Tensor Engine maximum (~512),
    # giving 100% matmul utilization instead of out_width / 512.
    MAX_F_M = 512
    if out_width <= MAX_F_M and (MAX_F_M % out_width == 0):
        candidate = MAX_F_M // out_width
        # Shrink if it does not evenly tile out_height
        while candidate > 1 and out_height % candidate != 0:
            candidate //= 2
        block_rows = candidate
    else:
        block_rows = 1
    n_row_blocks = out_height // block_rows
    F_m = block_rows * out_width   # packed matmul free dimension

    # Output tensor in HBM
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Stage all weights into SBUF for nc_matmul (P=c_in, F=c_out). The HBM
    # slice W[out_tile, in_tile, i, j] is strided; nl.load rejects it, so we
    # dma_copy into contiguous w_staging then nc_transpose (same result as
    # load_transpose2d). w_staging is allocated inside the tap loops per NKI
    # buffer-scope guidance.
    w = nl.ndarray(
        shape=(c_in_tile, c_out_tile, n_tiles_c_out, n_tiles_c_in, filter_height, filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )
    for c_out_tile_idx in nl.affine_range(n_tiles_c_out):
        for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(filter_height):
                for j in nl.affine_range(filter_width):
                    w_staging = nl.ndarray(
                        shape=(c_out_tile, c_in_tile),
                        dtype=W.dtype,
                        buffer=nl.sbuf,
                    )
                    nisa.dma_copy(
                        dst=w_staging,
                        src=W[
                            c_out_tile_idx * c_out_tile : (c_out_tile_idx + 1) * c_out_tile,
                            c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                            i,
                            j,
                        ],
                    )
                    w[:, :, c_out_tile_idx, c_in_tile_idx, i, j] = nisa.nc_transpose(
                        w_staging
                    )

    # Hoist bias loads into one SBUF tensor: shape (128, n_tiles_c_out).
    bias_sbuf = nl.ndarray(
        shape=(c_out_tile, n_tiles_c_out),
        dtype=bias.dtype,
        buffer=nl.sbuf,
    )
    for c_out_idx in nl.affine_range(n_tiles_c_out):
        bias_sbuf[:, c_out_idx] = nl.load(
            bias[c_out_idx * c_out_tile : (c_out_idx + 1) * c_out_tile]
        )

    # Main compute loop. Outer (img, row_block) loops are sequential so the
    # compiler does NOT keep multiple iterations' SBUF allocations
    # (X_bands, X_packed) alive concurrently. With affine_range here, the
    # compiler was over-parallelizing for fp32 / small-batch shapes,
    # exceeding SBUF and spilling 80+ MiB to HBM. Sequential outer loops
    # cap the live working set at one spatial block. Inner loops stay
    # affine_range so matmul/pack/load instructions can still overlap
    # within a spatial block.
    for img in nl.sequential_range(batch_size):
        for row_block in nl.sequential_range(n_row_blocks):
            row_start = row_block * block_rows

            # Activation band for this spatial block. Allocated inside the
            # (img, row_block) loop so each parallel iteration gets its own
            # SBUF allocation (NKI affine_range parallelism requires this).
            X_bands = nl.ndarray(
                shape=(c_in_tile, n_tiles_c_in, block_rows + K - 1, out_width + K - 1),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )

            # Refill the activation band for this spatial block. One 2D DMA
            # per c_in_tile loads the whole (rows, cols) slab in a single
            # descriptor instead of (block_rows + K - 1) per-row descriptors.
            for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                X_bands[:, c_in_tile_idx, :, :] = nl.load(
                    X[img,
                      c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                      row_start : row_start + block_rows + K - 1,
                      0 : out_width + K - 1]
                )

            # One packed PSUM (128, F_m) per c_out tile.
            for c_out_idx in nl.affine_range(n_tiles_c_out):
                # Per-iteration packed PSUM accumulator. Reset to zero by
                # nl.zeros, lifetime bounded to one c_out_idx iteration so
                # the compiler can reuse the PSUM bank between iterations.
                psum_packed = nl.zeros(
                    shape=(c_out_tile, F_m),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )

                for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                    for i in nl.affine_range(K):
                        for j in nl.affine_range(K):
                            # Per-(i, j) packed moving operand. block_rows
                            # shifted (128, out_width) slices laid side-by-side
                            # along the free dim so one matmul produces all
                            # block_rows output rows for this (c_out, c_in, i, j).
                            # Allocated inside the (i, j) loop so each parallel
                            # affine_range iteration owns its own SBUF region.
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )

                            # Pack block_rows shifted slices into the moving tile.
                            # SBUF->SBUF copies are cheap relative to the matmul
                            # we save.
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = \
                                    X_bands[:, c_in_tile_idx, r + i, j : j + out_width]

                            W_tile = w[:, :, c_out_idx, c_in_tile_idx, i, j]
                            # Fused matmul+accumulate: writes into psum_packed
                            # in place, removing the transient PSUM result tile
                            # that += would otherwise allocate. Reduces PSUM
                            # bank pressure and helps the allocator avoid spill.
                            nisa.nc_matmul(
                                psum_packed,
                                W_tile,
                                X_packed,
                                accumulate=True,
                            )

                # Bias add and store: split the packed PSUM back into
                # block_rows separate output rows.
                for r in nl.affine_range(block_rows):
                    result = nl.add(
                        psum_packed[:, r * out_width : (r + 1) * out_width],
                        bias_sbuf[:, c_out_idx],
                    )
                    nl.store(
                        X_out[img,
                              c_out_idx * c_out_tile : (c_out_idx + 1) * c_out_tile,
                              row_start + r,
                              :],
                        result,
                    )

    return X_out
