import os
import numpy as np

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

    # Stage all weights into SBUF, transposed so that nc_matmul sees
    # stationary tile (P=c_in=128, F=c_out=128). nl.load_transpose2d is the
    # supported path for this strided HBM (out,in) plane; nl.load of the same
    # slice is illegal. Per NKI perf guide: load_transpose2d has lower peak DMA
    # bandwidth than nl.load but is reasonable when the kernel is matmul /
    # compute dominated and the transpose amount is modest (small K^2 * tiles).
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

    # Main compute loop: sequential (img, row_block) to cap SBUF live set
    # (one X_bands, one X_packed, one PSUM per stage; F_m = block_rows * out_width).
    for img in nl.sequential_range(batch_size):
        for row_block in nl.sequential_range(n_row_blocks):
            row_start = row_block * block_rows

            X_bands = nl.ndarray(
                shape=(c_in_tile, n_tiles_c_in, block_rows + K - 1, out_width + K - 1),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )

            for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                X_bands[:, c_in_tile_idx, :, :] = nl.load(
                    X[img,
                      c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                      row_start : row_start + block_rows + K - 1,
                      0 : out_width + K - 1]
                )

            for c_out_idx in nl.affine_range(n_tiles_c_out):
                psum_packed = nl.zeros(
                    shape=(c_out_tile, F_m),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )

                for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                    if K == 3:
                        # Hard-unrolled taps: constant X_bands / w indices (K=3 only).
                        X_packed = nl.ndarray(
                            shape=(c_in_tile, F_m),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for r in nl.affine_range(block_rows):
                            X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                X_bands[:, c_in_tile_idx, r + 0, 0 : out_width],
                                engine=nisa.engine.vector,
                            )
                        psum_packed += nisa.nc_matmul(
                            w[:, :, c_out_idx, c_in_tile_idx, 0, 0],
                            X_packed,
                        )
                        X_packed = nl.ndarray(
                            shape=(c_in_tile, F_m),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for r in nl.affine_range(block_rows):
                            X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                X_bands[:, c_in_tile_idx, r + 0, 1 : 1 + out_width],
                                engine=nisa.engine.vector,
                            )
                        psum_packed += nisa.nc_matmul(
                            w[:, :, c_out_idx, c_in_tile_idx, 0, 1],
                            X_packed,
                        )
                        X_packed = nl.ndarray(
                            shape=(c_in_tile, F_m),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for r in nl.affine_range(block_rows):
                            X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                X_bands[:, c_in_tile_idx, r + 0, 2 : 2 + out_width],
                                engine=nisa.engine.vector,
                            )
                        psum_packed += nisa.nc_matmul(
                            w[:, :, c_out_idx, c_in_tile_idx, 0, 2],
                            X_packed,
                        )
                        X_packed = nl.ndarray(
                            shape=(c_in_tile, F_m),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for r in nl.affine_range(block_rows):
                            X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                X_bands[:, c_in_tile_idx, r + 1, 0 : out_width],
                                engine=nisa.engine.vector,
                            )
                        psum_packed += nisa.nc_matmul(
                            w[:, :, c_out_idx, c_in_tile_idx, 1, 0],
                            X_packed,
                        )
                        X_packed = nl.ndarray(
                            shape=(c_in_tile, F_m),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for r in nl.affine_range(block_rows):
                            X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                X_bands[:, c_in_tile_idx, r + 1, 1 : 1 + out_width],
                                engine=nisa.engine.vector,
                            )
                        psum_packed += nisa.nc_matmul(
                            w[:, :, c_out_idx, c_in_tile_idx, 1, 1],
                            X_packed,
                        )
                        X_packed = nl.ndarray(
                            shape=(c_in_tile, F_m),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for r in nl.affine_range(block_rows):
                            X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                X_bands[:, c_in_tile_idx, r + 1, 2 : 2 + out_width],
                                engine=nisa.engine.vector,
                            )
                        psum_packed += nisa.nc_matmul(
                            w[:, :, c_out_idx, c_in_tile_idx, 1, 2],
                            X_packed,
                        )
                        X_packed = nl.ndarray(
                            shape=(c_in_tile, F_m),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for r in nl.affine_range(block_rows):
                            X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                X_bands[:, c_in_tile_idx, r + 2, 0 : out_width],
                                engine=nisa.engine.vector,
                            )
                        psum_packed += nisa.nc_matmul(
                            w[:, :, c_out_idx, c_in_tile_idx, 2, 0],
                            X_packed,
                        )
                        X_packed = nl.ndarray(
                            shape=(c_in_tile, F_m),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for r in nl.affine_range(block_rows):
                            X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                X_bands[:, c_in_tile_idx, r + 2, 1 : 1 + out_width],
                                engine=nisa.engine.vector,
                            )
                        psum_packed += nisa.nc_matmul(
                            w[:, :, c_out_idx, c_in_tile_idx, 2, 1],
                            X_packed,
                        )
                        X_packed = nl.ndarray(
                            shape=(c_in_tile, F_m),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for r in nl.affine_range(block_rows):
                            X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                X_bands[:, c_in_tile_idx, r + 2, 2 : 2 + out_width],
                                engine=nisa.engine.vector,
                            )
                        psum_packed += nisa.nc_matmul(
                            w[:, :, c_out_idx, c_in_tile_idx, 2, 2],
                            X_packed,
                        )
                    else:
                        for i in nl.affine_range(K):
                            for j in nl.affine_range(K):
                                X_packed = nl.ndarray(
                                    shape=(c_in_tile, F_m),
                                    dtype=X.dtype,
                                    buffer=nl.sbuf,
                                )
                                for r in nl.affine_range(block_rows):
                                    X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                        X_bands[:, c_in_tile_idx, r + i, j : j + out_width],
                                        engine=nisa.engine.vector,
                                    )
                                W_tile = w[:, :, c_out_idx, c_in_tile_idx, i, j]
                                psum_packed += nisa.nc_matmul(W_tile, X_packed)

                out_buf = nl.ndarray(
                    shape=(c_out_tile, block_rows, out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                for r in nl.affine_range(block_rows):
                    out_buf[:, r, :] = nl.add(
                        psum_packed[:, r * out_width : (r + 1) * out_width],
                        bias_sbuf[:, c_out_idx],
                    )
                nl.store(
                    X_out[img,
                          c_out_idx * c_out_tile : (c_out_idx + 1) * c_out_tile,
                          row_start : row_start + block_rows,
                          :],
                    out_buf,
                )

    return X_out
