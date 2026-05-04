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

    if n_tiles_c_out == 2:
        # out_channels == 256: split w into w0/w1; preload w0 only; pipeline w1
        # loads into first c_out0 reduction. Generic (i,j) only (no K unroll here).
        w0 = nl.ndarray(
            shape=(c_in_tile, c_out_tile, n_tiles_c_in, K, K),
            dtype=W.dtype,
            buffer=nl.sbuf,
        )
        w1 = nl.ndarray(
            shape=(c_in_tile, c_out_tile, n_tiles_c_in, K, K),
            dtype=W.dtype,
            buffer=nl.sbuf,
        )
        for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(K):
                for j in nl.affine_range(K):
                    w0[:, :, c_in_tile_idx, i, j] = nl.load_transpose2d(
                        W[0 * c_out_tile : 1 * c_out_tile,
                          c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                          i, j]
                    )

        # (c_out_tile, 1): NKI rejects 1D bias0[:] = nl.load(...); match bias_sbuf column layout.
        bias0 = nl.ndarray(shape=(c_out_tile, 1), dtype=bias.dtype, buffer=nl.sbuf)
        bias0[:, 0] = nl.load(bias[0 * c_out_tile : 1 * c_out_tile])
        bias1 = nl.ndarray(shape=(c_out_tile, 1), dtype=bias.dtype, buffer=nl.sbuf)
        bias1[:, 0] = nl.load(bias[1 * c_out_tile : 2 * c_out_tile])

        # Prologue: K==3 all w1 then w0; K==5 out256 uses one W1 HBM slab + nc_transpose per tap.
        img0 = 0
        row_start_p = 0
        X_bands = nl.ndarray(
            shape=(c_in_tile, n_tiles_c_in, block_rows + K - 1, out_width + K - 1),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )
        for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
            X_bands[:, c_in_tile_idx, :, :] = nl.load(
                X[img0,
                  c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                  row_start_p : row_start_p + block_rows + K - 1,
                  0 : out_width + K - 1]
            )

        psum0 = nl.zeros(
            shape=(c_out_tile, F_m),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
            if K == 3:
                # All w1 taps for this c_in tile (then pure w0 / TE work).
                w1[:, :, c_in_tile_idx, 0, 0] = nl.load_transpose2d(
                    W[1 * c_out_tile : 2 * c_out_tile,
                      c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                      0, 0]
                )
                w1[:, :, c_in_tile_idx, 0, 1] = nl.load_transpose2d(
                    W[1 * c_out_tile : 2 * c_out_tile,
                      c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                      0, 1]
                )
                w1[:, :, c_in_tile_idx, 0, 2] = nl.load_transpose2d(
                    W[1 * c_out_tile : 2 * c_out_tile,
                      c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                      0, 2]
                )
                w1[:, :, c_in_tile_idx, 1, 0] = nl.load_transpose2d(
                    W[1 * c_out_tile : 2 * c_out_tile,
                      c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                      1, 0]
                )
                w1[:, :, c_in_tile_idx, 1, 1] = nl.load_transpose2d(
                    W[1 * c_out_tile : 2 * c_out_tile,
                      c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                      1, 1]
                )
                w1[:, :, c_in_tile_idx, 1, 2] = nl.load_transpose2d(
                    W[1 * c_out_tile : 2 * c_out_tile,
                      c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                      1, 2]
                )
                w1[:, :, c_in_tile_idx, 2, 0] = nl.load_transpose2d(
                    W[1 * c_out_tile : 2 * c_out_tile,
                      c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                      2, 0]
                )
                w1[:, :, c_in_tile_idx, 2, 1] = nl.load_transpose2d(
                    W[1 * c_out_tile : 2 * c_out_tile,
                      c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                      2, 1]
                )
                w1[:, :, c_in_tile_idx, 2, 2] = nl.load_transpose2d(
                    W[1 * c_out_tile : 2 * c_out_tile,
                      c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                      2, 2]
                )
                # w0 row 0
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
                psum0 += nisa.nc_matmul(
                    w0[:, :, c_in_tile_idx, 0, 0],
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
                psum0 += nisa.nc_matmul(
                    w0[:, :, c_in_tile_idx, 0, 1],
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
                psum0 += nisa.nc_matmul(
                    w0[:, :, c_in_tile_idx, 0, 2],
                    X_packed,
                )
                # w0 row 1
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
                psum0 += nisa.nc_matmul(
                    w0[:, :, c_in_tile_idx, 1, 0],
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
                psum0 += nisa.nc_matmul(
                    w0[:, :, c_in_tile_idx, 1, 1],
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
                psum0 += nisa.nc_matmul(
                    w0[:, :, c_in_tile_idx, 1, 2],
                    X_packed,
                )
                # w0 row 2
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
                psum0 += nisa.nc_matmul(
                    w0[:, :, c_in_tile_idx, 2, 0],
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
                psum0 += nisa.nc_matmul(
                    w0[:, :, c_in_tile_idx, 2, 1],
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
                psum0 += nisa.nc_matmul(
                    w0[:, :, c_in_tile_idx, 2, 2],
                    X_packed,
                )
            elif K == 5:
                # W1: one contiguous nl.load [out,in,K,K] then TensorE transpose per tap (no transpose-DMA).
                W1_slab = nl.ndarray(
                    shape=(c_out_tile, c_in_tile, K, K),
                    dtype=W.dtype,
                    buffer=nl.sbuf,
                )
                W1_slab[:, :, :, :] = nl.load(
                    W[1 * c_out_tile : 2 * c_out_tile,
                      c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                      0:K,
                      0:K]
                )
                w1[:, :, c_in_tile_idx, 0, 0] = nisa.nc_transpose(W1_slab[:, :, 0, 0])
                w1[:, :, c_in_tile_idx, 0, 1] = nisa.nc_transpose(W1_slab[:, :, 0, 1])
                w1[:, :, c_in_tile_idx, 0, 2] = nisa.nc_transpose(W1_slab[:, :, 0, 2])
                w1[:, :, c_in_tile_idx, 0, 3] = nisa.nc_transpose(W1_slab[:, :, 0, 3])
                w1[:, :, c_in_tile_idx, 0, 4] = nisa.nc_transpose(W1_slab[:, :, 0, 4])
                w1[:, :, c_in_tile_idx, 1, 0] = nisa.nc_transpose(W1_slab[:, :, 1, 0])
                w1[:, :, c_in_tile_idx, 1, 1] = nisa.nc_transpose(W1_slab[:, :, 1, 1])
                w1[:, :, c_in_tile_idx, 1, 2] = nisa.nc_transpose(W1_slab[:, :, 1, 2])
                w1[:, :, c_in_tile_idx, 1, 3] = nisa.nc_transpose(W1_slab[:, :, 1, 3])
                w1[:, :, c_in_tile_idx, 1, 4] = nisa.nc_transpose(W1_slab[:, :, 1, 4])
                w1[:, :, c_in_tile_idx, 2, 0] = nisa.nc_transpose(W1_slab[:, :, 2, 0])
                w1[:, :, c_in_tile_idx, 2, 1] = nisa.nc_transpose(W1_slab[:, :, 2, 1])
                w1[:, :, c_in_tile_idx, 2, 2] = nisa.nc_transpose(W1_slab[:, :, 2, 2])
                w1[:, :, c_in_tile_idx, 2, 3] = nisa.nc_transpose(W1_slab[:, :, 2, 3])
                w1[:, :, c_in_tile_idx, 2, 4] = nisa.nc_transpose(W1_slab[:, :, 2, 4])
                w1[:, :, c_in_tile_idx, 3, 0] = nisa.nc_transpose(W1_slab[:, :, 3, 0])
                w1[:, :, c_in_tile_idx, 3, 1] = nisa.nc_transpose(W1_slab[:, :, 3, 1])
                w1[:, :, c_in_tile_idx, 3, 2] = nisa.nc_transpose(W1_slab[:, :, 3, 2])
                w1[:, :, c_in_tile_idx, 3, 3] = nisa.nc_transpose(W1_slab[:, :, 3, 3])
                w1[:, :, c_in_tile_idx, 3, 4] = nisa.nc_transpose(W1_slab[:, :, 3, 4])
                w1[:, :, c_in_tile_idx, 4, 0] = nisa.nc_transpose(W1_slab[:, :, 4, 0])
                w1[:, :, c_in_tile_idx, 4, 1] = nisa.nc_transpose(W1_slab[:, :, 4, 1])
                w1[:, :, c_in_tile_idx, 4, 2] = nisa.nc_transpose(W1_slab[:, :, 4, 2])
                w1[:, :, c_in_tile_idx, 4, 3] = nisa.nc_transpose(W1_slab[:, :, 4, 3])
                w1[:, :, c_in_tile_idx, 4, 4] = nisa.nc_transpose(W1_slab[:, :, 4, 4])
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
                        psum0 += nisa.nc_matmul(
                            w0[:, :, c_in_tile_idx, i, j],
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
                        psum0 += nisa.nc_matmul(
                            w0[:, :, c_in_tile_idx, i, j],
                            X_packed,
                        )
                        w1[:, :, c_in_tile_idx, i, j] = nl.load_transpose2d(
                            W[1 * c_out_tile : 2 * c_out_tile,
                              c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                              i, j]
                        )

        out_buf = nl.ndarray(
            shape=(c_out_tile, block_rows, out_width),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )
        for r in nl.affine_range(block_rows):
            out_buf[:, r, :] = nl.add(
                psum0[:, r * out_width : (r + 1) * out_width],
                bias0,
            )
        nl.store(
            X_out[img0,
                  0 * c_out_tile : 1 * c_out_tile,
                  row_start_p : row_start_p + block_rows,
                  :],
            out_buf,
        )

        psum1 = nl.zeros(
            shape=(c_out_tile, F_m),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
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
                    psum1 += nisa.nc_matmul(
                        w1[:, :, c_in_tile_idx, i, j],
                        X_packed,
                    )

        out_buf = nl.ndarray(
            shape=(c_out_tile, block_rows, out_width),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )
        for r in nl.affine_range(block_rows):
            out_buf[:, r, :] = nl.add(
                psum1[:, r * out_width : (r + 1) * out_width],
                bias1,
            )
        nl.store(
            X_out[img0,
                  1 * c_out_tile : 2 * c_out_tile,
                  row_start_p : row_start_p + block_rows,
                  :],
            out_buf,
        )

        # Steady: img=0, row_block 1 .. n_row_blocks-1 (w0/w1 resident; no w1 reload in c_out0 nest)
        if n_row_blocks > 1:
            for row_k in nl.sequential_range(n_row_blocks - 1):
                row_start_s = (row_k + 1) * block_rows
                img_s = 0
                X_bands = nl.ndarray(
                    shape=(c_in_tile, n_tiles_c_in, block_rows + K - 1, out_width + K - 1),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                    X_bands[:, c_in_tile_idx, :, :] = nl.load(
                        X[img_s,
                          c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                          row_start_s : row_start_s + block_rows + K - 1,
                          0 : out_width + K - 1]
                    )

                psum0 = nl.zeros(
                    shape=(c_out_tile, F_m),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
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
                            psum0 += nisa.nc_matmul(
                                w0[:, :, c_in_tile_idx, i, j],
                                X_packed,
                            )

                out_buf = nl.ndarray(
                    shape=(c_out_tile, block_rows, out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                for r in nl.affine_range(block_rows):
                    out_buf[:, r, :] = nl.add(
                        psum0[:, r * out_width : (r + 1) * out_width],
                        bias0,
                    )
                nl.store(
                    X_out[img_s,
                          0 * c_out_tile : 1 * c_out_tile,
                          row_start_s : row_start_s + block_rows,
                          :],
                    out_buf,
                )

                psum1 = nl.zeros(
                    shape=(c_out_tile, F_m),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
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
                            psum1 += nisa.nc_matmul(
                                w1[:, :, c_in_tile_idx, i, j],
                                X_packed,
                            )

                out_buf = nl.ndarray(
                    shape=(c_out_tile, block_rows, out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                for r in nl.affine_range(block_rows):
                    out_buf[:, r, :] = nl.add(
                        psum1[:, r * out_width : (r + 1) * out_width],
                        bias1,
                    )
                nl.store(
                    X_out[img_s,
                          1 * c_out_tile : 2 * c_out_tile,
                          row_start_s : row_start_s + block_rows,
                          :],
                    out_buf,
                )

        # Steady: img 1 .. batch_size-1, all row blocks
        if batch_size > 1:
            for img_k in nl.sequential_range(batch_size - 1):
                img_s = img_k + 1
                for row_block in nl.sequential_range(n_row_blocks):
                    row_start_s = row_block * block_rows
                    X_bands = nl.ndarray(
                        shape=(c_in_tile, n_tiles_c_in, block_rows + K - 1, out_width + K - 1),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                    for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                        X_bands[:, c_in_tile_idx, :, :] = nl.load(
                            X[img_s,
                              c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                              row_start_s : row_start_s + block_rows + K - 1,
                              0 : out_width + K - 1]
                        )

                    psum0 = nl.zeros(
                        shape=(c_out_tile, F_m),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
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
                                psum0 += nisa.nc_matmul(
                                    w0[:, :, c_in_tile_idx, i, j],
                                    X_packed,
                                )

                    out_buf = nl.ndarray(
                        shape=(c_out_tile, block_rows, out_width),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                    for r in nl.affine_range(block_rows):
                        out_buf[:, r, :] = nl.add(
                            psum0[:, r * out_width : (r + 1) * out_width],
                            bias0,
                        )
                    nl.store(
                        X_out[img_s,
                              0 * c_out_tile : 1 * c_out_tile,
                              row_start_s : row_start_s + block_rows,
                              :],
                        out_buf,
                    )

                    psum1 = nl.zeros(
                        shape=(c_out_tile, F_m),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )
                    for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
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
                                psum1 += nisa.nc_matmul(
                                    w1[:, :, c_in_tile_idx, i, j],
                                    X_packed,
                                )

                    out_buf = nl.ndarray(
                        shape=(c_out_tile, block_rows, out_width),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                    for r in nl.affine_range(block_rows):
                        out_buf[:, r, :] = nl.add(
                            psum1[:, r * out_width : (r + 1) * out_width],
                            bias1,
                        )
                    nl.store(
                        X_out[img_s,
                              1 * c_out_tile : 2 * c_out_tile,
                              row_start_s : row_start_s + block_rows,
                              :],
                        out_buf,
                    )

    else:
        # Stage all weights into SBUF (global w), then known-good main loop with K unroll.
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
                        elif K == 5:
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
                                    X_bands[:, c_in_tile_idx, r + 0, 3 : 3 + out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 0, 3],
                                X_packed,
                            )
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + 0, 4 : 4 + out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 0, 4],
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
                                    X_bands[:, c_in_tile_idx, r + 1, 3 : 3 + out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 1, 3],
                                X_packed,
                            )
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + 1, 4 : 4 + out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 1, 4],
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
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + 2, 3 : 3 + out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 2, 3],
                                X_packed,
                            )
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + 2, 4 : 4 + out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 2, 4],
                                X_packed,
                            )
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + 3, 0 : out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 3, 0],
                                X_packed,
                            )
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + 3, 1 : 1 + out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 3, 1],
                                X_packed,
                            )
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + 3, 2 : 2 + out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 3, 2],
                                X_packed,
                            )
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + 3, 3 : 3 + out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 3, 3],
                                X_packed,
                            )
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + 3, 4 : 4 + out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 3, 4],
                                X_packed,
                            )
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + 4, 0 : out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 4, 0],
                                X_packed,
                            )
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + 4, 1 : 1 + out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 4, 1],
                                X_packed,
                            )
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + 4, 2 : 2 + out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 4, 2],
                                X_packed,
                            )
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + 4, 3 : 3 + out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 4, 3],
                                X_packed,
                            )
                            X_packed = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + 4, 4 : 4 + out_width],
                                    engine=nisa.engine.vector,
                                )
                            psum_packed += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, 4, 4],
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
