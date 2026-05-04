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

    use_fastpath = (
        batch_size == 4
        and in_channels == 128
        and out_channels == 256
        and input_height == 34
        and input_width == 34
        and K == 3
        and X.dtype == nl.float32
    )

    if use_fastpath:
        # Shape-specialized in128_out256 3x3 34x34 batch4: literal K/F_m/block_rows, w0/w1, X_band.
        X_out = nl.ndarray(
            shape=(4, 256, 32, 32),
            dtype=X.dtype,
            buffer=nl.hbm,
        )

        W0_slab = nl.ndarray(
            shape=(128, 128, 3, 3),
            dtype=W.dtype,
            buffer=nl.sbuf,
        )
        W1_slab = nl.ndarray(
            shape=(128, 128, 3, 3),
            dtype=W.dtype,
            buffer=nl.sbuf,
        )
        W0_slab[:, :, :, :] = nl.load(
            W[
                0:128,
                0:128,
                0:3,
                0:3,
            ]
        )
        W1_slab[:, :, :, :] = nl.load(
            W[
                128:256,
                0:128,
                0:3,
                0:3,
            ]
        )

        w0 = nl.ndarray(
            shape=(128, 128, 3, 3),
            dtype=W.dtype,
            buffer=nl.sbuf,
        )
        w1 = nl.ndarray(
            shape=(128, 128, 3, 3),
            dtype=W.dtype,
            buffer=nl.sbuf,
        )
        for i in nl.affine_range(3):
            for j in nl.affine_range(3):
                w0[:, :, i, j] = nisa.nc_transpose(W0_slab[:, :, i, j])
                w1[:, :, i, j] = nisa.nc_transpose(W1_slab[:, :, i, j])

        bias0 = nl.ndarray(
            shape=(128,),
            dtype=bias.dtype,
            buffer=nl.sbuf,
        )
        bias1 = nl.ndarray(
            shape=(128,),
            dtype=bias.dtype,
            buffer=nl.sbuf,
        )
        bias0[:] = nl.load(bias[0:128])
        bias1[:] = nl.load(bias[128:256])

        for img in nl.sequential_range(0, 4):
            for rb in nl.sequential_range(0, 2):
                row_start = rb * 16

                X_band = nl.ndarray(
                    shape=(128, 18, 34),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                X_band[:, :, :] = nl.load(
                    X[
                        img,
                        0:128,
                        row_start : row_start + 18,
                        0:34,
                    ]
                )

                psum0 = nl.zeros(
                    shape=(128, 512),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                psum1 = nl.zeros(
                    shape=(128, 512),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )

                for i in nl.affine_range(3):
                    for j in nl.affine_range(3):
                        X_packed = nl.ndarray(
                            shape=(128, 512),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for r in nl.affine_range(16):
                            X_packed[:, r * 32 : (r + 1) * 32] = nisa.tensor_copy(
                                X_band[:, r + i, j : j + 32],
                            )
                        psum0 += nisa.nc_matmul(w0[:, :, i, j], X_packed)
                        psum1 += nisa.nc_matmul(w1[:, :, i, j], X_packed)

                out0 = nl.ndarray(
                    shape=(128, 16, 32),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                for r in nl.affine_range(16):
                    out0[:, r, :] = nl.add(
                        psum0[:, r * 32 : (r + 1) * 32],
                        bias0[:],
                    )
                nl.store(
                    X_out[
                        img,
                        0:128,
                        row_start : row_start + 16,
                        0:32,
                    ],
                    out0,
                )

                out1 = nl.ndarray(
                    shape=(128, 16, 32),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                for r in nl.affine_range(16):
                    out1[:, r, :] = nl.add(
                        psum1[:, r * 32 : (r + 1) * 32],
                        bias1[:],
                    )
                nl.store(
                    X_out[
                        img,
                        128:256,
                        row_start : row_start + 16,
                        0:32,
                    ],
                    out1,
                )

        return X_out

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

    # Chunked HBM stores for out256 path (epilogue only); requires block_rows % CHUNK_ROWS == 0.
    CHUNK_ROWS = 4

    # Output tensor in HBM
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_height, out_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    X_bands_first = nl.ndarray(
        shape=(c_in_tile, n_tiles_c_in, block_rows + K - 1, out_width + K - 1),
        dtype=X.dtype,
        buffer=nl.sbuf,
    )
    for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
        X_bands_first[:, c_in_tile_idx, :, :] = nl.load(
            X[
                0,
                c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                0 : block_rows + K - 1,
                0 : out_width + K - 1,
            ]
        )

    # Stage weights: one contiguous nl.load per (c_out,c_in) slab + nc_transpose per tap.
    w = nl.ndarray(
        shape=(c_in_tile, c_out_tile, n_tiles_c_out, n_tiles_c_in, filter_height, filter_width),
        dtype=W.dtype,
        buffer=nl.sbuf,
    )
    for c_out_tile_idx in nl.affine_range(n_tiles_c_out):
        for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
            W_slab = nl.ndarray(
                shape=(c_out_tile, c_in_tile, K, K),
                dtype=W.dtype,
                buffer=nl.sbuf,
            )
            W_slab[:, :, :, :] = nl.load(
                W[c_out_tile_idx * c_out_tile : (c_out_tile_idx + 1) * c_out_tile,
                  c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                  0:K,
                  0:K]
            )
            for wi in nl.affine_range(K):
                for wj in nl.affine_range(K):
                    w[:, :, c_out_tile_idx, c_in_tile_idx, wi, wj] = nisa.nc_transpose(
                        W_slab[:, :, wi, wj]
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

    # First spatial block (img=0, row_block=0): prefetched X_bands_first overlaps weight/bias staging.
    row_start = 0
    if n_tiles_c_out == 2:
        psum0_first = nl.zeros(
            shape=(c_out_tile, F_m),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        psum1_first = nl.zeros(
            shape=(c_out_tile, F_m),
            dtype=nl.float32,
            buffer=nl.psum,
        )

        for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
            for i in nl.affine_range(K):
                for j in nl.affine_range(K):
                    X_packed_first_block = nl.ndarray(
                        shape=(c_in_tile, F_m),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                    for r in nl.affine_range(block_rows):
                        X_packed_first_block[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                            X_bands_first[:, c_in_tile_idx, r + i, j : j + out_width],
                        )
                    psum0_first += nisa.nc_matmul(
                        w[:, :, 0, c_in_tile_idx, i, j],
                        X_packed_first_block,
                    )
                    psum1_first += nisa.nc_matmul(
                        w[:, :, 1, c_in_tile_idx, i, j],
                        X_packed_first_block,
                    )

        if block_rows % CHUNK_ROWS == 0:
            for rr in nl.affine_range(block_rows // CHUNK_ROWS):
                out_chunk0_first = nl.ndarray(
                    shape=(c_out_tile, CHUNK_ROWS, out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                for cr in nl.affine_range(CHUNK_ROWS):
                    r = rr * CHUNK_ROWS + cr
                    out_chunk0_first[:, cr, :] = nl.add(
                        psum0_first[:, r * out_width : (r + 1) * out_width],
                        bias_sbuf[:, 0],
                    )
                nl.store(
                    X_out[
                        0,
                        0 * c_out_tile : 1 * c_out_tile,
                        rr * CHUNK_ROWS : (rr + 1) * CHUNK_ROWS,
                        :,
                    ],
                    out_chunk0_first,
                )

            for rr in nl.affine_range(block_rows // CHUNK_ROWS):
                out_chunk1_first = nl.ndarray(
                    shape=(c_out_tile, CHUNK_ROWS, out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                for cr in nl.affine_range(CHUNK_ROWS):
                    r = rr * CHUNK_ROWS + cr
                    out_chunk1_first[:, cr, :] = nl.add(
                        psum1_first[:, r * out_width : (r + 1) * out_width],
                        bias_sbuf[:, 1],
                    )
                nl.store(
                    X_out[
                        0,
                        1 * c_out_tile : 2 * c_out_tile,
                        rr * CHUNK_ROWS : (rr + 1) * CHUNK_ROWS,
                        :,
                    ],
                    out_chunk1_first,
                )
        else:
            out_buf0_first = nl.ndarray(
                shape=(c_out_tile, block_rows, out_width),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )
            for r in nl.affine_range(block_rows):
                out_buf0_first[:, r, :] = nl.add(
                    psum0_first[:, r * out_width : (r + 1) * out_width],
                    bias_sbuf[:, 0],
                )
            nl.store(
                X_out[
                    0,
                    0 * c_out_tile : 1 * c_out_tile,
                    0 : block_rows,
                    :,
                ],
                out_buf0_first,
            )

            out_buf1_first = nl.ndarray(
                shape=(c_out_tile, block_rows, out_width),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )
            for r in nl.affine_range(block_rows):
                out_buf1_first[:, r, :] = nl.add(
                    psum1_first[:, r * out_width : (r + 1) * out_width],
                    bias_sbuf[:, 1],
                )
            nl.store(
                X_out[
                    0,
                    1 * c_out_tile : 2 * c_out_tile,
                    0 : block_rows,
                    :,
                ],
                out_buf1_first,
            )

    else:
        for c_out_idx in nl.affine_range(n_tiles_c_out):
            psum_packed_first = nl.zeros(
                shape=(c_out_tile, F_m),
                dtype=nl.float32,
                buffer=nl.psum,
            )

            for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                for i in nl.affine_range(K):
                    for j in nl.affine_range(K):
                        X_packed_first_block = nl.ndarray(
                            shape=(c_in_tile, F_m),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for r in nl.affine_range(block_rows):
                            X_packed_first_block[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                X_bands_first[:, c_in_tile_idx, r + i, j : j + out_width],
                            )
                        psum_packed_first += nisa.nc_matmul(
                            w[:, :, c_out_idx, c_in_tile_idx, i, j],
                            X_packed_first_block,
                        )

            out_buf_first = nl.ndarray(
                shape=(c_out_tile, block_rows, out_width),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )
            for r in nl.affine_range(block_rows):
                out_buf_first[:, r, :] = nl.add(
                    psum_packed_first[:, r * out_width : (r + 1) * out_width],
                    bias_sbuf[:, c_out_idx],
                )
            nl.store(
                X_out[
                    0,
                    c_out_idx * c_out_tile : (c_out_idx + 1) * c_out_tile,
                    0 : block_rows,
                    :,
                ],
                out_buf_first,
            )

    # Remaining row blocks for img=0 (sequential_range(1, n_row_blocks) is empty if n_row_blocks==1)
    for row_block in nl.sequential_range(1, n_row_blocks):
        row_start = row_block * block_rows

        X_bands = nl.ndarray(
            shape=(c_in_tile, n_tiles_c_in, block_rows + K - 1, out_width + K - 1),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )

        for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
            X_bands[:, c_in_tile_idx, :, :] = nl.load(
                X[
                    0,
                    c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                    row_start : row_start + block_rows + K - 1,
                    0 : out_width + K - 1,
                ]
            )

        if n_tiles_c_out == 2:
            psum0_row = nl.zeros(
                shape=(c_out_tile, F_m),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            psum1_row = nl.zeros(
                shape=(c_out_tile, F_m),
                dtype=nl.float32,
                buffer=nl.psum,
            )

            for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                for i in nl.affine_range(K):
                    for j in nl.affine_range(K):
                        X_packed_row = nl.ndarray(
                            shape=(c_in_tile, F_m),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for r in nl.affine_range(block_rows):
                            X_packed_row[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                X_bands[:, c_in_tile_idx, r + i, j : j + out_width],
                            )
                        psum0_row += nisa.nc_matmul(
                            w[:, :, 0, c_in_tile_idx, i, j],
                            X_packed_row,
                        )
                        psum1_row += nisa.nc_matmul(
                            w[:, :, 1, c_in_tile_idx, i, j],
                            X_packed_row,
                        )

            if block_rows % CHUNK_ROWS == 0:
                for rr in nl.affine_range(block_rows // CHUNK_ROWS):
                    out_chunk0_row = nl.ndarray(
                        shape=(c_out_tile, CHUNK_ROWS, out_width),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                    for cr in nl.affine_range(CHUNK_ROWS):
                        r = rr * CHUNK_ROWS + cr
                        out_chunk0_row[:, cr, :] = nl.add(
                            psum0_row[:, r * out_width : (r + 1) * out_width],
                            bias_sbuf[:, 0],
                        )
                    nl.store(
                        X_out[
                            0,
                            0 * c_out_tile : 1 * c_out_tile,
                            row_start + rr * CHUNK_ROWS : row_start + (rr + 1) * CHUNK_ROWS,
                            :,
                        ],
                        out_chunk0_row,
                    )

                for rr in nl.affine_range(block_rows // CHUNK_ROWS):
                    out_chunk1_row = nl.ndarray(
                        shape=(c_out_tile, CHUNK_ROWS, out_width),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                    for cr in nl.affine_range(CHUNK_ROWS):
                        r = rr * CHUNK_ROWS + cr
                        out_chunk1_row[:, cr, :] = nl.add(
                            psum1_row[:, r * out_width : (r + 1) * out_width],
                            bias_sbuf[:, 1],
                        )
                    nl.store(
                        X_out[
                            0,
                            1 * c_out_tile : 2 * c_out_tile,
                            row_start + rr * CHUNK_ROWS : row_start + (rr + 1) * CHUNK_ROWS,
                            :,
                        ],
                        out_chunk1_row,
                    )
            else:
                out_buf0_row = nl.ndarray(
                    shape=(c_out_tile, block_rows, out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                for r in nl.affine_range(block_rows):
                    out_buf0_row[:, r, :] = nl.add(
                        psum0_row[:, r * out_width : (r + 1) * out_width],
                        bias_sbuf[:, 0],
                    )
                nl.store(
                    X_out[
                        0,
                        0 * c_out_tile : 1 * c_out_tile,
                        row_start : row_start + block_rows,
                        :,
                    ],
                    out_buf0_row,
                )

                out_buf1_row = nl.ndarray(
                    shape=(c_out_tile, block_rows, out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                for r in nl.affine_range(block_rows):
                    out_buf1_row[:, r, :] = nl.add(
                        psum1_row[:, r * out_width : (r + 1) * out_width],
                        bias_sbuf[:, 1],
                    )
                nl.store(
                    X_out[
                        0,
                        1 * c_out_tile : 2 * c_out_tile,
                        row_start : row_start + block_rows,
                        :,
                    ],
                    out_buf1_row,
                )

        else:
            for c_out_idx in nl.affine_range(n_tiles_c_out):
                psum_packed_row = nl.zeros(
                    shape=(c_out_tile, F_m),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )

                for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                    for i in nl.affine_range(K):
                        for j in nl.affine_range(K):
                            X_packed_row = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed_row[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + i, j : j + out_width],
                                )
                            psum_packed_row += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, i, j],
                                X_packed_row,
                            )

                out_buf_row = nl.ndarray(
                    shape=(c_out_tile, block_rows, out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                for r in nl.affine_range(block_rows):
                    out_buf_row[:, r, :] = nl.add(
                        psum_packed_row[:, r * out_width : (r + 1) * out_width],
                        bias_sbuf[:, c_out_idx],
                    )
                nl.store(
                    X_out[
                        0,
                        c_out_idx * c_out_tile : (c_out_idx + 1) * c_out_tile,
                        row_start : row_start + block_rows,
                        :,
                    ],
                    out_buf_row,
                )

    # All row blocks for img >= 1 (outer loop empty if batch_size==1)
    for img in nl.sequential_range(1, batch_size):
        for row_block in nl.sequential_range(n_row_blocks):
            row_start = row_block * block_rows

            X_bands = nl.ndarray(
                shape=(c_in_tile, n_tiles_c_in, block_rows + K - 1, out_width + K - 1),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )

            for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                X_bands[:, c_in_tile_idx, :, :] = nl.load(
                    X[
                        img,
                        c_in_tile_idx * c_in_tile : (c_in_tile_idx + 1) * c_in_tile,
                        row_start : row_start + block_rows + K - 1,
                        0 : out_width + K - 1,
                    ]
                )

            if n_tiles_c_out == 2:
                psum0_img = nl.zeros(
                    shape=(c_out_tile, F_m),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                psum1_img = nl.zeros(
                    shape=(c_out_tile, F_m),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )

                for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                    for i in nl.affine_range(K):
                        for j in nl.affine_range(K):
                            X_packed_img = nl.ndarray(
                                shape=(c_in_tile, F_m),
                                dtype=X.dtype,
                                buffer=nl.sbuf,
                            )
                            for r in nl.affine_range(block_rows):
                                X_packed_img[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, r + i, j : j + out_width],
                                )
                            psum0_img += nisa.nc_matmul(
                                w[:, :, 0, c_in_tile_idx, i, j],
                                X_packed_img,
                            )
                            psum1_img += nisa.nc_matmul(
                                w[:, :, 1, c_in_tile_idx, i, j],
                                X_packed_img,
                            )

                if block_rows % CHUNK_ROWS == 0:
                    for rr in nl.affine_range(block_rows // CHUNK_ROWS):
                        out_chunk0_img = nl.ndarray(
                            shape=(c_out_tile, CHUNK_ROWS, out_width),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for cr in nl.affine_range(CHUNK_ROWS):
                            r = rr * CHUNK_ROWS + cr
                            out_chunk0_img[:, cr, :] = nl.add(
                                psum0_img[:, r * out_width : (r + 1) * out_width],
                                bias_sbuf[:, 0],
                            )
                        nl.store(
                            X_out[
                                img,
                                0 * c_out_tile : 1 * c_out_tile,
                                row_start + rr * CHUNK_ROWS : row_start + (rr + 1) * CHUNK_ROWS,
                                :,
                            ],
                            out_chunk0_img,
                        )

                    for rr in nl.affine_range(block_rows // CHUNK_ROWS):
                        out_chunk1_img = nl.ndarray(
                            shape=(c_out_tile, CHUNK_ROWS, out_width),
                            dtype=X.dtype,
                            buffer=nl.sbuf,
                        )
                        for cr in nl.affine_range(CHUNK_ROWS):
                            r = rr * CHUNK_ROWS + cr
                            out_chunk1_img[:, cr, :] = nl.add(
                                psum1_img[:, r * out_width : (r + 1) * out_width],
                                bias_sbuf[:, 1],
                            )
                        nl.store(
                            X_out[
                                img,
                                1 * c_out_tile : 2 * c_out_tile,
                                row_start + rr * CHUNK_ROWS : row_start + (rr + 1) * CHUNK_ROWS,
                                :,
                            ],
                            out_chunk1_img,
                        )
                else:
                    out_buf0_img = nl.ndarray(
                        shape=(c_out_tile, block_rows, out_width),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                    for r in nl.affine_range(block_rows):
                        out_buf0_img[:, r, :] = nl.add(
                            psum0_img[:, r * out_width : (r + 1) * out_width],
                            bias_sbuf[:, 0],
                        )
                    nl.store(
                        X_out[
                            img,
                            0 * c_out_tile : 1 * c_out_tile,
                            row_start : row_start + block_rows,
                            :,
                        ],
                        out_buf0_img,
                    )

                    out_buf1_img = nl.ndarray(
                        shape=(c_out_tile, block_rows, out_width),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                    for r in nl.affine_range(block_rows):
                        out_buf1_img[:, r, :] = nl.add(
                            psum1_img[:, r * out_width : (r + 1) * out_width],
                            bias_sbuf[:, 1],
                        )
                    nl.store(
                        X_out[
                            img,
                            1 * c_out_tile : 2 * c_out_tile,
                            row_start : row_start + block_rows,
                            :,
                        ],
                        out_buf1_img,
                    )

            else:
                for c_out_idx in nl.affine_range(n_tiles_c_out):
                    psum_packed_img = nl.zeros(
                        shape=(c_out_tile, F_m),
                        dtype=nl.float32,
                        buffer=nl.psum,
                    )

                    for c_in_tile_idx in nl.affine_range(n_tiles_c_in):
                        for i in nl.affine_range(K):
                            for j in nl.affine_range(K):
                                X_packed_img = nl.ndarray(
                                    shape=(c_in_tile, F_m),
                                    dtype=X.dtype,
                                    buffer=nl.sbuf,
                                )
                                for r in nl.affine_range(block_rows):
                                    X_packed_img[:, r * out_width : (r + 1) * out_width] = nisa.tensor_copy(
                                        X_bands[:, c_in_tile_idx, r + i, j : j + out_width],
                                    )
                                psum_packed_img += nisa.nc_matmul(
                                    w[:, :, c_out_idx, c_in_tile_idx, i, j],
                                    X_packed_img,
                                )

                    out_buf_img = nl.ndarray(
                        shape=(c_out_tile, block_rows, out_width),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                    for r in nl.affine_range(block_rows):
                        out_buf_img[:, r, :] = nl.add(
                            psum_packed_img[:, r * out_width : (r + 1) * out_width],
                            bias_sbuf[:, c_out_idx],
                        )
                    nl.store(
                        X_out[
                            img,
                            c_out_idx * c_out_tile : (c_out_idx + 1) * c_out_tile,
                            row_start : row_start + block_rows,
                            :,
                        ],
                        out_buf_img,
                    )

    return X_out
