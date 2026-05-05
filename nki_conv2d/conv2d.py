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

    use_fastpath_b16_3x3 = (
        batch_size == 16
        and in_channels == 128
        and out_channels == 256
        and input_height == 34
        and input_width == 34
        and K == 3
        and X.dtype == nl.float32
    )

    use_fastpath_in128_out256_3x3_b4_66x66_fp16 = (
        batch_size == 4
        and in_channels == 128
        and out_channels == 256
        and input_height == 66
        and input_width == 66
        and K == 3
        and X.dtype == nl.float16
    )

    """use_fastpath_in256_out256_3x3_b4 = (
        batch_size == 4
        and in_channels == 256
        and out_channels == 256
        and input_height == 34
        and input_width == 34
        and K == 3
        and X.dtype == nl.float32
    )"""

    if use_fastpath:
        
        X_out = nl.ndarray(
            shape=(4, 256, 32, 32),
            dtype=X.dtype,
            buffer=nl.hbm,
        )

        X_band_first = nl.ndarray(
            shape=(128, 18, 34),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )
        X_band_first[:, :, :] = nl.load(
            X[
                0,
                0:128,
                0:18,
                0:34,
            ]
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

        bias_sbuf = nl.ndarray(
            shape=(128, 2),
            dtype=bias.dtype,
            buffer=nl.sbuf,
        )
        bias_sbuf[:, 0] = nl.load(bias[0:128])
        bias_sbuf[:, 1] = nl.load(bias[128:256])

        psum0_first = nl.zeros(
            shape=(128, 512),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        psum1_first = nl.zeros(
            shape=(128, 512),
            dtype=nl.float32,
            buffer=nl.psum,
        )

        X_pack0 = nl.ndarray(
            shape=(128, 512),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )
        X_pack1 = nl.ndarray(
            shape=(128, 512),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 0:16, 0:32]).reshape((128, 512))
        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 0:16, 1:33]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 0, 0], X_pack0)
        psum1_first += nisa.nc_matmul(w1[:, :, 0, 0], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 0:16, 2:34]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 0, 1], X_pack1)
        psum1_first += nisa.nc_matmul(w1[:, :, 0, 1], X_pack1)

        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 1:17, 0:32]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 0, 2], X_pack0)
        psum1_first += nisa.nc_matmul(w1[:, :, 0, 2], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 1:17, 1:33]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 1, 0], X_pack1)
        psum1_first += nisa.nc_matmul(w1[:, :, 1, 0], X_pack1)

        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 1:17, 2:34]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 1, 1], X_pack0)
        psum1_first += nisa.nc_matmul(w1[:, :, 1, 1], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 2:18, 0:32]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 1, 2], X_pack1)
        psum1_first += nisa.nc_matmul(w1[:, :, 1, 2], X_pack1)

        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 2:18, 1:33]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 2, 0], X_pack0)
        psum1_first += nisa.nc_matmul(w1[:, :, 2, 0], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 2:18, 2:34]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 2, 1], X_pack1)
        psum1_first += nisa.nc_matmul(w1[:, :, 2, 1], X_pack1)

        psum0_first += nisa.nc_matmul(w0[:, :, 2, 2], X_pack0)
        psum1_first += nisa.nc_matmul(w1[:, :, 2, 2], X_pack0)

        nl.store(
            X_out[
                0,
                0:128,
                0:16,
                0:32,
            ],
            nl.add(
                psum0_first,
                bias_sbuf[:, 0],
            ).reshape((128, 16, 32)),
        )
        nl.store(
            X_out[
                0,
                128:256,
                0:16,
                0:32,
            ],
            nl.add(
                psum1_first,
                bias_sbuf[:, 1],
            ).reshape((128, 16, 32)),
        )

        for rb in nl.sequential_range(1, 2):
            row_start = rb * 16

            X_band = nl.ndarray(
                shape=(128, 18, 34),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )
            X_band[:, :, :] = nl.load(
                X[
                    0,
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

            X_pack0 = nl.ndarray(
                shape=(128, 512),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )
            X_pack1 = nl.ndarray(
                shape=(128, 512),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )

            X_pack0[:, :] = nisa.tensor_copy(X_band[:, 0:16, 0:32]).reshape((128, 512))
            X_pack1[:, :] = nisa.tensor_copy(X_band[:, 0:16, 1:33]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 0, 0], X_pack0)
            psum1 += nisa.nc_matmul(w1[:, :, 0, 0], X_pack0)

            X_pack0[:, :] = nisa.tensor_copy(X_band[:, 0:16, 2:34]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 0, 1], X_pack1)
            psum1 += nisa.nc_matmul(w1[:, :, 0, 1], X_pack1)

            X_pack1[:, :] = nisa.tensor_copy(X_band[:, 1:17, 0:32]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 0, 2], X_pack0)
            psum1 += nisa.nc_matmul(w1[:, :, 0, 2], X_pack0)

            X_pack0[:, :] = nisa.tensor_copy(X_band[:, 1:17, 1:33]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 1, 0], X_pack1)
            psum1 += nisa.nc_matmul(w1[:, :, 1, 0], X_pack1)

            X_pack1[:, :] = nisa.tensor_copy(X_band[:, 1:17, 2:34]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 1, 1], X_pack0)
            psum1 += nisa.nc_matmul(w1[:, :, 1, 1], X_pack0)

            X_pack0[:, :] = nisa.tensor_copy(X_band[:, 2:18, 0:32]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 1, 2], X_pack1)
            psum1 += nisa.nc_matmul(w1[:, :, 1, 2], X_pack1)

            X_pack1[:, :] = nisa.tensor_copy(X_band[:, 2:18, 1:33]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 2, 0], X_pack0)
            psum1 += nisa.nc_matmul(w1[:, :, 2, 0], X_pack0)

            X_pack0[:, :] = nisa.tensor_copy(X_band[:, 2:18, 2:34]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 2, 1], X_pack1)
            psum1 += nisa.nc_matmul(w1[:, :, 2, 1], X_pack1)

            psum0 += nisa.nc_matmul(w0[:, :, 2, 2], X_pack0)
            psum1 += nisa.nc_matmul(w1[:, :, 2, 2], X_pack0)

            nl.store(
                X_out[
                    0,
                    0:128,
                    row_start : row_start + 16,
                    0:32,
                ],
                nl.add(
                    psum0,
                    bias_sbuf[:, 0],
                ).reshape((128, 16, 32)),
            )
            nl.store(
                X_out[
                    0,
                    128:256,
                    row_start : row_start + 16,
                    0:32,
                ],
                nl.add(
                    psum1,
                    bias_sbuf[:, 1],
                ).reshape((128, 16, 32)),
            )

        for img in nl.sequential_range(1, 4):
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

                X_pack0 = nl.ndarray(
                    shape=(128, 512),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                X_pack1 = nl.ndarray(
                    shape=(128, 512),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 0:16, 0:32]).reshape((128, 512))
                X_pack1[:, :] = nisa.tensor_copy(X_band[:, 0:16, 1:33]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 0, 0], X_pack0)
                psum1 += nisa.nc_matmul(w1[:, :, 0, 0], X_pack0)

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 0:16, 2:34]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 0, 1], X_pack1)
                psum1 += nisa.nc_matmul(w1[:, :, 0, 1], X_pack1)

                X_pack1[:, :] = nisa.tensor_copy(X_band[:, 1:17, 0:32]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 0, 2], X_pack0)
                psum1 += nisa.nc_matmul(w1[:, :, 0, 2], X_pack0)

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 1:17, 1:33]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 1, 0], X_pack1)
                psum1 += nisa.nc_matmul(w1[:, :, 1, 0], X_pack1)

                X_pack1[:, :] = nisa.tensor_copy(X_band[:, 1:17, 2:34]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 1, 1], X_pack0)
                psum1 += nisa.nc_matmul(w1[:, :, 1, 1], X_pack0)

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 2:18, 0:32]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 1, 2], X_pack1)
                psum1 += nisa.nc_matmul(w1[:, :, 1, 2], X_pack1)

                X_pack1[:, :] = nisa.tensor_copy(X_band[:, 2:18, 1:33]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 2, 0], X_pack0)
                psum1 += nisa.nc_matmul(w1[:, :, 2, 0], X_pack0)

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 2:18, 2:34]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 2, 1], X_pack1)
                psum1 += nisa.nc_matmul(w1[:, :, 2, 1], X_pack1)

                psum0 += nisa.nc_matmul(w0[:, :, 2, 2], X_pack0)
                psum1 += nisa.nc_matmul(w1[:, :, 2, 2], X_pack0)

                nl.store(
                    X_out[
                        img,
                        0:128,
                        row_start : row_start + 16,
                        0:32,
                    ],
                    nl.add(
                        psum0,
                        bias_sbuf[:, 0],
                    ).reshape((128, 16, 32)),
                )
                nl.store(
                    X_out[
                        img,
                        128:256,
                        row_start : row_start + 16,
                        0:32,
                    ],
                    nl.add(
                        psum1,
                        bias_sbuf[:, 1],
                    ).reshape((128, 16, 32)),
                )

        return X_out

    elif use_fastpath_b16_3x3:
        
        X_out = nl.ndarray(
            shape=(16, 256, 32, 32),
            dtype=X.dtype,
            buffer=nl.hbm,
        )

        X_band_first = nl.ndarray(
            shape=(128, 18, 34),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )
        X_band_first[:, :, :] = nl.load(
            X[
                0,
                0:128,
                0:18,
                0:34,
            ]
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

        bias_sbuf = nl.ndarray(
            shape=(128, 2),
            dtype=bias.dtype,
            buffer=nl.sbuf,
        )
        bias_sbuf[:, 0] = nl.load(bias[0:128])
        bias_sbuf[:, 1] = nl.load(bias[128:256])

        psum0_first = nl.zeros(
            shape=(128, 512),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        psum1_first = nl.zeros(
            shape=(128, 512),
            dtype=nl.float32,
            buffer=nl.psum,
        )

        X_pack0 = nl.ndarray(
            shape=(128, 512),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )
        X_pack1 = nl.ndarray(
            shape=(128, 512),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )

        # Ping-pong: depth-2 schedule; packs via tensor_copy(window).reshape(128,512).
        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 0:16, 0:32]).reshape((128, 512))
        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 0:16, 1:33]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 0, 0], X_pack0)
        psum1_first += nisa.nc_matmul(w1[:, :, 0, 0], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 0:16, 2:34]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 0, 1], X_pack1)
        psum1_first += nisa.nc_matmul(w1[:, :, 0, 1], X_pack1)

        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 1:17, 0:32]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 0, 2], X_pack0)
        psum1_first += nisa.nc_matmul(w1[:, :, 0, 2], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 1:17, 1:33]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 1, 0], X_pack1)
        psum1_first += nisa.nc_matmul(w1[:, :, 1, 0], X_pack1)

        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 1:17, 2:34]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 1, 1], X_pack0)
        psum1_first += nisa.nc_matmul(w1[:, :, 1, 1], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 2:18, 0:32]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 1, 2], X_pack1)
        psum1_first += nisa.nc_matmul(w1[:, :, 1, 2], X_pack1)

        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 2:18, 1:33]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 2, 0], X_pack0)
        psum1_first += nisa.nc_matmul(w1[:, :, 2, 0], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 2:18, 2:34]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w0[:, :, 2, 1], X_pack1)
        psum1_first += nisa.nc_matmul(w1[:, :, 2, 1], X_pack1)

        psum0_first += nisa.nc_matmul(w0[:, :, 2, 2], X_pack0)
        psum1_first += nisa.nc_matmul(w1[:, :, 2, 2], X_pack0)

        nl.store(
            X_out[
                0,
                0:128,
                0:16,
                0:32,
            ],
            nl.add(
                psum0_first,
                bias_sbuf[:, 0],
            ).reshape((128, 16, 32)),
        )
        nl.store(
            X_out[
                0,
                128:256,
                0:16,
                0:32,
            ],
            nl.add(
                psum1_first,
                bias_sbuf[:, 1],
            ).reshape((128, 16, 32)),
        )

        for rb in nl.sequential_range(1, 2):
            row_start = rb * 16

            X_band = nl.ndarray(
                shape=(128, 18, 34),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )
            X_band[:, :, :] = nl.load(
                X[
                    0,
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

            X_pack0[:, :] = nisa.tensor_copy(X_band[:, 0:16, 0:32]).reshape((128, 512))
            X_pack1[:, :] = nisa.tensor_copy(X_band[:, 0:16, 1:33]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 0, 0], X_pack0)
            psum1 += nisa.nc_matmul(w1[:, :, 0, 0], X_pack0)

            X_pack0[:, :] = nisa.tensor_copy(X_band[:, 0:16, 2:34]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 0, 1], X_pack1)
            psum1 += nisa.nc_matmul(w1[:, :, 0, 1], X_pack1)

            X_pack1[:, :] = nisa.tensor_copy(X_band[:, 1:17, 0:32]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 0, 2], X_pack0)
            psum1 += nisa.nc_matmul(w1[:, :, 0, 2], X_pack0)

            X_pack0[:, :] = nisa.tensor_copy(X_band[:, 1:17, 1:33]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 1, 0], X_pack1)
            psum1 += nisa.nc_matmul(w1[:, :, 1, 0], X_pack1)

            X_pack1[:, :] = nisa.tensor_copy(X_band[:, 1:17, 2:34]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 1, 1], X_pack0)
            psum1 += nisa.nc_matmul(w1[:, :, 1, 1], X_pack0)

            X_pack0[:, :] = nisa.tensor_copy(X_band[:, 2:18, 0:32]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 1, 2], X_pack1)
            psum1 += nisa.nc_matmul(w1[:, :, 1, 2], X_pack1)

            X_pack1[:, :] = nisa.tensor_copy(X_band[:, 2:18, 1:33]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 2, 0], X_pack0)
            psum1 += nisa.nc_matmul(w1[:, :, 2, 0], X_pack0)

            X_pack0[:, :] = nisa.tensor_copy(X_band[:, 2:18, 2:34]).reshape((128, 512))
            psum0 += nisa.nc_matmul(w0[:, :, 2, 1], X_pack1)
            psum1 += nisa.nc_matmul(w1[:, :, 2, 1], X_pack1)

            psum0 += nisa.nc_matmul(w0[:, :, 2, 2], X_pack0)
            psum1 += nisa.nc_matmul(w1[:, :, 2, 2], X_pack0)

            nl.store(
                X_out[
                    0,
                    0:128,
                    row_start : row_start + 16,
                    0:32,
                ],
                nl.add(
                    psum0,
                    bias_sbuf[:, 0],
                ).reshape((128, 16, 32)),
            )
            nl.store(
                X_out[
                    0,
                    128:256,
                    row_start : row_start + 16,
                    0:32,
                ],
                nl.add(
                    psum1,
                    bias_sbuf[:, 1],
                ).reshape((128, 16, 32)),
            )

        for img in nl.sequential_range(1, 16):
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

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 0:16, 0:32]).reshape((128, 512))
                X_pack1[:, :] = nisa.tensor_copy(X_band[:, 0:16, 1:33]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 0, 0], X_pack0)
                psum1 += nisa.nc_matmul(w1[:, :, 0, 0], X_pack0)

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 0:16, 2:34]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 0, 1], X_pack1)
                psum1 += nisa.nc_matmul(w1[:, :, 0, 1], X_pack1)

                X_pack1[:, :] = nisa.tensor_copy(X_band[:, 1:17, 0:32]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 0, 2], X_pack0)
                psum1 += nisa.nc_matmul(w1[:, :, 0, 2], X_pack0)

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 1:17, 1:33]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 1, 0], X_pack1)
                psum1 += nisa.nc_matmul(w1[:, :, 1, 0], X_pack1)

                X_pack1[:, :] = nisa.tensor_copy(X_band[:, 1:17, 2:34]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 1, 1], X_pack0)
                psum1 += nisa.nc_matmul(w1[:, :, 1, 1], X_pack0)

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 2:18, 0:32]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 1, 2], X_pack1)
                psum1 += nisa.nc_matmul(w1[:, :, 1, 2], X_pack1)

                X_pack1[:, :] = nisa.tensor_copy(X_band[:, 2:18, 1:33]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 2, 0], X_pack0)
                psum1 += nisa.nc_matmul(w1[:, :, 2, 0], X_pack0)

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 2:18, 2:34]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w0[:, :, 2, 1], X_pack1)
                psum1 += nisa.nc_matmul(w1[:, :, 2, 1], X_pack1)

                psum0 += nisa.nc_matmul(w0[:, :, 2, 2], X_pack0)
                psum1 += nisa.nc_matmul(w1[:, :, 2, 2], X_pack0)

                nl.store(
                    X_out[
                        img,
                        0:128,
                        row_start : row_start + 16,
                        0:32,
                    ],
                    nl.add(
                        psum0,
                        bias_sbuf[:, 0],
                    ).reshape((128, 16, 32)),
                )
                nl.store(
                    X_out[
                        img,
                        128:256,
                        row_start : row_start + 16,
                        0:32,
                    ],
                    nl.add(
                        psum1,
                        bias_sbuf[:, 1],
                    ).reshape((128, 16, 32)),
                )

        return X_out

    elif use_fastpath_in128_out256_3x3_b4_66x66_fp16:
        X_out = nl.ndarray(
            shape=(4, 256, 64, 64),
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
        for ti in nl.affine_range(3):
            for tj in nl.affine_range(3):
                w0[:, :, ti, tj] = nisa.nc_transpose(W0_slab[:, :, ti, tj])
                w1[:, :, ti, tj] = nisa.nc_transpose(W1_slab[:, :, ti, tj])

        bias_sbuf = nl.ndarray(
            shape=(128, 2),
            dtype=bias.dtype,
            buffer=nl.sbuf,
        )
        bias_sbuf[:, 0] = nl.load(bias[0:128])
        bias_sbuf[:, 1] = nl.load(bias[128:256])

        for img in nl.sequential_range(4):
            X_half = nl.ndarray(
                shape=(128, 34, 66),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )

            # Top half: input rows 0:34 → output row blocks 0–3
            X_half[:, :, :] = nl.load(
                X[
                    img,
                    0:128,
                    0:34,
                    0:66,
                ]
            )

            for rb in nl.sequential_range(4):
                local_row = rb * 8
                row_start = rb * 8

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

                for wi in nl.affine_range(3):
                    X_pack0 = nl.ndarray(
                        shape=(128, 512),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                    X_pack1 = nl.ndarray(
                        shape=(128, 512),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )

                    X_pack0[:, :] = nisa.tensor_copy(
                        X_half[
                            :,
                            local_row + wi : local_row + wi + 8,
                            0:64,
                        ]
                    ).reshape((128, 512))

                    psum0 += nisa.nc_matmul(w0[:, :, wi, 0], X_pack0)
                    psum1 += nisa.nc_matmul(w1[:, :, wi, 0], X_pack0)

                    X_pack1[:, :] = nisa.tensor_copy(
                        X_half[
                            :,
                            local_row + wi : local_row + wi + 8,
                            1:65,
                        ]
                    ).reshape((128, 512))

                    psum0 += nisa.nc_matmul(w0[:, :, wi, 1], X_pack1)
                    psum1 += nisa.nc_matmul(w1[:, :, wi, 1], X_pack1)

                    X_pack0[:, :] = nisa.tensor_copy(
                        X_half[
                            :,
                            local_row + wi : local_row + wi + 8,
                            2:66,
                        ]
                    ).reshape((128, 512))

                    psum0 += nisa.nc_matmul(w0[:, :, wi, 2], X_pack0)
                    psum1 += nisa.nc_matmul(w1[:, :, wi, 2], X_pack0)

                nl.store(
                    X_out[
                        img,
                        0:128,
                        row_start : row_start + 8,
                        0:64,
                    ],
                    nl.add(
                        psum0,
                        bias_sbuf[:, 0],
                    ).reshape((128, 8, 64)),
                )
                nl.store(
                    X_out[
                        img,
                        128:256,
                        row_start : row_start + 8,
                        0:64,
                    ],
                    nl.add(
                        psum1,
                        bias_sbuf[:, 1],
                    ).reshape((128, 8, 64)),
                )

            # Bottom half: input rows 32:66 → output row blocks 4–7
            X_half[:, :, :] = nl.load(
                X[
                    img,
                    0:128,
                    32:66,
                    0:66,
                ]
            )

            for rb in nl.sequential_range(4):
                row_start = (rb + 4) * 8
                local_row = rb * 8

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

                for wi in nl.affine_range(3):
                    X_pack0 = nl.ndarray(
                        shape=(128, 512),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                    X_pack1 = nl.ndarray(
                        shape=(128, 512),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )

                    X_pack0[:, :] = nisa.tensor_copy(
                        X_half[
                            :,
                            local_row + wi : local_row + wi + 8,
                            0:64,
                        ]
                    ).reshape((128, 512))

                    psum0 += nisa.nc_matmul(w0[:, :, wi, 0], X_pack0)
                    psum1 += nisa.nc_matmul(w1[:, :, wi, 0], X_pack0)

                    X_pack1[:, :] = nisa.tensor_copy(
                        X_half[
                            :,
                            local_row + wi : local_row + wi + 8,
                            1:65,
                        ]
                    ).reshape((128, 512))

                    psum0 += nisa.nc_matmul(w0[:, :, wi, 1], X_pack1)
                    psum1 += nisa.nc_matmul(w1[:, :, wi, 1], X_pack1)

                    X_pack0[:, :] = nisa.tensor_copy(
                        X_half[
                            :,
                            local_row + wi : local_row + wi + 8,
                            2:66,
                        ]
                    ).reshape((128, 512))

                    psum0 += nisa.nc_matmul(w0[:, :, wi, 2], X_pack0)
                    psum1 += nisa.nc_matmul(w1[:, :, wi, 2], X_pack0)

                nl.store(
                    X_out[
                        img,
                        0:128,
                        row_start : row_start + 8,
                        0:64,
                    ],
                    nl.add(
                        psum0,
                        bias_sbuf[:, 0],
                    ).reshape((128, 8, 64)),
                )
                nl.store(
                    X_out[
                        img,
                        128:256,
                        row_start : row_start + 8,
                        0:64,
                    ],
                    nl.add(
                        psum1,
                        bias_sbuf[:, 1],
                    ).reshape((128, 8, 64)),
                )

        return X_out

    elif False:
        # in256_out256 3x3 b4 34x34: 2 c_in tiles, ping-pong X_pack inside each tile (9 taps).
        X_out = nl.ndarray(
            shape=(4, 256, 32, 32),
            dtype=X.dtype,
            buffer=nl.hbm,
        )

        X_band_first = nl.ndarray(
            shape=(128, 2, 18, 34),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )
        X_band_first[:, 0, :, :] = nl.load(
            X[
                0,
                0:128,
                0:18,
                0:34,
            ]
        )

        w = nl.ndarray(
            shape=(128, 128, 2, 2, 3, 3),
            dtype=W.dtype,
            buffer=nl.sbuf,
        )
        for c_out_tile_idx in nl.affine_range(2):
            for c_in_tile_idx in nl.affine_range(2):
                W_slab = nl.ndarray(
                    shape=(128, 128, 3, 3),
                    dtype=W.dtype,
                    buffer=nl.sbuf,
                )
                W_slab[:, :, :, :] = nl.load(
                    W[
                        c_out_tile_idx * 128 : (c_out_tile_idx + 1) * 128,
                        c_in_tile_idx * 128 : (c_in_tile_idx + 1) * 128,
                        0:3,
                        0:3,
                    ]
                )
                for i in nl.affine_range(3):
                    for j in nl.affine_range(3):
                        w[:, :, c_out_tile_idx, c_in_tile_idx, i, j] = nisa.nc_transpose(
                            W_slab[:, :, i, j],
                        )

        bias_sbuf = nl.ndarray(
            shape=(128, 2),
            dtype=bias.dtype,
            buffer=nl.sbuf,
        )
        bias_sbuf[:, 0] = nl.load(bias[0:128])
        bias_sbuf[:, 1] = nl.load(bias[128:256])

        psum0_first = nl.zeros(
            shape=(128, 512),
            dtype=nl.float32,
            buffer=nl.psum,
        )
        psum1_first = nl.zeros(
            shape=(128, 512),
            dtype=nl.float32,
            buffer=nl.psum,
        )

        X_pack0 = nl.ndarray(
            shape=(128, 512),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )
        X_pack1 = nl.ndarray(
            shape=(128, 512),
            dtype=X.dtype,
            buffer=nl.sbuf,
        )

        # First c_in tile: depth-2 ping-pong; tensor_copy(16×32).reshape(128,512) per fill.
        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 0, 0:16, 0:32]).reshape((128, 512))
        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 0, 0:16, 1:33]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 0, 0, 0], X_pack0)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 0, 0, 0], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 0, 0:16, 2:34]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 0, 0, 1], X_pack1)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 0, 0, 1], X_pack1)

        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 0, 1:17, 0:32]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 0, 0, 2], X_pack0)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 0, 0, 2], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 0, 1:17, 1:33]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 0, 1, 0], X_pack1)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 0, 1, 0], X_pack1)

        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 0, 1:17, 2:34]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 0, 1, 1], X_pack0)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 0, 1, 1], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 0, 2:18, 0:32]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 0, 1, 2], X_pack1)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 0, 1, 2], X_pack1)

        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 0, 2:18, 1:33]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 0, 2, 0], X_pack0)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 0, 2, 0], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 0, 2:18, 2:34]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 0, 2, 1], X_pack1)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 0, 2, 1], X_pack1)

        psum0_first += nisa.nc_matmul(w[:, :, 0, 0, 2, 2], X_pack0)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 0, 2, 2], X_pack0)
        X_band_first[:, 1, :, :] = nl.load(
            X[
                0,
                128:256,
                0:18,
                0:34,
            ]
        )

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 1, 0:16, 0:32]).reshape((128, 512))
        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 1, 0:16, 1:33]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 1, 0, 0], X_pack0)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 1, 0, 0], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 1, 0:16, 2:34]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 1, 0, 1], X_pack1)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 1, 0, 1], X_pack1)

        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 1, 1:17, 0:32]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 1, 0, 2], X_pack0)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 1, 0, 2], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 1, 1:17, 1:33]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 1, 1, 0], X_pack1)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 1, 1, 0], X_pack1)

        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 1, 1:17, 2:34]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 1, 1, 1], X_pack0)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 1, 1, 1], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 1, 2:18, 0:32]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 1, 1, 2], X_pack1)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 1, 1, 2], X_pack1)

        X_pack1[:, :] = nisa.tensor_copy(X_band_first[:, 1, 2:18, 1:33]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 1, 2, 0], X_pack0)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 1, 2, 0], X_pack0)

        X_pack0[:, :] = nisa.tensor_copy(X_band_first[:, 1, 2:18, 2:34]).reshape((128, 512))
        psum0_first += nisa.nc_matmul(w[:, :, 0, 1, 2, 1], X_pack1)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 1, 2, 1], X_pack1)

        psum0_first += nisa.nc_matmul(w[:, :, 0, 1, 2, 2], X_pack0)
        psum1_first += nisa.nc_matmul(w[:, :, 1, 1, 2, 2], X_pack0)
        nl.store(
            X_out[
                0,
                0:128,
                0:16,
                0:32,
            ],
            nl.add(
                psum0_first,
                bias_sbuf[:, 0],
            ).reshape((128, 16, 32)),
        )
        nl.store(
            X_out[
                0,
                128:256,
                0:16,
                0:32,
            ],
            nl.add(
                psum1_first,
                bias_sbuf[:, 1],
            ).reshape((128, 16, 32)),
        )

        for rb in nl.sequential_range(1, 2):
            row_start = rb * 16

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

            X_pack0 = nl.ndarray(
                shape=(128, 512),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )
            X_pack1 = nl.ndarray(
                shape=(128, 512),
                dtype=X.dtype,
                buffer=nl.sbuf,
            )

            for c_in_tile_idx in nl.sequential_range(0, 2):
                X_band = nl.ndarray(
                    shape=(128, 18, 34),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                X_band[:, :, :] = nl.load(
                    X[
                        0,
                        c_in_tile_idx * 128 : (c_in_tile_idx + 1) * 128,
                        row_start : row_start + 18,
                        0:34,
                    ]
                )

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 0:16, 0:32]).reshape((128, 512))
                X_pack1[:, :] = nisa.tensor_copy(X_band[:, 0:16, 1:33]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 0, 0], X_pack0)
                psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 0, 0], X_pack0)

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 0:16, 2:34]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 0, 1], X_pack1)
                psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 0, 1], X_pack1)

                X_pack1[:, :] = nisa.tensor_copy(X_band[:, 1:17, 0:32]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 0, 2], X_pack0)
                psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 0, 2], X_pack0)

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 1:17, 1:33]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 1, 0], X_pack1)
                psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 1, 0], X_pack1)

                X_pack1[:, :] = nisa.tensor_copy(X_band[:, 1:17, 2:34]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 1, 1], X_pack0)
                psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 1, 1], X_pack0)

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 2:18, 0:32]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 1, 2], X_pack1)
                psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 1, 2], X_pack1)

                X_pack1[:, :] = nisa.tensor_copy(X_band[:, 2:18, 1:33]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 2, 0], X_pack0)
                psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 2, 0], X_pack0)

                X_pack0[:, :] = nisa.tensor_copy(X_band[:, 2:18, 2:34]).reshape((128, 512))
                psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 2, 1], X_pack1)
                psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 2, 1], X_pack1)

                psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 2, 2], X_pack0)
                psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 2, 2], X_pack0)
            nl.store(
                X_out[
                    0,
                    0:128,
                    row_start : row_start + 16,
                    0:32,
                ],
                nl.add(
                    psum0,
                    bias_sbuf[:, 0],
                ).reshape((128, 16, 32)),
            )
            nl.store(
                X_out[
                    0,
                    128:256,
                    row_start : row_start + 16,
                    0:32,
                ],
                nl.add(
                    psum1,
                    bias_sbuf[:, 1],
                ).reshape((128, 16, 32)),
            )

        for img in nl.sequential_range(1, 4):
            for rb in nl.sequential_range(0, 2):
                row_start = rb * 16

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

                X_pack0 = nl.ndarray(
                    shape=(128, 512),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )
                X_pack1 = nl.ndarray(
                    shape=(128, 512),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )

                for c_in_tile_idx in nl.sequential_range(0, 2):
                    X_band = nl.ndarray(
                        shape=(128, 18, 34),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )
                    X_band[:, :, :] = nl.load(
                        X[
                            img,
                            c_in_tile_idx * 128 : (c_in_tile_idx + 1) * 128,
                            row_start : row_start + 18,
                            0:34,
                        ]
                    )

                    X_pack0[:, :] = nisa.tensor_copy(X_band[:, 0:16, 0:32]).reshape((128, 512))
                    X_pack1[:, :] = nisa.tensor_copy(X_band[:, 0:16, 1:33]).reshape((128, 512))
                    psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 0, 0], X_pack0)
                    psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 0, 0], X_pack0)

                    X_pack0[:, :] = nisa.tensor_copy(X_band[:, 0:16, 2:34]).reshape((128, 512))
                    psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 0, 1], X_pack1)
                    psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 0, 1], X_pack1)

                    X_pack1[:, :] = nisa.tensor_copy(X_band[:, 1:17, 0:32]).reshape((128, 512))
                    psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 0, 2], X_pack0)
                    psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 0, 2], X_pack0)

                    X_pack0[:, :] = nisa.tensor_copy(X_band[:, 1:17, 1:33]).reshape((128, 512))
                    psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 1, 0], X_pack1)
                    psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 1, 0], X_pack1)

                    X_pack1[:, :] = nisa.tensor_copy(X_band[:, 1:17, 2:34]).reshape((128, 512))
                    psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 1, 1], X_pack0)
                    psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 1, 1], X_pack0)

                    X_pack0[:, :] = nisa.tensor_copy(X_band[:, 2:18, 0:32]).reshape((128, 512))
                    psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 1, 2], X_pack1)
                    psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 1, 2], X_pack1)

                    X_pack1[:, :] = nisa.tensor_copy(X_band[:, 2:18, 1:33]).reshape((128, 512))
                    psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 2, 0], X_pack0)
                    psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 2, 0], X_pack0)

                    X_pack0[:, :] = nisa.tensor_copy(X_band[:, 2:18, 2:34]).reshape((128, 512))
                    psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 2, 1], X_pack1)
                    psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 2, 1], X_pack1)

                    psum0 += nisa.nc_matmul(w[:, :, 0, c_in_tile_idx, 2, 2], X_pack0)
                    psum1 += nisa.nc_matmul(w[:, :, 1, c_in_tile_idx, 2, 2], X_pack0)
                nl.store(
                    X_out[
                        img,
                        0:128,
                        row_start : row_start + 16,
                        0:32,
                    ],
                    nl.add(
                        psum0,
                        bias_sbuf[:, 0],
                    ).reshape((128, 16, 32)),
                )
                nl.store(
                    X_out[
                        img,
                        128:256,
                        row_start : row_start + 16,
                        0:32,
                    ],
                    nl.add(
                        psum1,
                        bias_sbuf[:, 1],
                    ).reshape((128, 16, 32)),
                )

        return X_out



    # Tiling dimensions
    c_in_tile = nl.tile_size.pmax                       # partition dim = 128
    c_out_tile = nl.tile_size.gemm_stationary_fmax      # tensor engine free dim = 128
    n_tiles_c_in = in_channels // c_in_tile
    n_tiles_c_out = out_channels // c_out_tile

  
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
                    X_packed_first_block[:, :] = nisa.tensor_copy(
                        X_bands_first[:, c_in_tile_idx, i : i + block_rows, j : j + out_width]
                    ).reshape((c_in_tile, F_m))
                    psum0_first += nisa.nc_matmul(
                        w[:, :, 0, c_in_tile_idx, i, j],
                        X_packed_first_block,
                    )
                    psum1_first += nisa.nc_matmul(
                        w[:, :, 1, c_in_tile_idx, i, j],
                        X_packed_first_block,
                    )

        nl.store(
            X_out[
                0,
                0 * c_out_tile : 1 * c_out_tile,
                0 : block_rows,
                :,
            ],
            nl.add(
                psum0_first,
                bias_sbuf[:, 0],
            ).reshape((c_out_tile, block_rows, out_width)),
        )

        nl.store(
            X_out[
                0,
                1 * c_out_tile : 2 * c_out_tile,
                0 : block_rows,
                :,
            ],
            nl.add(
                psum1_first,
                bias_sbuf[:, 1],
            ).reshape((c_out_tile, block_rows, out_width)),
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
                        X_packed_first_block[:, :] = nisa.tensor_copy(
                            X_bands_first[:, c_in_tile_idx, i : i + block_rows, j : j + out_width]
                        ).reshape((c_in_tile, F_m))
                        psum_packed_first += nisa.nc_matmul(
                            w[:, :, c_out_idx, c_in_tile_idx, i, j],
                            X_packed_first_block,
                        )

            nl.store(
                X_out[
                    0,
                    c_out_idx * c_out_tile : (c_out_idx + 1) * c_out_tile,
                    0 : block_rows,
                    :,
                ],
                nl.add(
                    psum_packed_first,
                    bias_sbuf[:, c_out_idx],
                ).reshape((c_out_tile, block_rows, out_width)),
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
                        X_packed_row[:, :] = nisa.tensor_copy(
                            X_bands[:, c_in_tile_idx, i : i + block_rows, j : j + out_width]
                        ).reshape((c_in_tile, F_m))
                        psum0_row += nisa.nc_matmul(
                            w[:, :, 0, c_in_tile_idx, i, j],
                            X_packed_row,
                        )
                        psum1_row += nisa.nc_matmul(
                            w[:, :, 1, c_in_tile_idx, i, j],
                            X_packed_row,
                        )

            nl.store(
                X_out[
                    0,
                    0 * c_out_tile : 1 * c_out_tile,
                    row_start : row_start + block_rows,
                    :,
                ],
                nl.add(
                    psum0_row,
                    bias_sbuf[:, 0],
                ).reshape((c_out_tile, block_rows, out_width)),
            )

            nl.store(
                X_out[
                    0,
                    1 * c_out_tile : 2 * c_out_tile,
                    row_start : row_start + block_rows,
                    :,
                ],
                nl.add(
                    psum1_row,
                    bias_sbuf[:, 1],
                ).reshape((c_out_tile, block_rows, out_width)),
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
                            X_packed_row[:, :] = nisa.tensor_copy(
                                X_bands[:, c_in_tile_idx, i : i + block_rows, j : j + out_width]
                            ).reshape((c_in_tile, F_m))
                            psum_packed_row += nisa.nc_matmul(
                                w[:, :, c_out_idx, c_in_tile_idx, i, j],
                                X_packed_row,
                            )

                nl.store(
                    X_out[
                        0,
                        c_out_idx * c_out_tile : (c_out_idx + 1) * c_out_tile,
                        row_start : row_start + block_rows,
                        :,
                    ],
                    nl.add(
                        psum_packed_row,
                        bias_sbuf[:, c_out_idx],
                    ).reshape((c_out_tile, block_rows, out_width)),
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
                            X_packed_img[:, :] = nisa.tensor_copy(
                                X_bands[:, c_in_tile_idx, i : i + block_rows, j : j + out_width]
                            ).reshape((c_in_tile, F_m))
                            psum0_img += nisa.nc_matmul(
                                w[:, :, 0, c_in_tile_idx, i, j],
                                X_packed_img,
                            )
                            psum1_img += nisa.nc_matmul(
                                w[:, :, 1, c_in_tile_idx, i, j],
                                X_packed_img,
                            )

                nl.store(
                    X_out[
                        img,
                        0 * c_out_tile : 1 * c_out_tile,
                        row_start : row_start + block_rows,
                        :,
                    ],
                    nl.add(
                        psum0_img,
                        bias_sbuf[:, 0],
                    ).reshape((c_out_tile, block_rows, out_width)),
                )

                nl.store(
                    X_out[
                        img,
                        1 * c_out_tile : 2 * c_out_tile,
                        row_start : row_start + block_rows,
                        :,
                    ],
                    nl.add(
                        psum1_img,
                        bias_sbuf[:, 1],
                    ).reshape((c_out_tile, block_rows, out_width)),
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
                                X_packed_img[:, :] = nisa.tensor_copy(
                                    X_bands[:, c_in_tile_idx, i : i + block_rows, j : j + out_width]
                                ).reshape((c_in_tile, F_m))
                                psum_packed_img += nisa.nc_matmul(
                                    w[:, :, c_out_idx, c_in_tile_idx, i, j],
                                    X_packed_img,
                                )

                    nl.store(
                        X_out[
                            img,
                            c_out_idx * c_out_tile : (c_out_idx + 1) * c_out_tile,
                            row_start : row_start + block_rows,
                            :,
                        ],
                        nl.add(
                            psum_packed_img,
                            bias_sbuf[:, c_out_idx],
                        ).reshape((c_out_tile, block_rows, out_width)),
                    )

    return X_out
