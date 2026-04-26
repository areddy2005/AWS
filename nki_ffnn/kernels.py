import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

from utils import BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from matmul_kernels import nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_

@nki.jit
def nki_transpose(in_tensor):
    """NKI kernel to transpose a 2D tensor.

    Args:
        in_tensor: an input tensor of shape [#rows, #cols]

    Returns:
        out_tensor: an output (transposed) tensor of shape [#cols, #rows]
    """
    i_rows, i_cols = in_tensor.shape
    o_rows, o_cols = i_cols, i_rows

    out_tensor = nl.ndarray((o_rows, o_cols), dtype=in_tensor.dtype, buffer=nl.hbm)

    # YOUR CODE HERE
    P_max = nl.tile_size.pmax
    for i in nl.affine_range(i_rows//P_max):
        for j in nl.affine_range(i_cols//P_max):
            i_start = i*P_max
            j_start = j*P_max
            tile = nl.load_transpose2d(in_tensor[i_start:i_start+P_max,j_start:j_start+P_max])
            nl.store(out_tensor[j_start:j_start+P_max,i_start:i_start+P_max],tile)
    return out_tensor

@nki.jit
def nki_bias_add_act(A, b, act='relu'):
    """NKI kernel to add a bias vector to each row of a 2D tensor, and apply activation.

    Args:
        A: an input tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
        b: a bias vector of shape [1, HIDDEN_SIZE]
        act: an activation function to apply (e.g., 'relu', 'softmax')
    Returns:
        result: the resulting output tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
    """
    # Gather input shapes
    BATCH_SIZE, HIDDEN_SIZE = A.shape
    _, HIDDEN_SIZE_ = b.shape
    assert HIDDEN_SIZE == HIDDEN_SIZE_, "A and b must have the same HIDDEN_SIZE"

    # Create an output tensor
    result = nl.ndarray((BATCH_SIZE, HIDDEN_SIZE), dtype=A.dtype, buffer=nl.hbm)

    # YOUR CODE HERE
    p_max = nl.tile_size.pmax

    for i in nl.affine_range(BATCH_SIZE // p_max):
        a_tile = nl.load(A[i*p_max : (i+1)*p_max, 0:HIDDEN_SIZE])
        b_tile = nl.load(b[0:1, 0:HIDDEN_SIZE])
        data = nl.add(a_tile, b_tile)
        if act == 'relu':
            data = nl.relu(data)
        else:
            row_max = nl.max(data, axis=1)
            data_stable = nl.subtract(data, row_max)
            exps = nl.exp(data_stable)
            row_sum = nl.sum(exps, axis=1)
            data = nl.divide(exps, row_sum)
        nl.store(result[i*p_max : (i+1)*p_max, 0:HIDDEN_SIZE], data)
    return result

@nki.jit
def nki_forward(
    X,
    W1,
    b1,
    W2,
    b2,
    matmul_kernel='tiled'
):
  """NKI kernel to compute the forward pass of the feedforward neural network with 1 hidden layer.

  Args:
      X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
      W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
      b1: the bias vector of shape [HIDDEN_SIZE]
      W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
      b2: the bias vector of shape [OUTPUT_SIZE]
  Returns:
      probs: the resulting probability output tensor of shape [BATCH_SIZE, OUTPUT_SIZE]
  
  Option:
      matmul_kernel: the matrix multiplication kernel to use 
        - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'
  """
  if matmul_kernel == 'tiled':
    nki_matmul = nki_matmul_tiled_
  elif matmul_kernel == 'hoist_load':
    nki_matmul = nki_matmul_hoist_load_
  elif matmul_kernel == 'block_free_dimension':
    nki_matmul = nki_matmul_block_free_dimension_
  elif matmul_kernel == 'fully_optimized':
    nki_matmul = nki_matmul_fully_optimized_
  else:
    raise ValueError(f"Unsupported matmul kernel: {matmul_kernel}")

  # Layer 1
  # YOUR CODE HERE  
  Xt = nki_transpose(X)
  L1 = nki_matmul(Xt,W1)
  L1_act = nki_bias_add_act(L1,b1)
  # Layer 2 (output)
  # YOUR CODE HERE
  L1t = nki_transpose(L1_act)
  L2 = nki_matmul(L1t,W2)
  probs = nki_bias_add_act(L2,b2,act = "softmax")


  return probs


@nki.jit
def nki_predict(
    X,
    W1,
    b1,
    W2,
    b2,
    matmul_kernel='tiled'
):
  """NKI kernel run forward pass and predict the classes of the input tensor.

  Args:
      X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
      W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
      b1: the bias vector of shape [HIDDEN_SIZE]
      W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
      b2: the bias vector of shape [OUTPUT_SIZE]
  Returns:
      predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
  
  Option:
      matmul_kernel: the matrix multiplication kernel to use 
        - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'

  Returns:
      predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
  """
  probs = nki_forward(X,W1,b1,W2,b2,matmul_kernel = matmul_kernel)
  BATCH_SIZE, OUTPUT_SIZE = probs.shape
  predictions = nl.ndarray((BATCH_SIZE,), dtype=np.int32, buffer=nl.hbm)

  # YOUR CODE HERE
  p_max = nl.tile_size.pmax
  for i in nl.affine_range(BATCH_SIZE//p_max):
      data_tile = nl.load(probs[i*p_max:(i+1)*p_max,0:OUTPUT_SIZE])
      top8_vals = nisa.max8(src = data_tile)
      top8_idx = nisa.nc_find_index8(data = data_tile,vals = top8_vals)
      argmax_idx = top8_idx[:, 0:1]
      nl.store(predictions[i*p_max:(i+1)*p_max],argmax_idx)

  return predictions
