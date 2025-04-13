import numpy as np
# define a class to handle tensors
class ValueTensor:
  def __init__(self, data, _children = (), _op = ""):
    self.data = np.array(data, dtype = float) if not isinstance(data, np.ndarray) else data.astype(float)
    self.grad = np.zeros_like(data, dtype = float)
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.shape = self.data.shape

  def __repr__(self):
    return f"Tensor(data={self.data})"

  def __add__(self, other):
    other = other if isinstance(other, ValueTensor) else ValueTensor(other)
    out = ValueTensor(self.data + other.data, (self, other), "+")
    def _backward():
      self.grad += np.ones_like(self.data) * out.grad
      other.grad = other.grad + self.unbroadcast_to_shape(out.grad, other.data.shape) # not using += as numpy doesnt implicitly broadcasts
    out._backward = _backward
    return out

  def __mul__(self, other):
    other = other if isinstance(other, ValueTensor) else ValueTensor(other)
    out = ValueTensor(self.data * other.data, (self, other), "*")
    # write explaination here
    # remark: the .T is not needed
    def _backward():
      self.grad += out.grad * other.data
      other.grad = other.grad + self.unbroadcast_to_shape(self.data * out.grad, other.data.shape)
    out._backward = _backward
    return out

  def __matmul__(self, other):
    other = other if isinstance(other, ValueTensor) else ValueTensor(other)
    out = ValueTensor(self.data @ other.data, (self, other), "@")
    # write explaination here
    def _backward():
      self.grad = self.grad + out.grad @ other.data.T
      other.grad = other.grad + self.unbroadcast_to_shape(self.data.T @ out.grad, other.data.shape) # failts without broadcast reversing for bias terms in MLP
    out._backward = _backward
    return out

  def __rmul__(self, other):
    return self * other

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return other + (-self)

  def __truediv__(self, other):
    other = other if isinstance(other, ValueTensor) else ValueTensor(other)
    return self * other ** -1


  def __rtruediv__(self, other):
    other = other if isinstance(other, ValueTensor) else ValueTensor(other)
    return other/ self

  def __radd__(self, other):
    return self + other

  def __neg__(self):
    return self * -1

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only scaler exponents supported"
    out = ValueTensor(self.data ** other, (self, ), f"**{other}")
    def _backward():
      self.grad = self.grad +  self.unbroadcast_to_shape(out.grad * other * (self.data ** (other - 1)), self.data.shape)
    out._backward = _backward
    return out

  # activations
  def relu(self):
    out = ValueTensor(np.maximum(self.data, 0), (self,), "relu")
    def _backward():
      self.grad += (self.data > 0) * out.grad # grad is 0/1
    out._backward = _backward
    return out

  def tanh(self):
    t = np.tanh(self.data)
    out = ValueTensor(t, (self,), "tanh")
    def _backward():
      self.grad += (1- t**2) * out.grad
    out._backward = _backward
    return out

  def sigmoid(self):
    t = 1 / (1 + np.exp(-self.data))
    out = ValueTensor(t, (self,), "sigmoid")
    def _backward():
      self.grad += t * (1 - t) * out.grad
    out._backward = _backward
    return out

  def exp(self):
    out = ValueTensor(np.exp(self.data), (self,), "exp")
    def _backward():
      self.grad += out.grad * out.data
    out._backward = _backward
    return out

  # reduction operations
  def sum(self, axis = 0, keepdims=True):
    out = ValueTensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum")
    def _backward():
      self.grad += self.unbroadcast_to_shape(out.grad, self.data.shape)
    out._backward = _backward
    return out

  def mean(self, axis = None, keepdims=True):
    out = ValueTensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), "mean")
    def _backward():
      scale = np.prod(self.data.shape) if axis is None else self.data.shape[axis] # this gets the value to divisor for mean calculation
      self.grad += self.unbroadcast_to_shape(out.grad / scale, self.data.shape)
    out._backward = _backward
    return out
  
  def max(self, axis=None, keepdims=True):
    # Forward: compute maximum along the given axis.
    out_data = np.max(self.data, axis=axis, keepdims=keepdims)
    out = ValueTensor(out_data, (self,), "max")
    
    def _backward():
        # First, expand the output grad to the shape of self.data.
        grad_unbroadcast = self.unbroadcast_to_shape(out.grad, self.data.shape)
        
        # Create the mask that selects only the first occurrence of the maximum.
        if axis is None:
            # Global maximum: flatten self.data.
            flat_data = self.data.ravel()
            idx = np.argmax(flat_data)  # index of first maximum in flattened array
            mask = np.zeros_like(flat_data, dtype=float)
            mask[idx] = 1.0
            mask = mask.reshape(self.data.shape)
        else:
            # Compute argmax along the given axis, keeping dimensions
            argmax = np.argmax(self.data, axis=axis, keepdims=True)
            # Create a mask of zeros with same shape as self.data.
            mask = np.zeros_like(self.data, dtype=float)
            # Iterate over all indices in self.data.
            it = np.nditer(self.data, flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index  # index tuple for the current element
                # Build an index for argmax.
                # For the axis being reduced, we always use index 0 because argmax has that dimension of size 1.
                idx_argmax = list(idx)
                idx_argmax[axis] = 0
                # If the coordinate along the given axis matches the stored argmax value for that slice, mark it.
                if idx[axis] == argmax[tuple(idx_argmax)]:
                    mask[idx] = 1.0
                it.iternext()
        # Propagate gradient only to the selected positions.
        self.grad += mask * grad_unbroadcast
    out._backward = _backward
    return out

  def reshape(self, shape):
    out = ValueTensor(self.data.reshape(shape), (self,), "reshape")
    def _backward():
      self.grad += out.grad.reshape(self.data.shape)
    out._backward = _backward
    return out

  def transpose(self, axes=None):
    out = ValueTensor(np.transpose(self.data, axes=axes), (self,), "transpose")
    def _backward():
      reversed_axes = np.argsort(axes) if axes is not None else None
      self.grad += np.transpose(out.grad, reversed_axes)
    out._backward = _backward
    return out

  # utilities
  def zero_grad(self):
    # set the gradients to zero
    self.grad = np.zeros_like(self.data, dtype=float) # change from self.data.shape

  def detach(self):
    pass


  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad = np.ones_like(self.data)
    for node in reversed(topo):
      node._backward()

  def unbroadcast_to_shape(self, grad, shape):
    """converts the shape of grad to desired shape by summing"""
    while len(shape) < len(grad.shape):
      shape = (1,) + shape # padding shape with 1
    for axis, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
      if s_dim ==1 and g_dim >1:
        grad = grad.sum(axis=axis, keepdims=True)
    grad = np.broadcast_to(grad, shape)
    return grad

 