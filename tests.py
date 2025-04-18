import torch
import numpy as np  # in your code this might be cupy; we assume operations and API are similar
from engine import ValueTensor

# Helper: check that two arrays are close.
def assert_allclose(a, b, tol=1e-4, op=""):
    if not np.allclose(a, b, atol=tol):
        raise AssertionError(f"{op} mismatch\nExpected:\n{b}\nGot:\n{a}")
        
# ----------------------------
# Test Functions for binary operations
# ----------------------------

def test_add():
    # Test addition operation
    a_val = np.array([[1.0, 2.0], [3.0, 4.0]])
    b_val = np.array([[5.0, 6.0], [7.0, 8.0]])
    
    # Using ValueTensor
    a = ValueTensor(a_val)
    b = ValueTensor(b_val)
    out = a + b
    # Use a sum loss to drive gradient computation
    loss = out.sum(keepdims=True)
    loss.backward()
    
    # Torch version:
    ta = torch.tensor(a_val, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(b_val, dtype=torch.float32, requires_grad=True)
    tout = ta + tb
    loss_t = tout.sum()
    loss_t.backward()
    
    # Compare forward outputs:
    assert_allclose(out.data, tout.detach().numpy(), tol=1e-4, op="Add forward")
    # Compare gradients for a and b:
    assert_allclose(a.grad, ta.grad.numpy(), tol=1e-4, op="Add grad a")
    assert_allclose(b.grad, tb.grad.numpy(), tol=1e-4, op="Add grad b")
    print("Addition test passed.")

def test_mul():
    # Test elementwise multiplication
    a_val = np.array([[1.0, 2.0], [3.0, 4.0]])
    b_val = np.array([[2.0, 0.5], [1.0, 3.0]])
    
    a = ValueTensor(a_val)
    b = ValueTensor(b_val)
    out = a * b
    loss = out.sum(keepdims=True)
    loss.backward()
    
    ta = torch.tensor(a_val, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(b_val, dtype=torch.float32, requires_grad=True)
    tout = ta * tb
    loss_t = tout.sum()
    loss_t.backward()
    
    assert_allclose(out.data, tout.detach().numpy(), tol=1e-4, op="Mul forward")
    assert_allclose(a.grad, ta.grad.numpy(), tol=1e-4, op="Mul grad a")
    assert_allclose(b.grad, tb.grad.numpy(), tol=1e-4, op="Mul grad b")
    print("Multiplication test passed.")

def test_matmul():
    # Test matrix multiplication
    a_val = np.random.randn(10, 3)
    b_val = np.random.randn(3, 2)
    
    a = ValueTensor(a_val)
    b = ValueTensor(b_val)
    out = a @ b  # custom matmul
    loss = out.sum(keepdims=True)
    loss.backward()
    
    ta = torch.tensor(a_val, dtype=torch.float32, requires_grad=True)
    tb = torch.tensor(b_val, dtype=torch.float32, requires_grad=True)
    tout = ta @ tb
    loss_t = tout.sum()
    loss_t.backward()
    
    assert_allclose(out.data, tout.detach().numpy(), tol=1e-4, op="Matmul forward")
    # Compare gradients on left parameter:
    assert_allclose(a.grad, ta.grad.numpy(), tol=1e-4, op="Matmul grad a")
    # Compare gradients on right parameter:
    assert_allclose(b.grad, tb.grad.numpy(), tol=1e-4, op="Matmul grad b")
    print("Matmul test passed.")

def test_pow():
    # Test exponentiation with a scalar exponent
    a_val = np.array([[2.0, 3.0], [4.0, 5.0]])
    exp_val = 3
    a = ValueTensor(a_val)
    out = a ** exp_val
    loss = out.sum(keepdims=True)
    loss.backward()
    
    ta = torch.tensor(a_val, dtype=torch.float32, requires_grad=True)
    tout = ta ** exp_val
    loss_t = tout.sum()
    loss_t.backward()
    
    assert_allclose(out.data, tout.detach().numpy(), tol=1e-4, op="Pow forward")
    assert_allclose(a.grad, ta.grad.numpy(), tol=1e-4, op="Pow grad")
    print("Power test passed.")

# ----------------------------
# Test Functions for activations
# ----------------------------

def test_relu():
    x_val = np.array([[-1.0, 2.0], [3.0, -4.0]])
    x = ValueTensor(x_val)
    out = x.relu()
    loss = out.sum(keepdims=True)
    loss.backward()
    
    tx = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    tout = torch.nn.functional.relu(tx)
    loss_t = tout.sum()
    loss_t.backward()
    
    assert_allclose(out.data, tout.detach().numpy(), tol=1e-4, op="ReLU forward")
    assert_allclose(x.grad, tx.grad.numpy(), tol=1e-4, op="ReLU grad")
    print("ReLU test passed.")

def test_tanh():
    x_val = np.array([[-1.0, 0.0], [0.5, 2.0]])
    x = ValueTensor(x_val)
    out = x.tanh()
    loss = out.sum(keepdims=True)
    loss.backward()
    
    tx = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    tout = torch.tanh(tx)
    loss_t = tout.sum()
    loss_t.backward()
    
    assert_allclose(out.data, tout.detach().numpy(), tol=1e-4, op="Tanh forward")
    assert_allclose(x.grad, tx.grad.numpy(), tol=1e-4, op="Tanh grad")
    print("Tanh test passed.")

def test_sigmoid():
    x_val = np.array([[-1.0, 0.0], [0.5, 2.0]])
    x = ValueTensor(x_val)
    out = x.sigmoid()
    loss = out.sum(keepdims=True)
    loss.backward()
    
    tx = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    tout = torch.sigmoid(tx)
    loss_t = tout.sum()
    loss_t.backward()
    
    assert_allclose(out.data, tout.detach().numpy(), tol=1e-4, op="Sigmoid forward")
    assert_allclose(x.grad, tx.grad.numpy(), tol=1e-4, op="Sigmoid grad")
    print("Sigmoid test passed.")

def test_exp():
    x_val = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = ValueTensor(x_val)
    out = x.exp()
    loss = out.sum(keepdims=True)
    loss.backward()
    
    tx = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    tout = torch.exp(tx)
    loss_t = tout.sum()
    loss_t.backward()
    
    assert_allclose(out.data, tout.detach().numpy(), tol=1e-4, op="Exp forward")
    assert_allclose(x.grad, tx.grad.numpy(), tol=1e-4, op="Exp grad")
    print("Exp test passed.")

def test_log_forward_backward():
    # Forward: log
    data = np.array([1.0, 2.0, 4.0])
    x = ValueTensor(data)
    y = x.log()
    expected_forward = np.log(data)
    assert np.allclose(y.data, expected_forward), f"Forward log failed: {y.data} vs {expected_forward}"

    # Backward: gradient should be 1/x
    # Use sum to propagate a gradient of 1 for each element
    loss = y.sum(axis=0, keepdims=False)
    x.zero_grad()
    loss.backward()
    expected_grad = 1.0 / data
    assert np.allclose(x.grad, expected_grad), f"Backward log failed: {x.grad} vs {expected_grad}"
    print("Log test passed.")



def test_softmax_forward_backward():
    # Forward: softmax
    vec = np.array([1.0, 2.0, 3.0])
    z = ValueTensor(vec)
    s = z.softmax()
    # Expected softmax
    shift = vec - np.max(vec)
    exp_shift = np.exp(shift)
    expected_forward = exp_shift / exp_shift.sum()
    assert np.allclose(s.data, expected_forward), f"Forward softmax failed: {s.data} vs {expected_forward}"

    # Backward: use an upstream gradient g
    g = np.array([0.1, 0.2, 0.3])
    # define a scalar loss = sum(s * g)
    loss = (s * g).sum(axis=0, keepdims=False)
    z.zero_grad()
    loss.backward()
    # Expected gradient: s * (g - dot(g, s))
    D = np.dot(g, expected_forward)
    expected_grad = expected_forward * (g - D)
    assert np.allclose(z.grad, expected_grad), f"Backward softmax failed: {z.grad} vs {expected_grad}"
    print("Softmax test passed.")


# ----------------------------
# Test Functions for reductions and shape operations
# ----------------------------

def test_sum():
    x_val = np.random.randn(4, 5)
    x = ValueTensor(x_val)
    out = x.sum(axis=0, keepdims=True)
    loss = out.sum(keepdims=True)
    loss.backward()
    
    tx = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    tout = torch.sum(tx, dim=0, keepdim=True)
    loss_t = tout.sum()
    loss_t.backward()
    
    assert_allclose(out.data, tout.detach().numpy(), tol=1e-4, op="Sum forward")
    assert_allclose(x.grad, tx.grad.numpy(), tol=1e-4, op="Sum grad")
    print("Sum test passed.")

def test_mean():
    x_val = np.random.randn(4, 5)
    x = ValueTensor(x_val)
    out = x.mean(axis=0, keepdims=True)
    loss = out.sum(keepdims=True)
    loss.backward()
    
    tx = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    tout = torch.mean(tx, dim=0, keepdim=True)
    loss_t = tout.sum()
    loss_t.backward()
    
    assert_allclose(out.data, tout.detach().numpy(), tol=1e-4, op="Mean forward")
    assert_allclose(x.grad, tx.grad.numpy(), tol=1e-4, op="Mean grad")
    print("Mean test passed.")

def test_max():
    # Use a case with possible ties
    x_val = np.array([[1, 3, 2],
                      [3, 3, 1],
                      [0, 2, 4]])
    x = ValueTensor(x_val)
    out = x.max(axis=0, keepdims=True)
    loss = out.sum(keepdims=True)
    loss.backward()
    
    tx = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    # For torch, get only the maximum values (ignoring indices)
    tout, _ = torch.max(tx, dim=0, keepdim=True)
    loss_t = tout.sum()
    loss_t.backward()
    
    assert_allclose(out.data, tout.detach().numpy(), tol=1e-4, op="Max forward")
    assert_allclose(x.grad, tx.grad.numpy(), tol=1e-4, op="Max grad")
    print("Max test passed.")

def test_reshape():
    x_val = np.random.randn(2, 3, 4)
    x = ValueTensor(x_val)
    new_shape = (6, 4)
    out = x.reshape(new_shape)
    loss = out.sum(keepdims=True)
    loss.backward()
    
    tx = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    tout = tx.reshape(new_shape)
    loss_t = tout.sum()
    loss_t.backward()
    
    assert_allclose(out.data, tout.detach().numpy(), tol=1e-4, op="Reshape forward")
    assert_allclose(x.grad, tx.grad.numpy(), tol=1e-4, op="Reshape grad")
    print("Reshape test passed.")

def test_transpose():
    x_val = np.random.randn(2, 3, 4)
    # Let’s transpose to (4, 3, 2) (here we simply reverse the axes)
    axes = (2, 1, 0)
    x = ValueTensor(x_val)
    out = x.transpose(axes)
    loss = out.sum(keepdims=True)
    loss.backward()
    
    tx = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    tout = torch.transpose(tx, 0, 2)  # for 3D, a full permutation is a bit more complex; here we switch axis0 and axis2
    # Since our custom transpose uses np.transpose with axes, we mimic that with torch.permute:
    tout = tx.permute(2, 1, 0)
    loss_t = tout.sum()
    loss_t.backward()
    
    assert_allclose(out.data, tout.detach().numpy(), tol=1e-4, op="Transpose forward")
    assert_allclose(x.grad, tx.grad.numpy(), tol=1e-4, op="Transpose grad")
    print("Transpose test passed.")

# ----------------------------
# Run All Tests
# ----------------------------

def run_all_tests():
    test_add()
    test_mul()
    test_matmul()
    test_pow()
    test_relu()
    test_tanh()
    test_sigmoid()
    test_exp()
    test_log_forward_backward()
    test_softmax_forward_backward()
    test_sum()
    test_mean()
    test_max() # currently pytorch does not distribute gradients in case of a tie
    test_reshape()
    test_transpose()
    print("All tests passed successfully!")

if __name__ == '__main__':
    run_all_tests()
