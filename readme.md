# Nanograd: A Minimal Autograd Engine for Tensors

**Nanograd** is a lightweight, NumPy/CuPy-based autodiff engine inspired by Karpathy's [micrograd](https://github.com/karpathy/micrograd), but extended to support **tensors**, **broadcasting**, **matrix operations**, and **modular neural networks**.

This library is built from scratch to provide a clear, minimal implementation of reverse-mode automatic differentiation on multidimensional arrays (tensors). It's useful for educational purposes, debugging autograd mechanics, and building intuition for how frameworks like PyTorch and JAX work internally.

---

## Features
- Tensor support: Operate on N-dimensional arrays (NumPy or CuPy)
- Reverse-mode autodiff (backpropagation)
- Broadcasting support (via `unbroadcast_to_shape`)
- Basic math operations: `+`, `-`, `*`, `/`, `**`, `@`
- Reductions: `sum`, `mean`, `max`
- Nonlinearities: `relu`, `tanh`, `sigmoid`, `exp`
- Shape manipulation: `reshape`, `transpose`
- Custom MLP + Linear layer support
- Graph visualization using Graphviz with statistics & color-coding
- Pluggable backend: NumPy (CPU) or CuPy (GPU)

---

## Installation
```bash
pip install numpy graphviz
# Optional (for GPU):
pip install cupy-cuda11x  # replace with your CUDA version
```

---

## Getting Started

### Define Values and Compute Gradients
```python
from engine import ValueTensor
import numpy as np

x = ValueTensor(np.array([[1.0, 2.0]]))
w = ValueTensor(np.array([[2.0], [3.0]]))
b = ValueTensor(np.array([[1.0]]))

out = x @ w + b
y = out.relu().sum()
y.backward()

print("Output:", out.data)
print("Gradient w.r.t w:", w.grad)
```

### Build a Multi-Layer Perceptron (MLP)
```python
from model import MLP

mlp = MLP([2, 8, 1], activation='relu')

x = ValueTensor(np.random.randn(4, 2))  # batch of 4, 2 features each
pred = mlp(x)
loss = pred.mean()
loss.backward()

# Update parameters manually (SGD)
lr = 0.01
for p in mlp.parameters():
    p.data -= lr * p.grad
    p.zero_grad()
```

---

## Visualizing the Computation Graph
```python
from visualize import draw_dot

dot = draw_dot(loss)
dot.render('graph', format='svg', cleanup=False)
```
- Nodes show operation, shape, summary of `data` and `grad`
- Colored backgrounds indicate operation type
- Graphs can be large, so summarization avoids clutter

---

## Roadmap
- [x] Tensor support
- [x] Broadcasting-safe gradients
- [x] Modular MLP + training loop
- [x] Graphviz-based visualization
- [ ] Add more optimizers (Adam, RMSprop)
- [ ] Save/load models
- [ ] BatchNorm, Dropout layers
- [ ] Jupyter demo notebooks
- [ ] CLI trainer for simple datasets (e.g., XOR, MNIST subset)

---

## Contributing
This project is educational and evolving. Feel free to fork, suggest improvements, or open PRs for any enhancements or fixes!

---

## License
MIT License

---

## Credits
- Inspired by [micrograd](https://github.com/karpathy/micrograd)
- Built to extend the same philosophy to tensors, broadcasting, and modular neural networks.

