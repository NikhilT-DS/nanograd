# Nanograd: A Minimal Autograd Engine for Tensors

**Nanograd** is a lightweight, NumPy-based autodiff engine inspired by Karpathy's [micrograd](https://github.com/karpathy/micrograd), but extended to support **tensors**, **broadcasting**, **matrix operations**, and **modular neural networks**.

This library is built from scratch to provide a clear, minimal implementation of reverse-mode automatic differentiation on multidimensional arrays (tensors). It's useful for educational purposes, debugging autograd mechanics, and building intuition for how frameworks like PyTorch and JAX work internally.

---

## Features
- Tensor support: Operate on N-dimensional arrays using NumPy
- Reverse-mode autodiff (backpropagation)
- Broadcasting-safe gradients
- Basic math operations: `+`, `-`, `*`, `/`, `**`, `@`
- Reductions: `sum`, `mean`, `max`
- Nonlinearities: `relu`, `tanh`, `sigmoid`, `exp`, `log`, `softmax`
- Losses: `mse`, `cross_entropy`, `binary_cross_entropy`
- Shape ops: `reshape`, `transpose`
- Modular MLP and Linear layers
- Optimizers: `SGD`, `SGDMomentum`, `Adam`
- Visualizer: Graphviz-based computation graph
  - Color-coded gradients by magnitude
  - Node summaries (shape, mean/std/min/max)
  - Operation-based color themes
- Example notebooks: regression + classification with California Housing and Iris datasets

---

## Installation
```bash
pip install numpy graphviz scikit-learn
```

To visualize graphs, also install:
```bash
sudo apt install graphviz        # or brew install graphviz on macOS
```

---

## Usage Examples

### Forward + Backward on a Simple Graph
```python
from engine import ValueTensor

x = ValueTensor([[1.0, 2.0]])
w = ValueTensor([[2.0], [3.0]])
b = ValueTensor([[1.0]])

out = x @ w + b
loss = out.relu().sum()
loss.backward()

print("Output:", out.data)
print("Gradient w.r.t w:", w.grad)
```

### Train an MLP for Regression
```python
from model import MLP
from optim import SGD
from objective import mse

model = MLP([8, 32, 1], activation='relu')
optimizer = SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    preds = model(X_train)              # ValueTensor
    loss = mse(y_train, preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Visualization

```python
from visuals import draw_dot

# Visualize any ValueTensor graph (e.g., loss)
dot = draw_dot(loss)
dot.render('graph', format='svg')
```

- Nodes summarize tensor shape and gradient stats
- Gradients color-coded by magnitude
- Works well even for large MLPs

---

## Datasets and Notebooks

See `nanograd.ipynb` for:
- ✅ Regression on California Housing
- ✅ Classification on Iris Dataset
- ✅ Cross-entropy loss + softmax
- ✅ Training loop with SGD optimizer
- ✅ Graph visualization of computation trace

---

## Roadmap
- [x] Tensor engine with broadcasting support
- [x] MLPs and optimizer integration
- [x] Visual computation graph with color encoding
- [x] Cross-entropy for classification
- [x] Regression/classification examples
- [ ] Add Sequential and Dropout layers
- [ ] Add Trainer class with metric tracking
- [ ] Add model save/load functionality
- [ ] Add more interactive visualizations (Streamlit / live graph updates)

---

## Contributing
PRs and ideas are welcome! This is an open-ended project intended for learning and exploration.

---

## License
MIT License

---

## Credits
- Inspired by [micrograd](https://github.com/karpathy/micrograd)
- Built to explore and extend autograd to tensors, models, and visual computation

