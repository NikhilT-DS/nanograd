import numpy as np

class Optimizers:
    def __init__(self, params, lr = 1e-5):
        self.params = params
        self.lr = lr
    
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

class SGD(Optimizers):
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad
    def __repr__(self):
        return f"SGD, learning rate {self.lr}"

class SGDMomentum(Optimizers):
    def __init__(self, params, momentum, lr = 1e-5):
        self.params = params
        self.momentum = momentum
        self.lr = lr
        self.velocities = [np.zeros_like(p.data, dtype=float) for p in self.params]
    
    def step(self):
        for i, p in enumerate(self.params):
            self.velocities[i] = self.momentum * self.velocities[i] + self.lr * p.grad
            p.data -= self.velocities[i]
    def __repr__(self) -> str:
        return f"SGD with momentum {self.momentum}, lr {self.lr}"

class Adam(Optimizers):
    def __init__(self, params, lr=1e-5, betas= (0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = [np.zeros_like(p.data, dtype=float) for p in self.params] # first moment
        self.v = [np.zeros_like(p.data, dtype=float) for p in self.params] # second moment
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def __repr__(self) -> str:
        return f"Adam optimizer, lr {self.lr}"